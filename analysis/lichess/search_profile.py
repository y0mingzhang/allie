"""Where a search leaf's time goes on the fast CPU backend: allie.lichess.tree's Coverage and KL searches at a
live game's position, timed around the oracle (tree.Nodes), its per-leaf cache copies (Nodes.fast) and the C++
step (fast.Fast.step, with its per-phase profile), against plain steps of 1, 8 and 24 independent tokens (the
forward's own cost per token at those batch sizes) and the weight bytes a step streams.

usage: search_profile.py --model DIR --pgn FILE [--ply 40] [--threads 4] [--searchers coverage,kl]
       [--budgets 32,128,1024] [--views true] [--backend fast|rust] [--native] [--cprofile] [--out FILE]
One JSON line per (searcher, budget) with ms per leaf, its parts and the phases; --out appends them. --native: the
all-Rust loop (treers.Coverage / KL through the model's Server; Rust backend), the steps counted by the Server.
"""

import argparse
import cProfile
import io
import json
import pstats
import sys
import time

import chess.pgn
import numpy as np
import torch

from allie.lichess import tree, treers
from allie.lichess.engine import Engine, Game
from allie.lichess.model import Cache, Model, step

PHASES = (
    "start",
    "cnn",
    "rows",
    "smear",
    "norm",
    "qkv",
    "rotary",
    "attn",
    "o",
    "norm2",
    "router",
    "topk",
    "group",
    "up",
    "down",
    "hnorm",
    "head",
)  # fast.py's enum order


def read_pgn(path, ply):
    g = chess.pgn.read_game(open(path))
    h = g.headers
    base, inc = map(int, h["TimeControl"].split("+"))
    speed = next(
        (
            s
            for s in ("bullet", "blitz", "rapid", "classical")
            if s in h.get("Event", "").lower()
        ),
        "rapid",
    )
    moves, clocks, node = [], [], g
    while node.variations:
        node = node.variation(0)
        moves.append(node.move.uci())
        clocks.append(None if node.clock() is None else int(node.clock()))
    return dict(
        white=int(h["WhiteElo"]),
        black=int(h["BlackElo"]),
        base=base,
        inc=inc,
        speed=speed,
        moves=moves[:ply],
        clocks=clocks[:ply],
        site=h.get("Site", path),
    )


def setup(model, g):
    engine = Engine(model)
    game = Game(engine, g["white"], g["black"], g["base"], g["inc"], g["speed"])
    moves, clocks = g["moves"], g["clocks"]
    for k in range(1, len(moves) + 1):
        j = k - 1
        own, other = clocks[j], clocks[j - 1] if j else None
        wt, bt = (own, other) if j % 2 == 0 else (other, own)
        game.update(moves[:k], wt, bt)
    game.sync()
    return engine, game


def weight_bytes(m):
    """Bytes a step reads of the matrices: every non-expert matrix once, each routed expert the batch touches."""
    size = lambda k: (
        m.w[k].numel() * m.w[k].element_size()
        + (m.scales[k].numel() * m.scales[k].element_size() if k in m.scales else 0)
    )
    rows = ("embed", "embed2", "value_embed", "cos", "sin")
    dense = sum(
        size(k)
        for k in m.w
        if k.split(".")[-1] not in ("up", "down") and not k.startswith(rows)
    )
    experts = sum(size(k) for k in m.w if k.split(".")[-1] in ("up", "down"))
    return dense, experts


class Meter:
    """Wraps the oracle, its cache copies and the C++ step with timers and counters."""

    def __init__(self, model):
        self.model, self.t = model, dict(oracle=0.0, fast=0.0, step=0.0)
        self.n = dict(calls=0, leaves=0, steps=0, items=0)
        self.sizes = []
        self._orig = (tree.Nodes.__call__, tree.Nodes.fast, model.fast.step)
        me = self

        def call(self_, handles):
            t0 = time.perf_counter()
            z = me._orig[0](self_, handles)
            me.t["oracle"] += time.perf_counter() - t0
            me.n["calls"] += 1
            me.n["leaves"] += len(handles)
            return z

        def fast(self_, ids, chunk=8):
            t0 = time.perf_counter()
            z = me._orig[1](self_, ids, chunk)
            me.t["fast"] += time.perf_counter() - t0
            return z

        def stp(items):
            t0 = time.perf_counter()
            z = me._orig[2](items)
            me.t["step"] += time.perf_counter() - t0
            me.n["steps"] += 1
            me.n["items"] += len(items)
            me.sizes.append(len(items))
            return z

        tree.Nodes.__call__, tree.Nodes.fast, model.fast.step = call, fast, stp

    def close(self):
        tree.Nodes.__call__, tree.Nodes.fast, self.model.fast.step = self._orig


def plain_steps(m, game, batches=(1, 8, 24), reps=5):
    """Median ms of a step of b independent one-token items on copies of the game's cache: the forward's cost per
    token at batch b (what a perfectly batched leaf would cost)."""
    out = {}
    n0 = game.cache.n
    legal = [mv.uci() for mv in game.board.legal_moves]
    for b in batches:
        caches = []
        for _ in range(b):
            c = Cache(m, n0 + 8)
            c.k[:, :, :n0], c.v[:, :, :n0], c.e[:n0] = (
                game.cache.k[:, :, :n0],
                game.cache.v[:, :, :n0],
                game.cache.e[:n0],
            )
            c.n = n0
            caches.append(c)
        items = []
        for j, c in enumerate(caches):
            mv = legal[j % len(legal)]
            tok = tree.MOVE_START + tree.MOVES.index(mv)
            board = (
                np.frombuffer(tree.advance(game.boards[-1], tok), np.uint8)
                .reshape(1, 68)
                .copy()
            )
            feats = torch.tensor(game.features()[-1:], dtype=torch.float32)
            items.append((c, torch.tensor([tok]), feats, torch.tensor(board)))
        times = []
        for _ in range(reps):
            for c in caches:
                c.n = n0
            t0 = time.perf_counter()
            step(m, items)
            times.append(1000 * (time.perf_counter() - t0))
        out[b] = round(float(np.median(times)), 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pgn", required=True)
    p.add_argument("--ply", type=int, default=40)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--backend",
        default="fast",
        choices=("fast", "rust"),
        help="the C++ kernels or the Rust engine",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        help="torch's own threads (default: --threads, as the bot sets them)",
    )
    p.add_argument(
        "--native",
        action="store_true",
        help="the all-Rust search loop (treers; the Rust backend)",
    )
    p.add_argument("--searchers", default="coverage,kl")
    p.add_argument("--budgets", default="32,128,1024")
    p.add_argument("--views", default="true")
    p.add_argument("--batches", default="1,8,24", help="plain-step batch sizes to time")
    p.add_argument("--cprofile", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(a.torch_threads or a.threads)
    t0 = time.perf_counter()
    m = Model(a.model, "cpu", torch.bfloat16, None, True, a.backend, a.threads)
    assert m.fast is not None, "fast backend needed"
    g = read_pgn(a.pgn, a.ply)
    engine, game = setup(m, g)
    dense, experts = weight_bytes(m)
    base = dict(
        model=str(a.model),
        game=g["site"],
        ply=len(g["moves"]),
        tokens=len(game.tokens),
        backend=a.backend,
        threads=m.fast.threads,
        torch_threads=torch.get_num_threads(),
        cpu=open("/proc/cpuinfo")
        .read()
        .split("model name")[1]
        .split("\n")[0]
        .split(":")[1]
        .strip(),
        load_s=round(time.perf_counter() - t0, 1),
        config=dict(
            layers=m.layers,
            width=m.width,
            heads=m.heads,
            experts=m.config["experts"],
            topk=m.topk,
            keep=m.keep,
        ),
        weight_mb=dict(dense=dense >> 20, experts=experts >> 20),
    )
    base["plain_step_ms"] = plain_steps(m, game, tuple(map(int, a.batches.split(","))))
    srv = getattr(m.fast, "server", None)
    assert not a.native or srv is not None, "--native needs the Rust backend"
    base["native"] = a.native
    print(json.dumps(base), flush=True)
    views = tuple(a.views.split(","))
    rows = []
    for name in a.searchers.split(","):
        for budget in map(int, a.budgets.split(",")):
            mod = treers if a.native else tree
            searcher = (
                mod.Coverage(views=views) if name == "coverage" else mod.KL(views=views)
            )
            meter = Meter(m)
            before = srv.stats() if srv else None
            m.fast.profile(True)  # resets the phase clocks
            prof = cProfile.Profile() if a.cprofile else None
            t1 = time.perf_counter()
            if prof:
                prof.enable()
            res = searcher(game, budget)
            if prof:
                prof.disable()
            wall = time.perf_counter() - t1
            phases = m.fast.profile(False)
            meter.close()
            if srv:  # every step runs on the server: its counts (the native loop makes no Python calls)
                after = srv.stats()
                d = {
                    k: after[k] - before[k]
                    for k in ("steps", "items", "waited", "wait_s", "joined")
                }
                meter.n["steps"], meter.n["items"] = d["steps"], d["items"]
                if a.native:
                    meter.n["leaves"], meter.n["calls"] = (
                        game.last_search["evaluated"],
                        d["steps"] // len(views),
                    )
            leaves = max(meter.n["leaves"], 1)
            ms = lambda s: round(1000 * s / leaves, 3)  # noqa: E731
            row = dict(
                searcher=name,
                budget=budget,
                views=list(views),
                leaves=meter.n["leaves"],
                calls=meter.n["calls"],
                steps=meter.n["steps"],
                items_per_step=round(meter.n["items"] / max(meter.n["steps"], 1), 1),
                leaves_per_call=round(meter.n["leaves"] / max(meter.n["calls"], 1), 1),
                gather=d if srv else None,
                wall_ms=round(1000 * wall, 1),
                ms_per_leaf=ms(wall),
                oracle_ms_per_leaf=ms(meter.t["oracle"]),
                copies_ms_per_leaf=ms(meter.t["fast"] - meter.t["step"]),
                step_ms_per_leaf=ms(meter.t["step"]),
                tree_ms_per_leaf=ms(wall - meter.t["oracle"]),
                bookkeeping_ms_per_leaf=ms(meter.t["oracle"] - meter.t["fast"]),
                phase_ms_per_leaf={
                    k: ms(v)
                    for k, v in zip(
                        PHASES, phases.values() if isinstance(phases, dict) else phases
                    )
                    if v
                },
                moves=None if res is None else len(res[0]),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            if prof:
                s = io.StringIO()
                pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(28)
                print(s.getvalue()[:6000], flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(base | r) + "\n")
    engine.close()


if __name__ == "__main__":
    sys.exit(main())
