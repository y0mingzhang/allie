"""Per-leaf search cost under load, as the live bot sees it: G concurrent games (threads) on one shared Engine,
each searching mid-game positions of live PGNs with a fixed budget; the time per leaf across games and the engine's
batch sizes.

usage: search_load.py --model DIR --pgns 'GLOB' [--games 10] [--rounds 3] [--budget 128] [--searcher coverage|kl]
       [--kw JSON] [--threads 4] [--torch-threads 1] [--ply 30] [--backend fast|rust] [--native] [--max-items N] [--min-items N]
       [--gather SECONDS] [--moves-first 0|1] [--out FILE]
--native: the all-Rust loop (treers through the model's Server, Rust backend); --max-items / --min-items / --gather set
the Server's merge policy; the Server's gather statistics are reported. Also the plain-move latency a game sees under
the load (an extra game playing one move every 100 ms: its one-token sync queued behind the leaf batches, median and
p90 ms) and the process's peak RSS.
"""

import argparse
import glob
import json
import resource
import sys
import threading
import time

import numpy as np
import torch

from allie.lichess import tree, treers
from allie.lichess.engine import Engine, Game
from allie.lichess.model import Model

from search_profile import read_pgn


def load_game(engine, g):
    game = Game(engine, g["white"], g["black"], g["base"], g["inc"], g["speed"])
    moves, clocks = g["moves"], g["clocks"]
    for k in range(1, len(moves) + 1):
        j = k - 1
        own, other = clocks[j], clocks[j - 1] if j else None
        wt, bt = (own, other) if j % 2 == 0 else (other, own)
        game.update(moves[:k], wt, bt)
    game.sync()
    return game


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pgns", required=True)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--searcher", default="coverage")
    p.add_argument("--kw", default="{}", help="the searcher's keywords (JSON), e.g. views, grow, read")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--backend",
        default="fast",
        choices=("fast", "rust"),
        help="the C++ kernels or the Rust engine",
    )
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--ply", type=int, default=30)
    p.add_argument("--native", action="store_true")
    p.add_argument("--max-items", type=int)
    p.add_argument("--min-items", type=int)
    p.add_argument("--gather", type=float)
    p.add_argument("--moves-first", type=lambda s: bool(int(s)), choices=(False, True))
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(a.torch_threads)
    m = Model(a.model, "cpu", torch.bfloat16, None, True, a.backend, a.threads)
    assert m.fast is not None
    srv = getattr(m.fast, "server", None)
    assert not a.native or srv is not None, "--native needs the Rust backend"
    if srv:
        for k in ("max_items", "min_items", "gather", "moves_first"):
            if getattr(a, k) is not None:
                setattr(srv, k, getattr(a, k))
    engine = Engine(m)
    files = sorted(glob.glob(a.pgns))
    long = [
        f for f in files if len(read_pgn(f, 10**6)["moves"]) >= a.ply + 2 * a.rounds + 4
    ]  # mid-game, not over
    assert len(long) >= a.games, f"{len(long)} long enough PGNs for {a.games} games"
    games = []
    for f in long:  # a position with a choice to make (not over, not a forced move)
        g = load_game(engine, read_pgn(f, a.ply + 2 * len(games)))
        if not g.board.is_game_over() and g.board.legal_moves.count() > 1:
            games.append(g)
        if len(games) == a.games:
            break
    assert len(games) == a.games, f"{len(games)} usable positions"
    mod = treers if a.native else tree
    kw = json.loads(a.kw)
    searcher = mod.Coverage(**kw) if a.searcher == "coverage" else mod.KL(**kw)
    leaves, inner = [], tree.Nodes.__call__

    def counted(self_, handles):
        leaves.append(len(handles))
        return inner(self_, handles)

    tree.Nodes.__call__ = counted
    sizes, orig = [], m.fast.step

    def stp(items):
        sizes.append(len(items))
        return orig(items)

    m.fast.step = stp
    per_search = []

    def run(game):
        for _ in range(a.rounds):
            t0 = time.perf_counter()
            res = searcher(game, a.budget)
            per_search.append((time.perf_counter() - t0, res is not None))
            if a.native:
                leaves.append(game.last_search["evaluated"])

    mover = load_game(
        engine, read_pgn(long[a.games % len(long)], a.ply + 1)
    )  # plays a move every 100 ms meanwhile
    move_ms, stop, clock = [], threading.Event(), [mover.base, mover.base]

    def play():
        while not stop.is_set() and not mover.board.is_game_over():
            clock[len(mover.moves) % 2] -= 1
            mover.update(
                mover.moves + [next(iter(mover.board.legal_moves)).uci()], *clock
            )
            t0 = time.perf_counter()
            mover.sync()  # one new token, queued behind the searches' leaf batches
            move_ms.append(1000 * (time.perf_counter() - t0))
            stop.wait(0.1)

    for g in games:  # warm every game's cache and the kernels
        searcher(g, 8)
    leaves.clear(), sizes.clear()
    before = srv.stats() if srv else None
    if srv:
        srv.sizes()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=run, args=(g,)) for g in games]
    for t in threads:
        t.start()
    threading.Thread(target=play).start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    stop.set()
    n_leaves = sum(leaves)
    secs = np.array([s for s, _ in per_search])
    gather = None
    if srv:  # every step runs on the server: its sizes and gather counts
        after = srv.stats()
        sizes = srv.sizes()
        gather = {
            k: after[k] - before[k]
            for k in ("steps", "items", "requests", "waited", "wait_s", "joined")
        }
        gather.update(
            max_items=srv.max_items,
            min_items=srv.min_items,
            gather_s=srv.gather,
            moves_first=srv.moves_first,
        )
    row = dict(model=str(a.model), backend=a.backend, native=a.native, searcher=a.searcher, kw=kw, budget=a.budget, games=a.games, rounds=a.rounds,
               threads=m.fast.threads, torch_threads=torch.get_num_threads(), server=gather,
               move_ms=dict(n=len(move_ms), median=round(float(np.median(move_ms)), 1), p90=round(float(np.percentile(move_ms, 90)), 1)) if move_ms else None,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10,
               cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].split(":")[1].strip(),
               searches=len(per_search), leaves=n_leaves, wall_s=round(wall, 2),
               ms_per_leaf_throughput=round(1000 * wall / n_leaves, 3),  # the node's cost per leaf under load
               ms_per_leaf_latency=round(1000 * secs.sum() / n_leaves, 3),  # what one search waits per leaf
               search_s=dict(median=round(float(np.median(secs)), 2), p90=round(float(np.percentile(secs, 90)), 2)),
               steps=len(sizes), items_per_step=dict(mean=round(float(np.mean(sizes)), 1), p50=int(np.median(sizes)),
               p90=int(np.percentile(sizes, 90)), max=int(max(sizes))), forwards=engine.forwards)  # fmt: skip
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    engine.close()


if __name__ == "__main__":
    sys.exit(main())
