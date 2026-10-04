"""CPU fallback of calib_search.py on the bot's own code (allie.lichess Model and Tree): coverage and
Allie-paper MCTS at the given budgets for a subset of positions, in calib_search.py's chunk format.

python calib_search_cpu.py POSITIONS.npz OUT_DIR --select SEL.npy [--coverage 8,32] [--mcts 8,32]
    [--shard i --shards n --threads 8]
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from allie.data.vocab import MOVES
from allie.lichess.engine import Engine, Game
from allie.lichess.model import Cache, Model
from allie.lichess.tokens import HEADER, START, advance
from allie.lichess.tree import CALIBRATION, Tree
from allie.search import Search
from allie.search.native import from_prefix, load

from play import MODEL


class Position(Game):
    """A Game at a stored position: its exact tokens and causal clock features."""

    def __init__(self, engine, tokens, feats, cell):
        self.engine, self.tokens, self.feats, self._cell = (
            engine,
            [int(t) for t in tokens],
            feats,
            cell,
        )
        self.boards = [START] * HEADER
        for t in self.tokens[HEADER:]:
            self.boards.append(advance(self.boards[-1], t))
        self.moves = [MOVES[t - 378] for t in self.tokens[HEADER:]]
        self.inc = (
            -1
        )  # Tree reads game.inc only for imagined clocks; set from the header below
        self.cache, self.logits = Cache(engine.model), None

    def features(self, lo=0):
        return self.feats[lo:].tolist()


def run(game, method, budget, par):
    """Search._batch on the game's cache: (legal tokens, output dict, visits or None)."""
    z = game.sync()
    feats = np.array(game.features(), np.float32)
    legal = from_prefix(np.asarray(game.tokens)).legal()

    def go():
        tree = Tree(game, z, capacity=4 * budget + 256)
        s = Search(tree, threads=4, calibration=par)
        if method == "coverage":
            row = dict(prefix=list(game.tokens), cell=game._cell, legal=legal)
            return s._batch(
                [row], [feats], "coverage", budget, "predicted", False, 0.9, 2.0, 1.25
            )[0], None
        bridge = tree.handles([list(game.tokens)], [feats])
        t = load().Allie(
            [list(game.tokens)],
            bridge.root_logits,
            [0 if len(legal) == 1 else budget],
            [1.25],
        )
        t.first_prior = t.preserve_depth = True
        Search._advance(t, bridge)
        moves, counts, values, prior = t.summaries()[0]
        return dict(values=np.array(values), legal_prior=np.array(prior)), np.array(
            counts
        )

    out, visits = game.engine.run(go)
    return legal, out, visits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("out")
    p.add_argument("--select", required=True, help="npy of position indices")
    p.add_argument("--coverage", default="8,32")
    p.add_argument("--mcts", default="8,32")
    p.add_argument(
        "--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    z = np.load(a.positions)
    off, tokens, feats, meta = z["offsets"], z["tokens"], z["feats"], z["meta"]
    sel = np.load(a.select)
    cov = [int(b) for b in a.coverage.split(",") if b]
    mc = [int(b) for b in a.mcts.split(",") if b]
    par = json.loads(CALIBRATION.read_text())
    for b in cov:
        par["budget_policies"].setdefault(str(b), par["budget_policies"]["128"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = Engine(Model(MODEL))
    for lo in list(range(0, len(sel), a.chunk))[a.shard :: a.shards]:
        dst = out / f"c{lo:06d}.npz"
        if dst.exists():
            continue
        t0 = time.monotonic()
        idx = sel[lo : lo + a.chunk]
        res = {k: [] for k in ["legal", "prior"] + [f"cov_q{b}" for b in cov] + [f"cov_p{b}" for b in cov]
               + [f"mcts_q{b}" for b in mc] + [f"mcts_n{b}" for b in mc]}  # fmt: skip
        for i in idx:
            g = Position(
                engine,
                tokens[off[i] : off[i + 1]],
                feats[off[i] : off[i + 1]],
                int(meta[i, 9]),
            )
            inc = tokens[
                off[i] + 2
            ]  # increment token; Tree uses game.inc in imagined clocks
            from allie.data.vocab import INCREMENTS_ID

            g.inc = next(
                (int(k) for k, v in INCREMENTS_ID.items() if v == inc and k.isdigit()),
                None,
            )
            for b in cov:
                legal, o, _ = run(g, "coverage", b, par)
                res[f"cov_q{b}"].append(o["values"])
                res[f"cov_p{b}"].append(o["probabilities"])
            for b in mc:
                legal, o, n = run(g, "allie", b, par)
                res[f"mcts_q{b}"].append(o["values"])
                res[f"mcts_n{b}"].append(n)
            res["legal"].append(np.array(legal))
            res["prior"].append(o["legal_prior"])
        cat = {
            k: np.concatenate(v).astype(np.int16 if k == "legal" else np.float32)
            for k, v in res.items()
        }
        cat.update(
            {k: v.astype(np.int32) for k, v in cat.items() if k.startswith("mcts_n")}
        )
        np.savez(
            dst.with_suffix(".partial.npz"),
            index=idx,
            offsets=np.cumsum([0] + [len(x) for x in res["legal"]]),
            **cat,
        )
        dst.with_suffix(".partial.npz").replace(dst)
        print(
            f"{dst.name}: {len(idx)} positions, {time.monotonic() - t0:.0f} s",
            flush=True,
        )
    engine.close()


if __name__ == "__main__":
    main()
