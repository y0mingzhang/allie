"""GPU pass of the calibration eval. Per position: the legal policy and root heads (think time, W/D/L);
coverage search (allie.search with the bot's Allie 2.0 calibration) keeping per-move Q and its
calibrated distribution; Allie-paper MCTS (allie.search's "allie" tree) keeping per-move Q and visits.
Runs in the search-bench environment on the training model (moe-oracle); see run-calib.sbatch.

python calib_search.py POSITIONS.npz OUT_DIR --shard i --shards n
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

H = Path("/home/yimingz3/src/allie/results/recipe10x/maia3-bench/search")
sys.path[:0] = [str(H / "moe-oracle"), "/home/yimingz3/src/allie"]
from oracle import MoEOracle
from search import Search
from search.native import from_prefix, load

CALIBRATION = Path(
    "/home/yimingz3/src/allie-wt-elo-pkg/src/allie/lichess/calibration-allie-v3.0.json"
)


class Capture:
    """The oracle, keeping each batch's root logits."""

    def __init__(self, oracle):
        self.oracle = oracle

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def handles(self, *a):
        h = self.oracle.handles(*a)
        self.root = np.asarray(h.root_logits)
        return h


def mcts(search, rows, feats, budget, cpuct=1.25):
    """Search._batch's "allie" branch, keeping the root visit counts it discards."""
    search.oracle.reset()
    bridge = search.oracle.handles([r["prefix"] for r in rows], feats, "predicted")
    root = np.asarray(bridge.root_logits)
    sims = [0 if len(r["legal"]) == 1 else budget for r in rows]
    tree = load().Allie([r["prefix"] for r in rows], root, sims, [cpuct] * len(rows))
    tree.first_prior = tree.preserve_depth = True
    Search._advance(tree, bridge)
    out = []
    for r, (moves, counts, values, _) in zip(rows, tree.summaries()):
        np.testing.assert_array_equal(moves, np.array(r["legal"]) - 378)
        out.append((np.array(counts, np.int32), np.array(values, np.float32)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("out")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--mcts", default="8,32,128,512", help="MCTS budgets ('' for none)")
    p.add_argument("--coverage", default="8,32,128", help="coverage budgets")
    p.add_argument("--cells", default="", help="only these format:bin cells, e.g. 3:2400,3:2600 (format 0-3)")
    p.add_argument("--batch", type=int, default=128, help="rows per search batch (large budgets: fewer)")
    a = p.parse_args()
    z = np.load(a.positions)
    off, tokens, feats, meta = z["offsets"], z["tokens"], z["feats"], z["meta"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    par = json.loads(CALIBRATION.read_text())
    for b in (
        8,
        32,
    ):  # as allie.lichess.tree: budgets without their own policy use 128's
        par["budget_policies"].setdefault(str(b), par["budget_policies"]["128"])
    for b in [int(x) for x in a.coverage.split(",") if x]:  # larger budgets: 256's (beta tilts use only Q)
        par["budget_policies"].setdefault(str(b), par["budget_policies"]["256"])
    t0 = time.monotonic()
    oracle = Capture(MoEOracle(a.checkpoint, a.source, slots=1 << 17, rows=1 << 17))
    search = Search(oracle, batch_size=a.batch, threads=8, calibration=par)
    keep = np.arange(len(meta))
    if a.cells:
        want = {tuple(map(int, c.split(":"))) for c in a.cells.split(",")}
        keep = np.array([i for i in keep if (int(meta[i, 2]), int(meta[i, 3])) in want])
    print(f"startup {time.monotonic() - t0:.0f}s", flush=True)
    chunks = range(0, len(keep), a.chunk)
    for lo in list(chunks)[a.shard :: a.shards]:
        dst = out / f"{lo:06d}.npz"
        if dst.exists():
            continue
        idx = keep[lo : lo + a.chunk]
        rows, fts = [], []
        for i in idx:
            prefix = tokens[off[i] : off[i + 1]].astype(np.int64)
            rows.append(
                dict(
                    prefix=prefix.tolist(),
                    cell=int(meta[i, 9]),
                    legal=from_prefix(prefix).legal(),
                )
            )
            fts.append(feats[off[i] : off[i + 1]].astype(np.float32))
        res, timing = {}, {}
        for b in [int(x) for x in a.coverage.split(",") if x]:
            t = time.monotonic()
            r, heads = [], []
            for s in range(0, len(rows), a.batch):
                r += search._batch(
                    rows[s : s + a.batch],
                    fts[s : s + a.batch],
                    "coverage",
                    b,
                    "predicted",
                    False,
                    0.9,
                    2.0,
                    1.25,
                )
                heads.append(oracle.root[:, 2350:2416].astype(np.float32))
            res[f"cov_q{b}"] = np.concatenate([x["values"] for x in r]).astype(
                np.float32
            )
            res[f"cov_p{b}"] = np.concatenate([x["probabilities"] for x in r]).astype(
                np.float32
            )
            res["prior"] = np.concatenate([x["legal_prior"] for x in r]).astype(
                np.float32
            )
            res["heads"] = np.concatenate(heads)
            timing[f"cov{b}"] = time.monotonic() - t
        for b in [int(x) for x in a.mcts.split(",") if x]:
            t = time.monotonic()
            r = []
            for s in range(0, len(rows), a.batch):
                r += mcts(search, rows[s : s + a.batch], fts[s : s + a.batch], b)
            res[f"mcts_n{b}"] = np.concatenate([c for c, _ in r])
            res[f"mcts_q{b}"] = np.concatenate([q for _, q in r])
            timing[f"mcts{b}"] = time.monotonic() - t
        legal = [np.array(r["legal"], np.int16) for r in rows]
        tmp = dst.with_suffix(".partial.npz")
        np.savez(tmp, index=idx, legal=np.concatenate(legal), offsets=np.cumsum([0] + [len(x) for x in legal]),
                 seconds=json.dumps(timing), **res)  # fmt: skip
        tmp.replace(dst)
        print(
            json.dumps(
                dict(
                    chunk=lo, n=len(idx), **{k: round(v, 1) for k, v in timing.items()}
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
