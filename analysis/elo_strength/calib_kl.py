"""The calibration's KL export: allie.lichess.calibration's search (kl.grow with GROW over VIEWS, the root's prior read
under the game's own header) on GPU with a training checkpoint (allie.search.moe_oracle), on calib_positions.py's
positions, read by READ at every rung of LADDER up to --budget. A rung's tree is the grown tree's nodes of cost <= b.
Writes calib_cells chunks (index, legal, prior, heads, offsets; per rung cov_x{b}k, the values' log odds with moves
left unexpanded at the root's own value, and cov_q{b}k in W - L units) for calib_unified.py --extra (ladder tags
{b}kx). Run each of calib_kl_subsets.py's subsets at its own budget.

python calib_kl.py OUT --pos positions-human.npz --subset subset-1024.npy --budget 1024 --checkpoint CKPT
    [--source SRC] [--batch 64 --chunk 256] [--shard i --shards n]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from allie.lichess import calibration as C
from allie.lichess.tokens import header
from allie.search import kl
from allie.search.moe_oracle import MoEOracle
from allie.search.native import from_prefix

SLOTS = 1 << 17  # the oracle's key/value slots and node rows


def under(spec, tokens, feats):
    """A position read under a view spec, as allie.lichess.tree.View reads the game."""
    t, f = tokens.copy(), feats
    for part in [] if spec == "true" else spec.split("/"):
        if part == "noclock":
            f = np.full_like(f, -1)
        elif part == "swap":
            t[3:11] = np.r_[t[7:11], t[3:7]]
        elif part.startswith("tc"):
            t[1:3] = header(*map(int, part[2:].split("+")), 0, 0)[1:3]
        elif part.startswith("r") and part[1:].isdigit():
            t[3:11] = header(None, None, int(part[1:]), int(part[1:]))[3:11]
        else:
            raise ValueError(f"unknown view part {part!r}")
    return t, f


def search(oracle, tokens, feats, rungs):
    """One batch of positions: calib_cells arrays at every rung, each position's legal-move count, and its leaves."""
    oracle.reset()
    views = [oracle.handles(*zip(*[under(v, t, f) for t, f in zip(tokens, feats)]), "zero") for v in C.VIEWS]
    prior = oracle.prefill(tokens, feats)[0]
    F = kl.Forest(views, [from_prefix(t) for t in tokens], [len(t) for t in tokens], oracle.capacity, prior=prior)
    kl.grow(F, rungs[-1], **C.GROW)
    assert F.size < F.cap, "the forest filled up: lower --batch"
    E, s = len(F.rootw), C.READ["squash"]  # the roots' edges come first
    kid, cost = F.ekid[:E], F.cost[: F.size]
    unsearched = np.repeat(np.arctanh(s * np.clip(F.value[: F.n], -1, 1)), F.count[: F.n])
    out = dict(legal=F.etok[:E].astype(np.int16), prior=F.ep[:E], heads=F.heads)
    for b in rungs:
        keep = cost <= b
        V = kl.backup(F.view(), keep, **C.READ)[0]
        x = np.where((kid >= 0) & keep[np.maximum(kid, 0)], -V[np.maximum(kid, 0)], unsearched)
        out[f"cov_x{b}k"], out[f"cov_q{b}k"] = x, np.clip(np.tanh(x) / s, -1, 1)
    return {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in out.items()}, F.count[: F.n], F.spent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--pos", required=True, help="calib_positions.py's positions")
    p.add_argument("--subset", help="indices into them (default all)")
    p.add_argument("--budget", type=int, required=True, help="the largest rung, leaves a position")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", help="a pre-package checkpoint's frozen source")
    p.add_argument("--batch", type=int, help="positions a forest (default: as many as the oracle's slots hold)")
    p.add_argument("--chunk", type=int, help="positions an output file (default 4 batches)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert a.budget in C.LADDER, C.LADDER
    rungs = [b for b in C.LADDER if b <= a.budget]
    z = np.load(a.pos)
    off, tokens, feats = z["offsets"], z["tokens"], z["feats"]
    subset = np.load(a.subset) if a.subset else np.arange(len(off) - 1)
    longest, nv = int(np.diff(off)[subset].max()), len(C.VIEWS)
    a.batch = a.batch or min(96, SLOTS // (nv * (longest + a.budget) + longest))  # views' prefills and nodes, prior's
    a.chunk = a.chunk or 4 * a.batch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = {k: v for k, v in vars(a).items() if k not in ("out", "shard", "shards")}
    run |= dict(grow=C.GROW, read=C.READ, views=list(C.VIEWS))
    if (f := out / "args.json").exists():
        assert json.loads(f.read_text()) == run, f"{f}: chunks of another run"
    else:
        (tmp := out / f"args.{a.shard}.tmp").write_text(json.dumps(run, indent=1))
        tmp.replace(f)
    oracle = MoEOracle(a.checkpoint, a.source, slots=SLOTS, rows=SLOTS)
    for lo in range(0, len(subset), a.chunk)[a.shard :: a.shards]:
        dst = out / f"{lo:06d}.npz"
        if dst.exists():
            continue
        t0, idx = time.monotonic(), subset[lo : lo + a.chunk]
        parts, counts, leaves = zip(*[
            search(oracle, [tokens[off[i] : off[i + 1]].astype(np.int64) for i in rows],
                   [feats[off[i] : off[i + 1]].astype(np.float32) for i in rows], rungs)
            for rows in (idx[s : s + a.batch] for s in range(0, len(idx), a.batch))
        ])  # fmt: skip
        res = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        tmp = dst.with_suffix(".partial.npz")
        np.savez(tmp, index=idx, budgets=np.array(rungs), offsets=np.r_[0, np.cumsum(np.concatenate(counts))], **res)
        tmp.replace(dst)
        print(json.dumps(dict(chunk=lo, n=len(idx), seconds=round(time.monotonic() - t0, 1),
                              leaves=float(np.concatenate(leaves).mean()))), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
