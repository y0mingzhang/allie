"""KL-regularized search (allie.search.kl) on the calibration positions with the annealed training checkpoint on
GPU (search_eff's harness: its MoEOracle and views). Each chunk's forest is saved whole (kl.Forest.dump),
loadable by search_eff's evaluate.load_forest; the tree of any budget b is its nodes of cost <= b.

python gpu_kl.py OUT [--subset subset.npy] --shard i --shards n --budget 4096 --own 4 --opp 0 --views 0:zero
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [
    "/home/yimingz3/src/allie-wt-seff/analysis/search_eff",
    str(Path(__file__).resolve().parents[2] / "src/allie/search"),
]
from harness import POS, SUBSET, MoEOracle, from_prefix, view_bridges  # noqa: E402
import kl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--subset", default=str(SUBSET))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=96)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--own", type=float, default=4.0)
    p.add_argument("--opp", type=float, default=0.0)
    p.add_argument("--soft", action="store_true")
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--root", type=float, default=0.0, help="kl.grow root: the output tilt beta that steers root weights")
    p.add_argument("--full", action="store_true", help="kl.grow full: every root move first")
    p.add_argument("--prior", default="", help="a view (e.g. 0) read at the roots only, for their policy and heads")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--g", type=float, default=0.125)
    p.add_argument("--width", type=int, default=4)
    p.add_argument(
        "--views",
        default="0:zero",
        help="harness.py views RATING[:CLOCK]; the first gives the policy",
    )
    p.add_argument(
        "--values",
        default="",
        help="the views whose W/D/L the search averages (default all)",
    )
    p.add_argument("--policy", default="0,0", help="the views whose policy the mover's and the opponent's nodes read")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True)
    a = p.parse_args()
    z = np.load(POS)
    off, tokens, feats = z["offsets"], z["tokens"], z["feats"]
    subset = np.load(a.subset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1) + "\n")
    views = a.views.split(",")
    values = [int(v) for v in a.values.split(",")] if a.values else None
    policy = tuple(int(v) for v in a.policy.split(","))
    t0 = time.monotonic()
    oracle = MoEOracle(a.checkpoint, a.source, slots=1 << 17, rows=1 << 17)
    print(f"startup {time.monotonic() - t0:.0f}s", flush=True)
    for lo in list(range(0, len(subset), a.chunk))[a.shard :: a.shards]:
        dst = out / f"{lo:06d}.npz"
        if dst.exists():
            continue
        t = time.monotonic()
        idx = subset[lo : lo + a.chunk]
        parts = []
        for s in range(0, len(idx), a.batch):
            rows = idx[s : s + a.batch]
            prefixes = [tokens[off[i] : off[i + 1]].astype(np.int64) for i in rows]
            fts = [feats[off[i] : off[i + 1]].astype(np.float32) for i in rows]
            oracle.reset()
            bridges = view_bridges(oracle, prefixes, fts, views)
            prior = view_bridges(oracle, prefixes, fts, [a.prior])[0].root_logits if a.prior else None
            F = kl.Forest(
                bridges,
                [from_prefix(q) for q in prefixes],
                [len(q) for q in prefixes],
                oracle.capacity,
                values,
                policy,
                prior=prior,
            )
            kl.grow(F, a.budget, a.own, a.opp, a.soft, a.kappa, a.k, a.g, a.width, a.root, a.full)
            parts.append(F.dump())
        n0 = np.cumsum([0] + [len(d["parent"]) for d in parts])
        r0 = np.cumsum([0] + [len(d["calls"]) for d in parts])
        cat = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
        cat["parent"] = np.concatenate(
            [
                np.where(d["parent"] >= 0, d["parent"] + n0[j], -1)
                for j, d in enumerate(parts)
            ]
        )
        cat["owner"] = np.concatenate([d["owner"] + r0[j] for j, d in enumerate(parts)])
        cat["roots"] = np.concatenate(
            [np.arange(len(d["calls"])) + n0[j] for j, d in enumerate(parts)]
        )
        tmp = dst.with_suffix(".partial.npz")
        np.savez(tmp, index=idx, **cat)
        tmp.replace(dst)
        print(json.dumps(dict(chunk=lo, n=len(idx), nodes=int(n0[-1]), seconds=round(time.monotonic() - t, 1),
                              calls=float(cat["calls"].mean()), leaves=float(cat["leaves"].mean()))), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
