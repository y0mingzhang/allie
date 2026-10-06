"""Pruned budgets for the KL view-tree export: per position the largest of --budgets the bot could search at --sim
seconds a leaf (P(think >= b sim + MARGIN, within the clock's tenth) >= --p; calib_cells.reach on search-human-ann2c's
think-time heads), the smallest budget otherwise. Writes disjoint index files OUT/subset-{b}.npy: grow each position's
forest to its budget once; every smaller budget is read from it.

python calib_kl_subsets.py CALIB_DIR OUT_DIR --sim 0.0045 [--margin 0.2] [--budgets 1024,2048,4096] [--p 0.01]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import calib_cells as cc
import calib_fit as cf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("out")
    p.add_argument("--sim", type=float, required=True)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--budgets", default="1024,2048,4096")
    p.add_argument("--p", type=float, default=0.01)
    a = p.parse_args()
    cc.MARGIN = a.margin
    d, out = Path(a.dir), Path(a.out)
    budgets = np.array([int(b) for b in a.budgets.split(",")])
    z = np.load(d / "positions-human.npz")
    meta, tokens, starts = z["meta"], z["tokens"], z["offsets"]
    top, cells = {}, {}
    for c in cf.chunks(d / "search-human-ann2c"):
        for n, i in enumerate(c["index"]):
            m, i = meta[int(i)], int(i)
            tok = int(tokens[starts[i] + 2])
            clock = float(m[7]) if m[7] >= 0 else None
            P = cc.reach(
                c["heads"][n, :63].astype(float),
                clock,
                tok - 10 if 10 <= tok <= 190 else 0,
                budgets * a.sim,
            )
            top[i] = budgets[max(np.flatnonzero(P >= a.p), default=0)]
            key = f"{cf.FORMATS[int(m[2])]}/{int(m[3])}"
            cells.setdefault(key, []).append(top[i])
    out.mkdir(parents=True, exist_ok=True)
    idx = np.array(sorted(top))
    b = np.array([top[i] for i in idx])
    for x in budgets:
        np.save(out / f"subset-{x}.npy", idx[b == x])
    summary = dict(sim=a.sim, margin=a.margin, p=a.p, n={int(x): int((b == x).sum()) for x in budgets},
                   cells={k: {int(x): int(np.sum(np.array(v) == x)) for x in budgets} for k, v in sorted(cells.items())})  # fmt: skip
    (out / "subsets.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["n"]))


if __name__ == "__main__":
    main()
