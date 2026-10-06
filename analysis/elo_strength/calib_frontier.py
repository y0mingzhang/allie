"""Per-cell frontier: each cell's Elo gaps (accuracy, blunder rate) and cross-entropy gap at a constant beta,
for the bot's think-time-capped rung mixture and for its top rung on every position: how close any beta,
let alone one rule, can bring a cell to human strength under the cross-entropy bar.

python calib_frontier.py CALIB_DIR --ladder ... --extra ... --sim S --cells 3:2400,3:2600 --out frontier.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import calib_cells as cc
import calib_fit as cf
import calib_unified as cu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--ladder", required=True)
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--sim", type=float, default=0.0045)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cc.MARGIN = a.margin
    d = Path(a.dir)
    A = cu.load(d, a.ladder.split(","), a.extra, a.sim)
    human = cf.summarize(cf.table(np.load(d / "positions-human.npz")["meta"], cf.load_mpv(d / "mpv-human"), {}, {}), [])
    top = dict(A, W=np.eye(A["W"].shape[1])[np.full(len(A["f"]), A["W"].shape[1] - 1)])
    out = {}
    for c in a.cells.split(","):
        f, b = map(int, c.split(":"))
        sel = (A["f"] == f) & (A["bin"] == b)
        rows = []
        for beta in cu.G:
            r = {"beta": float(beta)}
            for name, B in (("capped", A), ("top", top)):
                st = cu.cell_stats(B, cu.mixed(B, np.full(len(B["f"]), beta)), sel, human)[f, b]
                r[name] = [float(x) for x in st]
            rows.append(r)
        out[f"{cf.FORMATS[f]}/{b}"] = rows
        for name in ("capped", "top"):
            ok = [r for r in rows if r[name][4] - 1.96 * r[name][5] <= 0]
            best = min(ok, key=lambda r: abs(r[name][0]) + abs(r[name][2])) if ok else None
            print(c, name, "best under the bar:", best and (round(best["beta"], 2), [round(x, 4) for x in best[name]]))
    (d / a.out).write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
