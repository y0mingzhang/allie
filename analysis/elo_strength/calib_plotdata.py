"""Per-cell move accuracy and blunder rate (means, standard errors clustered by game) of the humans, the raw
policy and the calibrated bot as shipped (calib_cells.py --think picks: the rungs mixed by the bot's
think-time draws), for plotting the metrics themselves rather than Elo gaps; plus each time control's
slope of the metric against rating, the bot's over the humans'.

python calib_plotdata.py CALIB_DIR TABLE.json --extra DIR [--betas ...] --out plot.json
TABLE: calib_cells.py --think output (its picks); same --extra and --betas as that run.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import calib_cells as cc
import calib_fit as cf


def fit(x, y, se, log=False):
    """Weighted least-squares slope of y (log y) against x, per 1,000 rating points, and its SE."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    if log:
        y, se = np.log(y), se / y
    w = 1 / np.maximum(se, 1e-9) ** 2
    X = np.c_[np.ones(len(x)), (np.asarray(x, float) - 1700) / 1000]
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    return float((cov @ X.T @ (w * y))[1]), float(np.sqrt(cov[1, 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("table")
    p.add_argument("--coverage", default="search-human-ann2c")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--betas", default="0.5,1,2,3,4,6,8,12,16,20,24,32")
    p.add_argument("--sim", type=float, default=0.015)
    p.add_argument("--out", default="plot-capped.json")
    a = p.parse_args()
    cc.BETAS, cc.THINK, cc.SIM = (
        tuple(float(x) for x in a.betas.split(",")),
        True,
        a.sim,
    )
    cc.names()
    d = Path(a.dir)
    table = json.load(open(a.table))
    info, X, H = cc.rows(d, a.coverage, {}, [cc.extra(Path(e)) for e in a.extra])
    present = np.isfinite(X[:, :, 0]).sum(1)
    col = {n: j for j, n in enumerate(cc.NAMES)}
    cells = {}
    for f in range(4):
        for b in range(800, 2601, 200):
            sel = (info["f"] == f) & (info["bin"] == b)
            if not sel.any():
                continue
            sel &= present == present[sel].max()
            key = f"{cf.FORMATS[f]}/{b}"
            pick = table[key]["choice"] if key in table else "raw"
            g = info["game"][sel]
            row = dict(
                n=int(sel.sum()),
                games=len(np.unique(g)),
                elo=float(info["elo"][sel].mean()),
                choice=pick,
            )
            row |= metrics(
                g, human=H[sel], raw=X[sel, col["raw"], :2], calibrated=X[sel, col[pick], :2]
            )
            cells[key] = row
    Path(d / a.out).write_text(
        json.dumps(dict(cells=cells, slopes=slopes(cells)), indent=1) + "\n"
    )


def metrics(game, **sets):
    """{name: accuracy, blunder rate and their standard errors clustered by game} per (N, 2) set."""
    out = {}
    for name, v in sets.items():
        m, se = cc.clustered(np.asarray(v, float)[:, :2], game)
        out[name] = dict(
            accuracy=float(m[0]),
            accuracy_se=float(se[0]),
            blunder=float(m[1]),
            blunder_se=float(se[1]),
        )
    return out


def slopes(cells):
    """Per time control and metric, each source's slope against rating and its ratio to the humans'."""
    slopes = {}
    print(
        "slope per 1,000 rating points (accuracy points; log blunder rate), and the bot's over the humans'"
    )
    for fmt in cf.FORMATS:
        rows = [c for k, c in cells.items() if k.startswith(fmt + "/")]
        x = [c["elo"] for c in rows]
        out = {}
        for metric, log in (("accuracy", False), ("blunder", True)):
            s = {
                w: fit(
                    x,
                    [c[w][metric] for c in rows],
                    [c[w][metric + "_se"] for c in rows],
                    log,
                )
                for w in ("human", "raw", "calibrated")
            }
            out[metric] = {
                w: dict(slope=v[0], se=v[1], ratio=v[0] / s["human"][0])
                for w, v in s.items()
            }
            print(f"  {fmt:9s} {metric:8s} human {s['human'][0]:+7.3f}  raw {s['raw'][0]:+7.3f} ({s['raw'][0] / s['human'][0]:.2f}x)"
                  f"  calibrated {s['calibrated'][0]:+7.3f} ({s['calibrated'][0] / s['human'][0]:.2f}x)")  # fmt: skip
        slopes[fmt] = out
    return slopes


if __name__ == "__main__":
    main()
