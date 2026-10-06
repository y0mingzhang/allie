"""The think-time-capped frontier of another track's search runs (search_eff harness format: index, legal, prior,
heads, q{b}[_variant], leaves{b} per chunk) on their positions: per cell and constant beta, the Elo gaps and the
cross-entropy gap of the rungs mixed by the bot's think-time draws at --sim seconds a leaf (calib_frontier.py's
view), and with the largest budget on every position.

python calib_runs.py CALIB_DIR RUN [--variant soft] [--fill mean|root] [--atanh 0.99] --cells 3:2400,3:2600 --out F.json
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
    p.add_argument("run")
    p.add_argument("--variant", default="")
    p.add_argument("--fill", default="mean", choices=["mean", "root"])
    p.add_argument("--atanh", type=float, default=0.0)
    p.add_argument("--sim", type=float, default=0.0045)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cc.MARGIN = a.margin
    d = Path(a.dir)
    z = np.load(d / "positions-human.npz")
    meta, tokens, starts = z["meta"], z["tokens"], z["offsets"]
    want = {tuple(map(int, c.split(":"))) for c in a.cells.split(",")}
    mpv = cf.load_mpv(d / "mpv-human")
    parts = [np.load(f) for f in sorted(Path(a.run).glob("[0-9]*.npz")) if ".partial" not in f.name]
    budgets = [int(b) for b in parts[0]["budgets"]]
    A = {k: [] for k in ("f", "bin", "elo", "half", "game", "H", "base", "ce0", "R", "W")}
    for zz in parts:
        off = zz["offsets"]
        for r, i in enumerate(zz["index"]):
            m = meta[int(i)]
            if (int(m[2]), int(m[3])) not in want or (cps := mpv.get(int(i))) is None:
                continue
            legal = zz["legal"][off[r] : off[r + 1]].astype(int)
            if set(legal.tolist()) != set(cps):
                continue
            mm = cf.move_metrics([cps[t] for t in legal])[cc.K]
            h = list(legal).index(int(m[8]))
            prior = zz["prior"][off[r] : off[r + 1]].astype(float)
            lp = np.log(np.maximum(prior, 1e-300))
            w = zz["heads"][r, 63:66].astype(float)
            w = np.exp(w - w.max())
            v0 = (w[0] - w[2]) / w.sum()
            R, cost = [], []
            for b in budgets:
                q = zz[f"q{b}_{a.variant}" if a.variant else f"q{b}"][off[r] : off[r + 1]].astype(float)
                miss = np.isnan(q)
                if miss.any():
                    fill = np.nansum(prior * np.nan_to_num(q)) / max(prior[~miss].sum(), 1e-12) if a.fill == "mean" and (~miss).any() else v0
                    q = np.where(miss, fill, q)
                if a.atanh:
                    q = np.arctanh(a.atanh * np.clip(q, -1, 1))
                R.append(cu.tilt_metrics(lp, q, mm, h))
                cost.append(float(zz[f"leaves{b}"][r]) * a.sim)
            tok = int(tokens[starts[int(i)] + 2])
            clock = float(m[7]) if m[7] >= 0 else None
            A["W"].append(cc.weights(zz["heads"][r, :63].astype(float), clock, tok - 10 if 10 <= tok <= 190 else 0, np.maximum.accumulate(cost)))
            A["R"].append(np.array(R))
            A["f"].append(int(m[2]))
            A["bin"].append(int(m[3]))
            A["elo"].append(float(m[4]))
            A["game"].append(hash((int(m[0]), int(m[1]))))
            A["half"].append(A["game"][-1] % 2)
            A["H"].append(mm[:, h])
            A["base"].append(np.r_[mm @ prior, prior[h]])
            A["ce0"].append(-np.log(max(prior[h], 1e-12)))
    A = {k: np.array(v) for k, v in A.items()}
    human = cf.summarize(cf.table(meta, mpv, {}, {}), [])
    top = dict(A, W=np.eye(A["W"].shape[1])[np.full(len(A["f"]), A["W"].shape[1] - 1)])
    print(f"{len(A['f'])} positions; budgets {budgets}; P(rung) mean {np.round(A['W'].mean(0), 3).tolist()}")
    out = {}
    for f, b in sorted(want):
        sel = (A["f"] == f) & (A["bin"] == b)
        rows = [{"beta": float(beta)} | {name: [float(x) for x in cu.cell_stats(B, cu.mixed(B, np.full(len(B["f"]), beta)), sel, human)[f, b]]
                                          for name, B in (("capped", A), ("top", top))} for beta in cu.G]  # fmt: skip
        out[f"{cf.FORMATS[f]}/{b}"] = rows
        for name in ("capped", "top"):
            ok = [r for r in rows if r[name][4] - 1.96 * r[name][5] <= 0]
            best = max(ok, key=lambda r: r[name][0]) if ok else None
            reach = max(rows, key=lambda r: r[name][0])
            print(f"{cf.FORMATS[f]}/{b} {name}: strongest under the CE bar b{best['beta']:.2f} {best[name][0]:+.0f}/{best[name][2]:+.0f} (+-{1.96 * best[name][1]:.0f}/{1.96 * best[name][3]:.0f}) ce {best[name][4]:+.4f}+-{1.96 * best[name][5]:.4f}"
                  f" | strongest at any beta b{reach['beta']:.2f} {reach[name][0]:+.0f}/{reach[name][2]:+.0f} ce {reach[name][4]:+.4f}")  # fmt: skip
    (d / a.out).write_text(json.dumps(dict(budgets=budgets, cells=out)) + "\n")


if __name__ == "__main__":
    main()
