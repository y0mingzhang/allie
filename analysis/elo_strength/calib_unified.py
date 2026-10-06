"""One calibrated rule for every cell, no rating or time-control thresholds: a single searcher's ladder,
the largest rung the move's drawn think time and the clock allow (calib_cells.py --think's mixture), and
a tilt pi ~ prior exp(beta v) on its values v (log-odds arctanh(0.99 Q), or v as given for "x" tags)
with beta = beta0 exp(gamma x), max(beta0 + gamma x, 0) or, the bot's, beta0 exp(gamma x) (t / 10 s)^delta,
x = (R - 1700) / 1000 and t the position's expected human think time. The parameters are fitted jointly
over the cells (debiased squared Elo error of accuracy and blunder rate, bullet excluded from the loss:
its human curve is too flat for Elo) with the human-move cross-entropy at or below the policy's on
average over cells; fitted on each game half and scored on the other.

python calib_unified.py CALIB_DIR --ladder 8a,32a,128a,256a,1024a [--extra DIR ...] [--cells subset|all]
--params B0,G,D [--plot plot.json]: score the think form at fixed parameters instead of fitting (and write
calib_plotdata.py's per-cell metrics and slopes for the README figure).
A ladder tag names its values: N (search-human-ann2c's cov_qN) or an --extra run's tag (e.g. 1024, 256c,
512kx: piKL's log-odds values); a trailing "a" tilts arctanh(0.99 Q) instead of the values as given. Positions without a rung's values
fall to the next lower rung the position has (so scores there understate that rung).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import calib_cells as cc
import calib_fit as cf
import calib_plotdata as cp
from allie.lichess.tokens import SECONDS

G = np.exp(
    np.linspace(np.log(0.02), np.log(64), 33)
)  # beta grid the fit interpolates in log beta


def tilt_metrics(lp, v, mm, h):
    """[beta in G, (accuracy, blunder, P(human move))] of pi ~ exp(lp + beta v)."""
    z = lp[None] + G[:, None] * v[None]
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return np.c_[p @ mm.T, p[:, h]]


def load(d, ladder, extras, sim):
    """Per position: its index (pos), cell, rating, half, game, human metrics, the policy's metrics, ce0 = -log P(human
    move) under the policy, the rungs' metrics on G [K, G, 3] and their think-time weights [K + 1]."""
    meta = np.load(d / "positions-human.npz")["meta"]
    z = np.load(d / "positions-human.npz")
    tokens, starts = z["tokens"], z["offsets"]
    mpv = cf.load_mpv(d / "mpv-human")
    dists = cf.load_dists([d / "search-human-ann2c"])
    ex = [cc.extra(Path(e)) for e in extras]
    K = len(ladder)
    names = [(t[:-1], True) if t.endswith("a") else (t, False) for t in ladder]  # (values tag, log-odds tilt)
    cost = np.array([cc.leaves(t) for t, _ in names]) * sim
    out = {
        k: []
        for k in (
            "f",
            "bin",
            "elo",
            "half",
            "game",
            "H",
            "base",
            "ce0",
            "R",
            "W",
            "have",
            "think",
            "clock",
            "est",
            "pos",
        )
    }
    for i, dd in dists.items():
        cps, m = mpv.get(i), meta[i]
        if cps is None or "cov_p256" not in dd or set(dd["legal"].tolist()) != set(cps):
            continue
        mm = cf.move_metrics([cps[t] for t in dd["legal"]])[cc.K]
        h = list(dd["legal"]).index(int(m[8]))
        raw = dd["prior"].astype(float)
        lp = np.log(np.maximum(raw, 1e-300))
        R, have = np.zeros((K, len(G), 3)), np.zeros(K, bool)
        for k, (base, atanh) in enumerate(names):
            vals, lpk = None, lp
            if base.isdigit() and f"cov_q{base}" in dd:
                vals = dd[f"cov_q{base}"].astype(float)
            else:
                for e in ex:
                    if i in e and base in e[i][2]:
                        legal, prior, runs = e[i]
                        perm = np.array(
                            [
                                {t: j for j, t in enumerate(legal)}[t]
                                for t in dd["legal"]
                            ]
                        )
                        vals, lpk = (
                            runs[base][1][perm],
                            np.log(np.maximum(prior[perm], 1e-300)),
                        )
                        break
            if vals is None:
                continue
            if atanh:
                vals = np.arctanh(0.99 * np.clip(vals, -1, 1))
            R[k], have[k] = tilt_metrics(lpk, vals, mm, h), True
        tok, btok = int(tokens[starts[i] + 2]), int(tokens[starts[i] + 1])
        clock = float(m[7]) if m[7] >= 0 else None
        base = SECONDS[btok - 192] if 192 <= btok < 192 + len(SECONDS) else None
        inc = tok - 10 if 10 <= tok <= 190 else None
        out["clock"].append(np.nan if clock is None else clock)
        out["est"].append(np.nan if base is None or inc is None else base + 40 * inc)
        w = cc.weights(
            dd["heads"][:63].astype(float),
            clock,
            tok - 10 if 10 <= tok <= 190 else 0,
            cost,
        )
        for k in range(
            K - 1, -1, -1
        ):  # a rung without values: its draws fall to the next lower one
            if not have[k]:
                w[k] += w[k + 1]
                w[k + 1] = 0.0
        out["pos"].append(i)
        out["f"].append(int(m[2]))
        out["bin"].append(int(m[3]))
        out["elo"].append(float(m[4]))
        out["half"].append(hash((int(m[0]), int(m[1]))) % 2)
        out["game"].append(hash((int(m[0]), int(m[1]))))
        out["H"].append(mm[:, h])
        out["base"].append(np.r_[mm @ raw, raw[h]])
        out["ce0"].append(-np.log(max(raw[h], 1e-12)))
        out["think"].append(cf.think_seconds(dd["heads"]))
        out["R"].append(R)
        out["W"].append(w)
        out["have"].append(have)
    return {k: np.array(v) for k, v in out.items()}


def mixed(A, beta):
    """Per position (accuracy, blunder, ce - ce0) of the rule at per-position beta."""
    t = np.interp(np.log(np.maximum(beta, G[0])), np.log(G), np.arange(len(G)))
    j = np.minimum(np.floor(t).astype(int), len(G) - 2)
    a = (t - j)[:, None, None]
    n = np.arange(len(beta))[:, None]
    k = np.arange(A["R"].shape[1])[None]
    rb = (1 - a) * A["R"][n, k, j[:, None]] + a * A["R"][
        n, k, j[:, None] + 1
    ]  # [N, K, 3]
    mix = A["W"][:, :1] * A["base"] + (A["W"][:, 1:, None] * rb).sum(1)
    return np.c_[mix[:, :2], -np.log(np.maximum(mix[:, 2], 1e-12)) - A["ce0"]]


def betas(A, par, form):
    """exp: beta0 exp(gamma x); linear: max(beta0 + gamma x, 0); think: beta0 exp(gamma x) (t / 10 s)^delta
    with t the position's expected human think time (its think-time head), x = (R - 1700) / 1000; clock
    and est: think times (c / 600 s)^kappa, c the mover's clock left or the game's estimated duration
    base + 40 increment (1 when unknown); thinkx: think with exponent delta + eta x (the rating slope grows
    with the think time)."""
    x = (A["elo"] - 1700) / 1000
    if form == "thinkx":
        b0, g, dl, eta = par
        return b0 * np.exp(g * x) * (A["think"] / 10) ** (dl + eta * x)
    if form in ("think", "clock", "est"):
        b0, g, dl, *k = par
        c = 1.0 if form == "think" else np.nan_to_num(A[form] / 600, nan=1.0) ** k[0]
        return b0 * np.exp(g * x) * (A["think"] / 10) ** dl * c
    b0, g = par
    return b0 * np.exp(g * x) if form == "exp" else np.maximum(b0 + g * x, 0.0)


def cell_stats(A, X, sel, human):
    """{(f, bin): [elo_acc, se, elo_blun, se, ce, se]} over the positions in sel."""
    out = {}
    for f in range(4):
        for b in range(800, 2601, 200):
            s = sel & (A["f"] == f) & (A["bin"] == b)
            if s.sum() < 100:
                continue
            out[f, b] = cc.score(
                X[s][:, None, :], A["H"][s], A["elo"][s].mean(), human, f, A["game"][s]
            )[0]
    return out


CE, CE_Z = "mean", 0.0  # --ce: the cross-entropy bar on the cells' mean gap, or on every cell's gap less CE_Z SEs
LOSS, BULLET = "elo", 9999  # --loss z: squared gaps in standard errors; --bullet-from: bullet bins in the loss


def objective(st):
    """Debiased squared error summed over the non-bullet cells and bullet bins from BULLET (in Elo, or with
    LOSS = "z" in standard errors), and the cross-entropy gap the bar holds: the cells' mean, or (CE = "cells")
    the largest cell's less CE_Z SEs."""
    z = LOSS == "z"
    loss = sum(
        ((v[0] ** 2 - v[1] ** 2) / (v[1] ** 2 if z else 1) + (v[2] ** 2 - v[3] ** 2) / (v[3] ** 2 if z else 1)) / 2
        for (f, b), v in st.items()
        if f > 0 or b >= BULLET
    )
    ce = [v[4] for v in st.values()]
    return loss, float(max(v[4] - CE_Z * v[5] for v in st.values()) if CE == "cells" else np.mean(ce))


def fit(A, sel, human, form, grid):
    best = (np.inf, None)
    for par in grid:
        loss, ce = objective(cell_stats(A, mixed(A, betas(A, par, form)), sel, human))
        if ce <= 0 and loss < best[0]:
            best = (loss, par)
    return best[1]


def plot(A, X, sel):
    """calib_plotdata.py's per-cell metrics (humans, the policy, the rule) and slopes against rating."""
    cells = {}
    for f, b in ((f, b) for f in range(4) for b in range(800, 2601, 200)):
        s = sel & (A["f"] == f) & (A["bin"] == b)
        if s.sum() < 100:
            continue
        g = A["game"][s]
        row = dict(n=int(s.sum()), games=len(np.unique(g)), elo=float(A["elo"][s].mean()), choice="unified")
        cells[f"{cf.FORMATS[f]}/{b}"] = row | cp.metrics(g, human=A["H"][s], raw=A["base"][s], calibrated=X[s])
    return dict(cells=cells, slopes=cp.slopes(cells))


def outside(cells, z=1.96):
    """Cells outside the noise: |gap| > z SE on accuracy or blunder rate, and cross-entropy above the policy's at
    z SEs; a cell's [half, stat] rows (held out: each half scored by the other's fit) pooled as their mean."""
    out = {}
    for c, v in cells.items():
        v = np.atleast_2d(v)
        m, se = v.mean(0), np.sqrt((v[:, 1::2] ** 2).sum(0)) / len(v)
        g = dict(acc=m[0] / se[0], blunder=m[2] / se[1], ce=m[4] / se[2])
        bad = [k for k in ("acc", "blunder") if abs(g[k]) > z] + (["ce"] if g["ce"] > z else [])
        if bad:
            out[c] = {k: round(float(x), 2) for k, x in g.items()} | dict(elo=[round(float(m[0])), round(float(m[2]))])
    return out


def report(st, label):
    """Per format: debiased RMS Elo (mean over cells and, held out, over the two halves' scores, each
    debiased by its own standard errors), mean cross-entropy gap, cells above the policy's."""
    rows = {}
    for f in range(4):
        vs = [s for (g, _), s in st.items() if g == f]
        if not vs:
            continue
        halves = [np.atleast_2d(s) for s in vs]
        deb = np.mean([np.mean((h[:, 0] ** 2 - h[:, 1] ** 2 + h[:, 2] ** 2 - h[:, 3] ** 2) / 2) for h in halves])
        v = [h.mean(0) for h in halves]
        rows[cf.FORMATS[f]] = dict(rms=float(np.sqrt(max(deb, 0))), ce_mean=float(np.mean([s[4] for s in v])),
                                   cells_above_raw=int(sum(s[4] > 0 for s in v)),
                                   cells_above_raw_95=int(sum(s[4] - 1.96 * s[5] > 0 for s in v)),
                                   worst_ce=float(max(s[4] for s in v)))  # fmt: skip
    print(
        label,
        json.dumps(
            {
                k: {a: round(b, 4) if isinstance(b, float) else b for a, b in r.items()}
                for k, r in rows.items()
            }
        ),
    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--ladder", required=True)
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--sim", type=float, default=0.004, help="seconds a leaf (allie.lichess.calibration.COST)")
    p.add_argument("--margin", type=float, default=0.3, help="seconds a search ends before the think time (calibration.MARGIN)")
    p.add_argument("--ce", choices=["mean", "cells"], default="mean", help="the cross-entropy bar: on the cells' mean or on every cell")
    p.add_argument("--ce-z", type=float, default=0.0, help="--ce cells: each cell's gap less this many standard errors")
    p.add_argument(
        "--cells",
        default="all",
        help="all, or subset: only cells where every rung has values for most positions",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--forms", help="comma-separated subset of exp, linear, think, clock, est, thinkx")
    floats = lambda x: tuple(float(v) for v in x.split(","))  # noqa: E731
    p.add_argument("--deltas", type=floats, help="think form: the deltas to try")
    p.add_argument("--kappas", type=floats, help="clock and est forms: the kappas to try; thinkx: the etas")
    p.add_argument("--gammas", type=floats, help="thinkx: the gammas to try")
    p.add_argument("--params", type=floats, help="score these parameters (--forms' first form, default think), no fit")
    p.add_argument("--plot", help="with --params: write per-cell metrics and slopes here (calib_plotdata.py's format)")
    p.add_argument("--loss", choices=["elo", "z"], default="elo", help="squared gaps in Elo or in standard errors")
    p.add_argument("--bullet-from", type=int, default=9999, help="bullet bins from this rating join the loss")
    a = p.parse_args()
    global CE, CE_Z, LOSS, BULLET
    cc.MARGIN, CE, CE_Z, LOSS, BULLET = a.margin, a.ce, a.ce_z, a.loss, a.bullet_from
    d = Path(a.dir)
    ladder = a.ladder.split(",")
    A = load(d, ladder, a.extra, a.sim)
    human = cf.summarize(
        cf.table(
            np.load(d / "positions-human.npz")["meta"],
            cf.load_mpv(d / "mpv-human"),
            {},
            {},
        ),
        [],
    )
    sel = np.ones(len(A["f"]), bool)
    if a.cells == "subset":
        sel = A["have"][:, :-1].all(1)  # the 1,024 rung exists only in some cells
    print(f"{sel.sum()} positions; ladder {ladder}")
    if a.params:
        X = mixed(A, betas(A, a.params, (a.forms or "think").split(",")[0]))
        st = cell_stats(A, X, sel, human)
        cells = {f"{cf.FORMATS[f]}/{b}": [float(x) for x in v] for (f, b), v in st.items()}
        out = dict(params=list(a.params), in_sample=report(st, "  in-sample"), cells=cells, outside=outside(cells))
        print("  outside the noise:", json.dumps(out["outside"]))
        (d / a.out).write_text(json.dumps(dict(think=out), indent=1) + "\n")
        if a.plot:
            (d / a.plot).write_text(json.dumps(plot(A, X, sel), indent=1) + "\n")
        return
    grids = dict(exp=[(b, g) for b in np.exp(np.linspace(np.log(0.05), np.log(32), 25)) for g in np.linspace(-2, 6, 17)],
                 linear=[(b, g) for b in np.linspace(0, 12, 25) for g in np.linspace(-4, 24, 15)],
                 think=[(b, g, dl) for b in np.exp(np.linspace(np.log(0.1), np.log(16), 15)) for g in np.linspace(0, 4, 9)
                        for dl in (a.deltas or (0, 0.25, 0.5, 0.75, 1.0, 1.5))])  # fmt: skip
    for f in ("clock", "est"):
        grids[f] = [(b, g, dl, k) for b in np.exp(np.linspace(np.log(0.05), np.log(16), 17)) for g in (2, 2.5, 3, 3.5, 4)
                    for dl in (a.deltas or (0.25, 0.5, 0.625, 0.75, 1.0)) for k in (a.kappas or (-0.25, 0, 0.25, 0.5, 0.75, 1))]  # fmt: skip
    grids["thinkx"] = [(b, g, dl, k) for b in np.exp(np.linspace(np.log(0.1), np.log(0.6), 11)) for g in (a.gammas or (2.5, 3, 3.5))
                       for dl in (a.deltas or (0.5, 0.75, 1.0)) for k in (a.kappas or (-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0))]  # fmt: skip
    if a.forms:
        grids = {k: v for k, v in grids.items() if k in a.forms.split(",")}
    out = {}
    for form, grid in grids.items():
        par = fit(A, sel, human, form, grid)
        if par is None:
            print(f"{form}: no parameters on the grid hold the cross-entropy bar")
            continue
        full = cell_stats(A, mixed(A, betas(A, par, form)), sel, human)
        held = {}
        for h in (0, 1):
            ph = fit(A, sel & (A["half"] == h), human, form, grid)
            if ph is None:
                print(f"{form}: half {h}: no parameters on the grid hold the cross-entropy bar")
                break
            st = cell_stats(
                A, mixed(A, betas(A, ph, form)), sel & (A["half"] != h), human
            )
            for c, v in st.items():
                held.setdefault(c, []).append((ph, v))
        else:
            heldst = {c: np.array([v for _, v in vs]) for c, vs in held.items()}  # [half, stat]
            print(
                f"{form}: params {tuple(round(float(x), 3) for x in par)}; halves {[tuple(round(float(x), 3) for x in held[next(iter(held))][h][0]) for h in (0, 1)]}"
            )
            out[form] = dict(params=[float(x) for x in par], in_sample=report(full, "  in-sample"), held_out=report(heldst, "  held-out "),
                             cells={f"{cf.FORMATS[f]}/{b}": [float(x) for x in v] for (f, b), v in full.items()},
                             cells_held={f"{cf.FORMATS[f]}/{b}": v.tolist() for (f, b), v in heldst.items()},
                             halves=[[float(x) for x in held[next(iter(held))][h][0]] for h in (0, 1)],
                             outside_held=outside({f"{cf.FORMATS[f]}/{b}": v for (f, b), v in heldst.items()}))  # fmt: skip
            print("  outside the noise, held out:", json.dumps(out[form]["outside_held"]))
    (d / a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
