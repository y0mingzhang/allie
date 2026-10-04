"""One calibration rule for every time control (CALIBRATION.md), fitted on the paired eval.

    x = (R - 1700) / 1000                                  R: the rating Allie is conditioned on
    T = exp(tau0 + tau x)                                  (tau0 = 0 unless stated)
    N = min(cap, k * t * exp(gamma x))                     t: predicted human think time (s)
    pi ~ s_N^(1/T)                                         s_0: the policy; s_N: allie.search
                                                           coverage's calibrated distribution
Families: temp (N = 0), search (T = 1), both. Per position, the expected accuracy / blunder / top-1
of the full move distribution is precomputed on a grid of T and the budget ladder 0/8/32/128/256,
then interpolated in log T and log2(1 + N). Fit: chi-square of the paired gaps (bot - human, in
standard errors) of accuracy and blunder rate over the 800-2600 bins of all four time controls;
2-fold over game halves. Reported per time control: debiased RMS in Elo (bullet: in SE; its human
curve is too flat for Elo).

python calib_rule.py CALIB_DIR [--families temp,temp0,search,both]
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

import calib_fit as cf

LADDER = np.array([0, 8, 32, 128, 256])
U = np.log2(1 + LADDER)
TGRID = np.exp(np.linspace(np.log(0.3), np.log(2.0), 25))
K = [cf.METRICS.index(m) for m in ("accuracy", "blunder", "top1")]
FIT = (0, 1)  # of K: accuracy and blunder rate
NO_BULLET = False  # --no-bullet-search: N = 0 in bullet
HW = (
    np.inf
)  # --hw: at most this many simulations per predicted second (the live bot's speed)


def precompute(d, dirs=("search-human", "search-human-256")):
    """Per position arrays: format, bin, elo, half, think seconds, human metrics [3], and
    grid [T, ladder, 3] of the distribution's expected metrics."""
    meta = np.load(d / "positions-human.npz")["meta"]
    mpv = cf.load_mpv(d / "mpv-human")
    dists = cf.load_dists([d / x for x in dirs])
    out = {k: [] for k in ("f", "bin", "elo", "half", "think", "human", "grid", "raw", "clock")}
    for i, dd in dists.items():
        cps, m = mpv.get(i), meta[i]
        if (
            cps is None
            or f"cov_p{LADDER[-1]}" not in dd
            or set(dd["legal"].tolist()) != set(cps)
        ):
            continue
        mm = cf.move_metrics([cps[t] for t in dd["legal"]])[K]
        mm = np.where(np.isnan(mm), 0, mm)
        logs = [np.log(np.maximum(dd["prior"].astype(float), 1e-30))]
        logs += [
            np.log(np.maximum(dd[f"cov_p{n}"].astype(float), 1e-30)) for n in LADDER[1:]
        ]
        z = np.stack(logs)[None] / TGRID[:, None, None]  # [T, ladder, moves]
        p = np.exp(z - z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        h = list(dd["legal"]).index(int(m[8]))
        out["grid"].append(np.concatenate([p @ mm.T, p[..., h : h + 1]], -1))  # metrics, P(human move)
        out["human"].append(mm[:, h])
        out["raw"].append(-np.log(max(float(dd["prior"][h]), 1e-12)))
        out["f"].append(int(m[2]))
        out["bin"].append(int(m[3]))
        out["elo"].append(float(m[4]))
        out["half"].append(hash((int(m[0]), int(m[1]))) % 2)
        out["think"].append(cf.think_seconds(dd["heads"]))
        out["clock"].append(float(m[7]) if m[7] >= 0 else np.inf)  # the mover's seconds left
    return {k: np.array(v) for k, v in out.items()}


def values(A, tau0, tau, k, gamma, cap, x0=None, alpha=1.0):
    """Expected metrics and P(human move) [n, 4] of every position under the rule. x0 set: the
    hinge temperature exp(tau max(0, x - x0)), exactly 1 below rating 1700 + 1000 x0. alpha: the
    budget grows as the predicted think time to this power."""
    x = (A["elo"] - 1700) / 1000
    lt = tau0 + tau * x if x0 is None else tau * np.maximum(x - x0, 0)
    t = np.clip(lt, np.log(TGRID[0]), np.log(TGRID[-1]))
    it = np.interp(t, np.log(TGRID), np.arange(len(TGRID)))
    n = np.minimum(cap, k * A["think"] ** alpha * np.exp(gamma * x)) if k > 0 else np.zeros_like(x)
    n = np.minimum(n, HW * A["think"])  # what the hardware can search in a human's think time
    if NO_BULLET:
        n = np.where(A["f"] == 0, 0, n)
    iu = np.interp(np.log2(1 + n), U, np.arange(len(U)))
    t0, u0 = np.floor(it).astype(int), np.floor(iu).astype(int)
    t1, u1 = np.minimum(t0 + 1, len(TGRID) - 1), np.minimum(u0 + 1, len(U) - 1)
    # the bot's hard limit after the rounding: no rung above a tenth of the clock above a 1 s reserve
    most = np.searchsorted(LADDER, HW * np.maximum(A["clock"] - 1, 0) / 10, side="right") - 1
    u0, u1 = np.minimum(u0, most), np.minimum(u1, most)
    wt, wu = (it - t0)[:, None], (iu - u0)[:, None]
    r = np.arange(len(x))
    g = A["grid"]
    return ((1 - wt) * ((1 - wu) * g[r, t0, u0] + wu * g[r, t0, u1])
            + wt * ((1 - wu) * g[r, t1, u0] + wu * g[r, t1, u1]))  # fmt: skip


BINS = [(f, b) for f in range(4) for b in range(800, 2601, 200)]


def prepare(A, human):
    """Bin ids (-1 outside 800-2600), human metric means and the human curves' slopes per bin."""
    ids = np.full(len(A["elo"]), -1)
    for j, (f, b) in enumerate(BINS):
        ids[(A["f"] == f) & (A["bin"] == b)] = j
    A["ids"] = ids
    A["slope"] = np.zeros((len(BINS), len(K)))
    A["hmean"] = np.zeros((len(BINS), len(K)))
    for j, (f, b) in enumerate(BINS):
        sel = ids == j
        if sel.any():
            e = A["elo"][sel].mean()
            A["hmean"][j] = A["human"][sel].mean(0)
            A["slope"][j] = [cf.slope(human, f, k, e) for k in K]


def gaps(A, v, keep):
    """{(format, bin): [(gap, se, gap_elo, se_elo) per metric of K] + [(ce gap, se)]} over bins with
    >= 100 positions; the ce gap is the human move's cross-entropy minus the raw policy's."""
    ids = np.where(keep, A["ids"], -1)
    m = ids >= 0
    ids = ids[m]
    dv = np.concatenate([v[m][:, : len(K)] - A["human"][m], (-np.log(np.maximum(v[m][:, -1], 1e-12)) - A["raw"][m])[:, None]], 1)
    n = np.bincount(ids, minlength=len(BINS)).astype(float)
    s1 = np.stack([np.bincount(ids, dv[:, j], len(BINS)) for j in range(dv.shape[1])], 1)
    s2 = np.stack([np.bincount(ids, dv[:, j] ** 2, len(BINS)) for j in range(dv.shape[1])], 1)
    out = {}
    for j, (f, b) in enumerate(BINS):
        if n[j] < 100:
            continue
        g = s1[j] / n[j]
        se = np.sqrt(np.maximum(s2[j] / n[j] - g**2, 1e-18) / n[j])
        row = []
        for q, k in enumerate(K):
            hm = A["hmean"][j, q]
            d_, s_ = (np.log((hm + g[q]) / hm), se[q] / hm) if cf.METRICS[k] in cf.LOG else (g[q], se[q])
            row.append((g[q], se[q], d_ / A["slope"][j, q], abs(s_ / A["slope"][j, q])))
        row.append((g[-1], se[-1]))
        out[f, b] = row
    return out


CE_WEIGHT = 0.0  # --ce-weight: penalty on cells whose human-move cross-entropy exceeds the raw policy's


def chi2(gp):
    fit = sum((r[j][0] / r[j][1]) ** 2 for r in gp.values() for j in FIT)
    ce = sum(max(r[-1][0] / r[-1][1], 0) ** 2 for r in gp.values())
    return float(fit + CE_WEIGHT * ce)


def ce_report(gp):
    """Per format: cells whose cross-entropy is worse than the raw policy's beyond 2 SE, and the worst."""
    out = {}
    for f in range(4):
        rows = [(b, r[-1]) for (g, b), r in gp.items() if g == f]
        if rows:
            worse = [b for b, (d, e) in rows if d > 2 * e]
            out[cf.FORMATS[f]] = dict(worse_cells=worse, max_gap=round(max(d for _, (d, _) in rows), 4))
    return out


def report(gp):
    """Per format: debiased RMS (Elo; bullet: SE) and mean signed gap (Elo), over FIT metrics."""
    out = {}
    for f in range(4):
        rows = [r for (g, b), r in gp.items() if g == f]
        if not rows:
            continue
        if f == 0:
            z = np.array([r[j][0] / r[j][1] for r in rows for j in FIT])
            out["bullet"] = dict(
                rms_se=float(np.sqrt(max(np.mean(z**2) - 1, 0))),
                mean_se=float(z.mean()),
            )
        else:
            e = np.array([(r[j][2], r[j][3]) for r in rows for j in FIT])
            out[cf.FORMATS[f]] = dict(rms=float(np.sqrt(max(np.mean(e[:, 0] ** 2) - np.mean(e[:, 1] ** 2), 0))),
                                      bias=float(e[:, 0].mean()))  # fmt: skip
    return out


GRIDS = {
    "temp0": dict(
        tau0=[0.0], tau=np.linspace(-1.5, 0.5, 41), k=[0], gamma=[0.0], cap=[256]
    ),
    "temp": dict(
        tau0=np.linspace(-0.6, 0.3, 19),
        tau=np.linspace(-1.5, 0.5, 41),
        k=[0],
        gamma=[0.0],
        cap=[256],
    ),
    "search": dict(
        tau0=[0.0],
        tau=[0.0],
        k=2.0 ** np.linspace(-4, 6, 21),
        gamma=np.linspace(0, 6, 13),
        cap=[256],
        alpha=[1.0, 1.5, 2.0, 3.0],
    ),
    "hinge": dict(  # search, plus sharpening only above rating 1700 + 1000 x0 (cross-entropy safe below)
        tau0=[0.0],
        tau=np.linspace(-1.0, 0.0, 11),
        k=2.0 ** np.linspace(-1, 4, 11),
        gamma=np.linspace(1, 5, 9),
        cap=[256],
        x0=[0.1, 0.3, 0.5, 0.7],
        alpha=[1.0, 2.0],
    ),
    "both1": dict(  # both, with the temperature's intercept, at budgets the live bot can afford
        tau0=[-0.3, -0.25, -0.2, -0.15, -0.1, 0.0],
        tau=np.linspace(-0.8, 0.2, 11),
        k=2.0 ** np.linspace(-1, 4, 11),
        gamma=np.linspace(-1, 4, 6),
        cap=[128, 256],
    ),
    "both": dict(
        tau0=[0.0],
        tau=np.linspace(-1.5, 0.5, 21),
        k=2.0 ** np.linspace(-2, 8, 21),
        gamma=np.linspace(-1, 5, 13),
        cap=[64, 128, 256],
    ),
}


def fit(A, family, keep):
    best = (np.inf, None)
    g = GRIDS[family]
    for par in itertools.product(g["tau0"], g["tau"], g["k"], g["gamma"], g["cap"], g.get("x0", [None]), g.get("alpha", [1.0])):
        c = chi2(gaps(A, values(A, *par), keep))
        if c < best[0]:
            best = (c, par)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--families", default="temp0,temp,search,both")
    p.add_argument("--out", default="rule.json")
    p.add_argument("--no-bullet-search", action="store_true")
    p.add_argument("--dists", default="search-human,search-human-256", help="search output dirs")
    p.add_argument("--ce-weight", type=float, default=0.0, help="penalty on cells whose cross-entropy exceeds the raw policy's")
    p.add_argument(
        "--hw",
        type=float,
        default=np.inf,
        help="simulations per predicted second the bot can afford",
    )
    p.add_argument(
        "--ladder",
        default="0,8,32,128,256",
        help="budgets precomputed (the largest bounds every cap)",
    )
    a = p.parse_args()
    global LADDER, U, HW, CE_WEIGHT, NO_BULLET
    NO_BULLET = a.no_bullet_search
    HW, CE_WEIGHT = a.hw, a.ce_weight
    LADDER = np.array([int(x) for x in a.ladder.split(",")])
    U = np.log2(1 + LADDER)
    for g in GRIDS.values():
        g["cap"] = [c for c in g["cap"] if c <= LADDER[-1]] or [LADDER[-1]]
    d = Path(a.dir)
    t0 = time.monotonic()
    A = precompute(d, a.dists.split(","))
    human = cf.summarize(
        cf.table(
            np.load(d / "positions-human.npz")["meta"],
            cf.load_mpv(d / "mpv-human"),
            {},
            {},
        ),
        [],
    )
    print(
        f"{len(A['elo'])} positions, precomputed in {time.monotonic() - t0:.0f} s",
        flush=True,
    )
    for f in range(4):
        print(
            f"  {cf.FORMATS[f]}: "
            + " ".join(
                f"{b}:{((A['f'] == f) & (A['bin'] == b)).sum()}"
                for b in range(800, 2601, 200)
            )
        )
    out = {}
    prepare(A, human)
    base = gaps(A, values(A, 0, 0, 0, 0, 256), np.ones(len(A["elo"]), bool))
    out["current"] = dict(
        params=(0, 0, 0, 0, 256), chi2=chi2(base), report=report(base)
    )
    print("current (T = 1, no search):", json.dumps(out["current"]["report"]), flush=True)
    for fam in a.families.split(","):
        res = {}
        for half in (0, 1):
            c, par = fit(A, fam, A["half"] == half)
            test = gaps(A, values(A, *par), A["half"] != half)
            res[half] = dict(
                params=[None if x is None else float(x) for x in par],
                train_chi2=c,
                test_chi2=chi2(test),
                test=report(test),
            )
        c, par = fit(A, fam, np.ones(len(A["elo"]), bool))
        allg = gaps(A, values(A, *par), np.ones(len(A["elo"]), bool))
        res["all"] = dict(params=[None if x is None else float(x) for x in par], chi2=c, report=report(allg),
                          gaps={f"{cf.FORMATS[f]}/{b}": [[round(float(x), 4) for x in m] for m in r] for (f, b), r in allg.items()})  # fmt: skip
        out[fam] = res
        cv = res[0]["test_chi2"] + res[1]["test_chi2"]
        print(f"{fam:6s} params (tau0, tau, k, gamma, cap) {tuple(None if x is None else round(float(x), 3) for x in par)}  chi2 all {c:7.0f}"
              f"  cross-validated chi2 {cv:7.0f}  halves {res[0]['params']} {res[1]['params']}", flush=True)  # fmt: skip
        print(f"       all: {json.dumps(res['all']['report'])}", flush=True)
        res["all"]["ce"] = ce_report(allg)
        sc = [sum((r[j][0] / r[j][1]) ** 2 for r in gaps(A, values(A, *res[h]["params"]), A["half"] != h).values() for j in FIT)
              for h in (0, 1)]  # fmt: skip
        res["strength_cv_chi2"] = float(sum(sc))
        print(f"       strength-only cross-validated chi2 {sum(sc):7.0f}; cross-entropy vs raw: {json.dumps(res['all']['ce'])}", flush=True)
    (d / a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
