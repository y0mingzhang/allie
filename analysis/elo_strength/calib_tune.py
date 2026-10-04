"""Tune calibrated move-selection families on the paired eval (calib_fit.py's metrics), per format,
with a game-level 2-fold split: fit on one half of the games, report on the other.

Families (R = the mover's conditioning rating, x = (R - 1700) / 1000):
  temp:  pi ~ p^(1/T),                 T = a + b x
  tilt:  pi ~ p^(1/T) exp(beta Q_n),   T = a + b x, beta = max(0, c + d x), Q_n from coverage search
         with the format's budget n
Objective: mean over solid bins of the squared paired Elo gaps of accuracy and blunder rate.

python calib_tune.py CALIB_DIR [--dists policy search] [--budget bullet=8,blitz=t8,rapid=32,classical=128]
    budget per format: a fixed coverage budget, or tC = the largest of 8/32/128 at most C x the
    model's predicted think time (s) for the position (0 below 8: no search there)
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import calib_fit as cf

TGRID = np.round(np.arange(0.3, 1.51, 0.05), 2)
BGRID = (0, 1, 2, 4, 8, 16)
K = [cf.METRICS.index(m) for m in ("accuracy", "blunder")]
ZSCORE = False  # --z: fit gaps in standard errors (formats whose human curve is too flat for Elo)


def position_grid(meta, mpv, dists, budgets):
    """Per position: (format, bin, elo, game half, human metrics, metrics[T, beta, metric])."""
    rows = []
    for i, cps in mpv.items():
        d, m = dists.get(i, {}), meta[i]
        if (
            "prior" not in d
            or int(m[8]) not in cps
            or set(d["legal"].tolist()) != set(cps)
        ):
            continue
        mm = cf.move_metrics([cps[t] for t in d["legal"]])
        mm = np.where(np.isnan(mm), 0, mm)
        h = mm[:, list(d["legal"]).index(int(m[8]))]
        spec = budgets[int(m[2])]
        if spec.startswith(
            "t"
        ):  # adaptive: the largest saved budget <= c * predicted think time
            want = float(spec[1:]) * cf.think_seconds(d["heads"]) if "heads" in d else 0
            n = max([b for b in (8, 32, 128) if b <= want], default=0)
        else:
            n = int(spec)
        searched = "cov_q8" in d  # this position's search outputs exist
        q = d.get(f"cov_q{n}") if n else None
        logp = np.log(np.maximum(d["prior"].astype(float), 1e-30))
        covp = (
            np.log(np.maximum(d[f"cov_p{n}"].astype(float), 1e-30))
            if n and searched
            else logp
        )
        # slots: one per beta (policy tilted by Q; no search budget: the policy), then the
        # search's calibrated distribution (no budget: the policy)
        out = np.empty((len(TGRID), len(BGRID) + 1, mm.shape[0]))
        for a, t in enumerate(TGRID):
            for b, beta in enumerate(BGRID):
                if beta and not searched:
                    out[a, b] = np.nan
                    continue
                z = logp / t + (beta * q if beta and q is not None else 0)
                p = np.exp(z - z.max())
                out[a, b] = mm @ (p / p.sum())
            z = covp / t
            p = np.exp(z - z.max())
            out[a, -1] = mm @ (p / p.sum()) if searched else np.nan
        half = hash((int(m[0]), int(m[1]))) % 2
        rows.append((int(m[2]), int(m[3]), int(m[4]), half, h, out))
    return rows


def arrays(rows, f):
    """Format f's solid-bin positions as arrays: bin, elo, half, human metrics, metric grid."""
    rs = [r for r in rows if r[0] == f and cf.solid(r[0], r[1])]
    return dict(bin=np.array([r[1] for r in rs]), elo=np.array([r[2] for r in rs], float),
                half=np.array([r[3] for r in rs]), human=np.array([r[4] for r in rs]),
                grid=np.array([r[5] for r in rs]))  # fmt: skip


def evaluate(A, human, f, T, B, half=None):
    """Mean squared paired Elo gap (accuracy, blunder) over format f's solid bins and the per-bin
    gaps, for T(elo) and beta(elo); positions of one half only if half is given."""
    keep = np.ones(len(A["elo"]), bool) if half is None else A["half"] == half
    elo, grid = A["elo"][keep], A["grid"][keep]
    ia = np.interp(np.clip(T(elo), TGRID[0], TGRID[-1]), TGRID, np.arange(len(TGRID)))
    braw = np.broadcast_to(np.asarray(B(elo), float), elo.shape)
    ib = np.interp(np.maximum(braw, 0), BGRID, np.arange(len(BGRID)))
    lo, w = np.floor(ia).astype(int), ia - np.floor(ia)
    blo, bw = np.floor(ib).astype(int), ib - np.floor(ib)
    hi, bhi = np.minimum(lo + 1, len(TGRID) - 1), np.minimum(blo + 1, len(BGRID) - 1)
    n = np.arange(len(elo))
    cov = braw < 0  # B < 0: sample the search's calibrated distribution
    blo, bhi = np.where(cov, len(BGRID), blo), np.where(cov, len(BGRID), bhi)
    v = np.zeros((len(elo), grid.shape[-1]))
    for x, wa in ((lo, 1 - w), (hi, w)):
        for y, wb in ((blo, 1 - bw), (bhi, bw)):
            wt = (wa * wb)[:, None]
            v += np.where(wt > 0, wt * grid[n, x, y], 0)
    ok = ~np.isnan(v).any(1)  # positions without the search outputs a beta > 0 needs
    elo, v, keep_idx = elo[ok], v[ok], np.flatnonzero(keep)[ok]
    gaps = {}
    bins, hums = A["bin"][keep_idx], A["human"][keep_idx]
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < 200:
            continue
        e, hm, vm = elo[sel].mean(), hums[sel].mean(0), v[sel].mean(0)
        se = (v[sel] - hums[sel]).std(0) / np.sqrt(sel.sum())
        g = []
        for k in K:
            log = cf.METRICS[k] in cf.LOG
            dk, ek = (
                (np.log(vm[k] / hm[k]), se[k] / hm[k])
                if log
                else (vm[k] - hm[k], se[k])
            )
            sl = (
                cf.slope(human, f, k, e) if not ZSCORE else ek
            )  # z: gap in standard errors
            g.append((dk / sl, abs(ek / sl)))
        gaps[int(b)] = g
    flat = np.array([x for g in gaps.values() for x in g])
    if not len(flat) or np.isnan(flat).any():
        return np.inf, gaps
    return float(np.mean(flat[:, 0] ** 2)), gaps


def debiased(gaps):
    """sqrt(mean gap^2 - mean se^2): the RMS calibration error with sampling noise removed."""
    flat = np.array([x for g in gaps.values() for x in g])
    return float(np.sqrt(max(np.mean(flat[:, 0] ** 2) - np.mean(flat[:, 1] ** 2), 0)))


def funcs(par):
    """(T(elo), B(elo)). covp parameters (a, b, R0, None): B = -1 (the search's distribution) at
    and above rating R0, else 0 (the policy)."""
    a, b, c, d = par
    T = lambda e: a + b * (e - 1700) / 1000
    if d is None:
        return T, lambda e: np.where(np.asarray(e) >= c, -1.0, 0.0)
    return T, lambda e: c + d * (e - 1700) / 1000


def fit(A, human, f, family, half):
    best = (np.inf, None)
    ab = list(
        itertools.product(np.arange(0.4, 1.31, 0.05), np.arange(-0.6, 0.31, 0.05))
    )
    cd = {"temp": [(0, 0)], "tilt": list(itertools.product((0, 1, 2, 4, 8), (0, 2, 4, 8, 16))),
          "covp": [(r0, None) for r0 in (0, 1600, 2000, 2400, 9999)]}[family]  # fmt: skip
    for (a, b), (c, d) in itertools.product(ab, cd):
        par = (round(float(a), 2), round(float(b), 2), c, d)
        loss, _ = evaluate(A, human, f, *funcs(par), half)
        if loss < best[0]:
            best = (loss, par)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--dists", nargs="+", default=["policy"])
    p.add_argument("--human", default="positions-human.npz:mpv-human")
    p.add_argument("--budget", default="bullet=8,blitz=32,rapid=32,classical=128")
    p.add_argument("--families", default="temp")
    p.add_argument("--out", default="tune.json")
    p.add_argument(
        "--z", action="store_true", help="objective in standard errors, not Elo"
    )
    p.add_argument("--formats", default="bullet,blitz,rapid,classical")
    a = p.parse_args()
    global ZSCORE
    ZSCORE = a.z
    d = Path(a.dir)
    hp, hm = a.human.split(":")
    human = cf.summarize(
        cf.table(np.load(d / hp)["meta"], cf.load_mpv(d / hm), {}, {}), []
    )
    budgets = {
        cf.FORMATS.index(k): v for k, v in (x.split("=") for x in a.budget.split(","))
    }
    rows = position_grid(
        np.load(d / "positions.npz")["meta"],
        cf.load_mpv(d / "mpv"),
        cf.load_dists([d / x for x in a.dists]),
        budgets,
    )
    print(f"{len(rows)} positions")
    out = {}
    for fam in a.families.split(","):
        for f in range(4):
            if not any(r[0] == f for r in rows) or cf.FORMATS[f] not in a.formats.split(
                ","
            ):
                continue
            A = arrays(rows, f)
            res = {}
            for half in (0, 1):
                loss, par = fit(A, human, f, fam, half)
                test, gaps = evaluate(A, human, f, *funcs(par), 1 - half)
                res[half] = dict(params=par, train_rms=float(np.sqrt(loss)), test_rms=float(np.sqrt(test)),
                                 test_debiased=debiased(gaps),
                                 test_gaps={b: [(round(x), round(e)) for x, e in g] for b, g in gaps.items()})  # fmt: skip

            full_loss, full = fit(A, human, f, fam, None)
            res["all"] = dict(params=full, rms=float(np.sqrt(full_loss)))
            out[f"{fam}/{cf.FORMATS[f]}"] = res
            full_gaps = evaluate(A, human, f, *funcs(full))[1]
            res["all"]["debiased"] = debiased(full_gaps)
            res["all"]["gaps"] = {
                b: [(round(x), round(e)) for x, e in g] for b, g in full_gaps.items()
            }
            cv = np.sqrt(np.mean([res[h]["test_rms"] ** 2 for h in (0, 1)]))
            cvd = np.sqrt(np.mean([res[h]["test_debiased"] ** 2 for h in (0, 1)]))
            print(f"{fam:5s} {cf.FORMATS[f]:9s} {full}  in-sample {res['all']['rms']:4.0f} (debiased {res['all']['debiased']:4.0f})"
                  f"  cross-validated {cv:4.0f} (debiased {cvd:4.0f})  halves {res[0]['params']} {res[1]['params']}")  # fmt: skip
    (d / a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
