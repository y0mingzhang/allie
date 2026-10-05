"""DiL-piKL style outputs over a searched Q: the tilt pi ~ prior exp(beta Q) with each position's beta scaled by
the mover's predicted think time (the root's time head), beta_i = beta (t_i / t_cell)^kappa (t_cell: the cell's
median), or the mixture over the time head's draws of t, pi_i = E_t[pi_beta(t)]: a human thinking longer
searches more. Per cell, the CE at the beta matching the humans' accuracy minus raw, and the blunder Elo gap
there (scale_score.py's columns; kappa 0 is its ce_match).

python dil.py RUN [--budgets 256,1024] [--kappas 0,0.5,1] [--fill root|mean]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path[:0] = ["/home/yimingz3/src/allie-wt-seff/analysis/search_eff"]
import common as c  # noqa: E402
import evaluate as ev  # noqa: E402
from scale_score import CELLS, load  # noqa: E402

BIN = np.where(np.arange(63) < 16, np.arange(63) + 0.5, 16 * np.exp((np.arange(63) - 16) / 7.06))  # time head bins, s
EDGES = np.array([2, 5, 10, 20, 45, 90])  # draws pooled into these think-time groups


def think(heads):
    """(expected seconds, group probabilities [n, groups], group mean seconds [n, groups])."""
    z = heads[:, :63].astype(float)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    g = np.searchsorted(EDGES, BIN)
    w = np.stack([p[:, g == k].sum(1) for k in range(len(EDGES) + 1)], 1)
    m = np.stack([(p[:, g == k] * BIN[g == k]).sum(1) for k in range(len(EDGES) + 1)], 1) / np.maximum(w, 1e-12)
    return p @ BIN, w, m


def reverse(S, p, q, lam):
    """argmax_pi E_pi q - lam KL(p || pi) (Grill et al. 2020; allie.search.policy.output): pi = lam p / (alpha - q)."""
    top = np.maximum.reduceat(q, S.offsets[:-1])
    lo, hi = np.maximum.reduceat(q + lam * p, S.offsets[:-1]), top + lam
    for _ in range(60):
        mid = (lo + hi) / 2
        above = c.segsum(lam * p / np.maximum(mid[S.seg] - q, 1e-300), S.offsets) > 1
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    y = lam * p / np.maximum(((lo + hi) / 2)[S.seg] - q, 1e-300)
    return y / c.segsum(y, S.offsets)[S.seg]


def tilt(S, lp, q, beta):
    """pi ~ exp(lp + beta q), beta per flat move."""
    return c.segsoftmax(lp + beta * q, S.offsets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--budgets", default="256,1024")
    p.add_argument("--kappas", default="0,0.25,0.5,1")
    p.add_argument("--fill", default="root", choices=["root", "mean"])
    p.add_argument("--contrast", default="", help="gammas for the rating-contrast term (runs with per-view root policies)")
    p.add_argument("--contrast-view", type=int, default=1)
    p.add_argument("--only", default="", help="variants to score (default all)")
    a = p.parse_args()
    z, budgets = load(a.run)
    D = c.Data()
    pos = np.searchsorted(D.index, z["index"])
    S = ev.subset(D, pos)
    perm = np.empty(len(S.legal), np.int64)
    for r in range(S.n):
        a0, b0 = z["offsets"][r], z["offsets"][r + 1]
        slot = {
            int(t): S.offsets[r] + j
            for j, t in enumerate(S.legal[S.offsets[r] : S.offsets[r + 1]])
        }
        perm[a0:b0] = [slot[int(t)] for t in z["legal"][a0:b0]]
    S.prior = np.empty(len(S.legal))
    S.prior[perm] = z["prior"]
    pv = None
    if "prior_v" in z and a.contrast:
        pv = np.empty(len(S.legal))
        pv[perm] = z["prior_v"][:, a.contrast_view]
    w = z["heads"][:, 63:66].astype(float)
    w = np.exp(w - w.max(1, keepdims=True))
    v0 = (w[:, 0] - w[:, 2]) / w.sum(1)
    secs, gw, gm = think(z["heads"])
    print(
        f"{a.run}: CE - raw at matched accuracy / blunder Elo gap there ('x': unmatched)"
    )
    for b in [int(x) for x in a.budgets.split(",") if int(x) in budgets]:
        q = np.full(len(S.legal), np.nan)
        q[perm] = z[f"q{b}"]
        q = (
            ev.fill(S, q, v0)
            if a.fill == "mean"
            else np.where(np.isnan(q), v0[S.seg], q)
        )
        for f, bin_ in CELLS:
            if not ((S.f == f) & (S.bin == bin_)).any():
                continue
            C = ev.subset(S, np.flatnonzero((S.f == f) & (S.bin == bin_)))
            rows = C.rows
            qc, lp = q[C.flat], np.log(np.maximum(C.prior, 1e-30))
            h = C.hmet[:, :3].mean(0)
            raw = ev.expect(C, C.prior).mean(0)[3]
            t0 = np.median(secs[rows])
            line = []
            variants = [("tilt", lambda lb: tilt(C, lp, qc, np.exp(lb)))]
            for kappa in [float(x) for x in a.kappas.split(",") if float(x)]:
                s_ = ((secs[rows] / t0) ** kappa)[C.seg]
                sk = [((gm[rows, k] / t0) ** kappa)[C.seg] for k in range(gw.shape[1])]
                variants += [(f"k{kappa:g}", lambda lb, s_=s_: tilt(C, lp, qc, np.exp(lb) * s_)),
                             (f"k{kappa:g}mix", lambda lb, sk=sk: sum(gw[rows, k][C.seg] * tilt(C, lp, qc, np.exp(lb) * x)
                                                                       for k, x in enumerate(sk)))]  # fmt: skip
            variants.append(("rkl", lambda lb: reverse(C, C.prior, qc, np.exp(-lb))))
            variants.append(("half", lambda lb: 0.5 * C.prior + 0.5 * tilt(C, lp, qc, np.exp(lb))))
            if pv is not None:  # rating contrast: pi ~ p exp(beta Q + gamma (log p_view - log p))
                adv = np.log(np.maximum(pv[C.flat], 1e-30)) - lp
                for gamma in map(float, a.contrast.split(",")):
                    variants.append((f"con{gamma:g}", lambda lb, g=gamma: tilt(C, lp + g * adv, qc, np.exp(lb))))
            variants = [v for v in variants if not a.only or v[0] in a.only.split(",")]
            for name, pi in variants:
                grid = np.linspace(-4, 5, 10)
                acc = [ev.expect(C, pi(x)).mean(0)[0] - h[0] for x in grid]
                up = np.flatnonzero(np.array(acc) >= 0)
                if not len(up) or up[0] == 0:
                    line.append(f"{name} x{max(acc) * 100:+.1f}%")
                    continue
                lb = brentq(lambda x: ev.expect(C, pi(x)).mean(0)[0] - h[0], grid[up[0] - 1], grid[up[0]], xtol=1e-3)
                e = ev.expect(C, pi(lb)).mean(0)
                gap = ev.cells(C, np.tile(e, (C.n, 1)))[f, bin_]
                line.append(f"{name} {e[3] - raw:+.4f}/{gap['elo_bl'] - gap['elo']:+4.0f} b{np.exp(lb):.1f}")
            print(f"{b:5d} {c.FORMATS[f]:9s} {bin_}  " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
