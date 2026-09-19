"""Fit the isoflop-v1 grid: per-budget parabola minima and the additive law.

L(N, D) = E + A (N/1e7)^-alpha + B (D/1e8)^-beta with N non-embedding parameters and
D tokens. C = N*D is a compute proxy: 6ND ignores embeddings and attention.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

STUDY = Path("/home/yimingz3/src/allie/results/recipe10x/isoflop-v1")
RECIPES = ("ours", "qwen")


def load(metric):
    rows = [
        json.loads(p.read_text()) for p in sorted((STUDY / "results").glob("*.json"))
    ]
    return {
        r: np.array(
            [
                (x["n_nonembed"], x["tokens"], x["budget"], x["ce"][metric])
                for x in rows
                if x["recipe"] == r
            ]
        )
        for r in RECIPES
    }


def isoflop(data):
    out = {}
    for r, d in data.items():
        minima = []
        for c in sorted(set(d[:, 2])):
            s = d[d[:, 2] == c]
            if len(s) < 3:
                continue
            x = np.log(s[:, 0])
            a, b, k = np.polyfit(x, s[:, 3], 2)
            xs = -b / (2 * a)
            minima.append(
                dict(
                    budget=c,
                    points=len(s),
                    n_opt=float(np.exp(xs)),
                    loss_opt=float(k - b * b / (4 * a)),
                    interior=bool(a > 0 and x.min() <= xs <= x.max()),
                )
            )
        ok = [m for m in minima if m["interior"]]
        slope = (
            float(
                np.polyfit(
                    np.log([m["budget"] for m in ok]),
                    np.log([m["n_opt"] for m in ok]),
                    1,
                )[0]
            )
            if len(ok) >= 2
            else None
        )
        out[r] = dict(minima=minima, n_opt_exponent=slope)
    return out


def predict(p, n, d):
    return (
        np.exp(p[0])
        + np.exp(p[1]) * (n / 1e7) ** -p[2]
        + np.exp(p[3]) * (d / 1e8) ** -p[4]
    )


def unpack(theta, shared):
    if shared:
        return dict(ours=theta[:5], qwen=np.r_[theta[0], theta[5:]])
    return dict(ours=theta[:5], qwen=theta[5:])


def objective(theta, data, shared, delta=1e-3):
    ps = unpack(theta, shared)
    r = np.concatenate(
        [
            np.log(predict(ps[k], d[:, 0], d[:, 1])) - np.log(d[:, 3])
            for k, d in data.items()
        ]
    )
    return np.where(
        np.abs(r) <= delta, 0.5 * r * r, delta * (np.abs(r) - 0.5 * delta)
    ).sum()


def additive(data, shared, starts=None):
    lo = np.log(min(d[:, 3].min() for d in data.values()))
    law = [(-6, 6), (0.02, 3), (-6, 6), (0.02, 3)]
    bounds = [(-4, lo), *law, *law] if shared else [(-4, lo), *law] * 2
    if starts is None:
        grid = [
            np.r_[e, a, al, b, be]
            for e in (lo - 0.7, lo - 0.1)
            for a in (-1, 1)
            for al in (0.3, 0.8)
            for b in (-1, 1)
            for be in (0.3, 0.8)
        ]
        starts = [np.r_[g, g[1:]] if shared else np.r_[g, g] for g in grid]
    fits = [
        minimize(objective, s, args=(data, shared), method="L-BFGS-B", bounds=bounds)
        for s in starts
    ]
    return min(fits, key=lambda m: m.fun).x


def describe(theta, data, shared):
    ps = unpack(theta, shared)
    out = {}
    for r, d in data.items():
        p = ps[r]
        res = predict(p, d[:, 0], d[:, 1]) - d[:, 3]
        out[r] = dict(
            E=float(np.exp(p[0])),
            A=float(np.exp(p[1])),
            alpha=float(p[2]),
            B=float(np.exp(p[3])),
            beta=float(p[4]),
            n_opt_exponent=float(p[4] / (p[2] + p[4])),
            rmse=float(np.sqrt(np.mean(res**2))),
            max_abs=float(np.abs(res).max()),
        )
    return out


def best_loss(p, c):
    f = lambda x: predict(p, np.exp(x), c / np.exp(x))
    return minimize_scalar(
        f, bounds=(np.log(1e4), np.log(c / 1e4)), method="bounded"
    ).fun


def multiplier(theta, shared, c):
    ps = unpack(theta, shared)
    target = best_loss(ps["ours"], c)
    g = lambda lc: best_loss(ps["qwen"], np.exp(lc)) - target
    lo, hi = np.log(c) - 12, np.log(c) + 12
    return float(np.exp(brentq(g, lo, hi)) / c) if g(lo) > 0 > g(hi) else None


def holdout(data, shared):
    top = max(d[:, 2].max() for d in data.values())
    train = {r: d[d[:, 2] < top] for r, d in data.items()}
    ps = unpack(additive(train, shared), shared)
    return {
        r: float(
            np.sqrt(
                np.mean(
                    (
                        predict(ps[r], d[d[:, 2] == top, 0], d[d[:, 2] == top, 1])
                        - d[d[:, 2] == top, 3]
                    )
                    ** 2
                )
            )
        )
        for r, d in data.items()
    }


def bootstrap(data, theta, shared, budgets, n, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        sample = {r: d[rng.integers(len(d), size=len(d))] for r, d in data.items()}
        t = additive(sample, shared, starts=[theta])
        samples.append(
            [
                *[
                    v
                    for p in unpack(t, shared).values()
                    for v in (np.exp(p[0]), p[2], p[4], p[4] / (p[2] + p[4]))
                ],
                *[multiplier(t, shared, c) or np.nan for c in budgets],
            ]
        )
    s = np.array(samples, dtype=float)
    keys = [
        f"{r}_{k}" for r in RECIPES for k in ("E", "alpha", "beta", "n_opt_exponent")
    ] + [f"cm_{c:.0e}" for c in budgets]
    return {
        k: [float(np.nanpercentile(s[:, i], 5)), float(np.nanpercentile(s[:, i], 95))]
        for i, k in enumerate(keys)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metric", choices=("move", "expert2400", "expert2600"), default="move"
    )
    p.add_argument("--boot", type=int, default=200)
    a = p.parse_args()
    data = load(a.metric)
    budgets = (1e17, 1e18, 1e19)
    report = dict(
        metric=a.metric,
        points={r: len(d) for r, d in data.items()},
        isoflop=isoflop(data),
    )
    for shared in (True, False):
        theta = additive(data, shared)
        key = "shared_E" if shared else "separate_E"
        report[key] = dict(
            fit=describe(theta, data, shared),
            holdout_top_budget_rmse=holdout(data, shared),
            multiplier={f"{c:.0e}": multiplier(theta, shared, c) for c in budgets},
        )
        if shared and a.boot:
            report[key]["bootstrap_90"] = bootstrap(
                data, theta, shared, budgets, a.boot
            )
    (STUDY / f"fit-{a.metric}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
