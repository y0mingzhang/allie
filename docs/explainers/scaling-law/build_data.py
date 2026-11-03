"""Build data.json, every number the scaling-law video draws, from the result files.

usage: python build_data.py  (needs results/ and ALLIE_DATA, as figures/make_figures.py --collect does)

Fits follow allie.experiments.isoflop and the sweep readout: Huber loss on log loss. Loss is the mean CE of the 16
format x rating cells, on the main evaluation (July 2026, strat-v1) or on September 2026 games (strat-v2).
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ALLIE_PROJECT_ROOT", HERE.parents[2]))
R = ROOT / "results"
X = R / "recipe10x"
LM = R / "lm-eval"
DATA = Path(os.environ["ALLIE_DATA"])
BATCH = 524288
BIG = "bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42"  # Allie 2.0's big run
ANN = "ann-all-p05-t1907"  # Allie 2.0 after its second anneal
V21 = "allie-2.1"  # Allie 2.1's released weights
SHAPES21 = ("additive", "coupled", "n_dep_data", "two_d_terms")
LR21 = 0.35  # Allie 2.1's learning rates, a multiple of the big run's base rates (DETAILS)
SIGMA = 0.0029


def jload(p):
    return json.loads(Path(p).read_text())


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def macro(run, golden="v1"):
    return jload(LM / run / f"strat-{golden}.json")["macro"]


def sig(v, n=5):
    return float(f"{v:.{n}g}")


def sigs(vs, n=5):
    return [sig(float(v), n) for v in vs]


def counts(L, d, k_dense=1):
    """Active non-embedding parameters of an S16 MoE."""
    H = round(8 * d / 3 / 16) * 16
    sh = round(H * 0.25 / 16) * 16
    ro = round((H - sh) / 16 / 16) * 16
    return 4 * L * d * d + k_dense * 3 * d * H + (L - k_dense) * (3 * d * (sh + 16 * ro) + 256 * d)


# ---------------------------------------------------------------- law forms


def add(t, n, d):
    return t[0] + np.exp(t[1]) * (n / 1e8) ** -t[2] + np.exp(t[3]) * (d / 1e9) ** -t[4]


def coupled(t, n, d):
    u, v = np.exp(t[1]) * (n / 1e8) ** -t[2], np.exp(t[3]) * (d / 1e9) ** -t[4]
    return t[0] + (u ** (1 / t[5]) + v ** (1 / t[5])) ** t[5]


def n_dep_data(t, n, d):
    x, z = n / 1e8, d / 1e9
    return t[0] + np.exp(t[1]) * x ** -t[2] + np.exp(t[3]) * x ** t[5] * z ** -t[4]


def two_d_terms(t, n, d):
    return add(t[:5], n, d) + np.exp(t[5]) * (d / 1e9) ** -t[6]


def no_floor(t, n, d):
    return add(np.r_[0, t], n, d)


def spec(name, ymin):
    lo, hi = [0.0, -12, 0.0, -12, 0.0], [ymin - 1e-4, 4, 3, 4, 3]
    g = [[e, a, al, b, be] for e in (ymin - 0.25, ymin - 0.05) for a in (-3, -1) for al in (0.3, 0.8)
         for b in (-4, -2) for be in (0.4, 1.0)]  # fmt: skip
    match name:
        case "additive":
            return lo, hi, g, add
        case "no_floor":
            return [-12, 0, -12, 0], [4, 3, 4, 3], [s[1:] for s in g], no_floor
        case "coupled":
            return lo + [0.2], hi + [5], [s + [q] for s in g[::2] for q in (0.5, 1.5, 3)], coupled
        case "n_dep_data":
            return lo + [-2], hi + [3], [s + [q] for s in g[::2] for q in (0.0, 0.5)], n_dep_data
        case "two_d_terms":
            return (lo[:4] + [0.6, -12, 0.0], hi[:4] + [3, 4, 0.6],
                    [s[:4] + [1.0, b, c] for s in g[::4] for b in (-6, -3) for c in (0.1, 0.3)], two_d_terms)  # fmt: skip
    raise KeyError(name)


def fit(name, n, d, y):
    lo, hi, g, f = spec(name, y.min())
    lo, hi = np.array(lo, float), np.array(hi, float)
    best = None
    for s in g:
        s = np.clip(np.array(s, float), lo + 1e-9, hi - 1e-9)

        def res(t):
            with np.errstate(all="ignore"):
                r = np.log(np.maximum(f(t, n, d), 1e-6)) - np.log(y)
            return np.where(np.isfinite(r), r, 1.0)

        m = least_squares(res, s, bounds=(lo, hi), loss="huber", f_scale=1e-3, x_scale="jac", xtol=1e-12, ftol=1e-12,
                          gtol=1e-12, max_nfev=4000)  # fmt: skip
        if best is None or m.cost < best.cost:
            best = m
    return best.x, f


def rmse(f, t, n, d, y):
    return float(np.sqrt(np.mean((f(t, n, d) - y) ** 2)))


# ---------------------------------------------------------------- the first sweep


def first_sweep():
    sw = jload(X / "sweep-readout-c8s200f0v4-w4.json")
    m = next(x for x in sw["metrics"] if x["metric"] == "macro")
    runs = sw["runs"]
    moe = [r for r in runs if r["fam"] == "s16"]
    a = {k: np.array([r[k] for r in moe], float) for k in ("n", "d", "c")}
    y = np.array([r["y"]["macro"] for r in moe])
    th = m["law"]["s16"]["theta"]
    E, A, al, B, be = np.exp(th[0]), np.exp(th[1]), th[2], np.exp(th[3]), th[4]
    law = lambda n, d: E + A * (n / 1e7) ** -al + B * (d / 1e8) ** -be
    ks = sorted((r["n"], r["c"] / (r["n"] * r["d"])) for r in runs)
    lnn, kap = np.log([k[0] for k in ks]), [k[1] for k in ks]
    kappa = lambda n: np.interp(np.log(n), lnn, kap)
    uniq = sorted({k[0] for k in ks})

    def optimum(c):
        o = minimize_scalar(lambda u: law(np.exp(u), c / (kappa(np.exp(u)) * np.exp(u))),
                            bounds=(np.log(1e5), np.log(c / 1e5)), method="bounded")  # fmt: skip
        n = float(np.exp(o.x))
        return float(o.fun), n, c / (kappa(n) * n)

    budgets = []
    for b in ("1e17", "3e17", "1e18"):
        cell = m["cells"]["s16"][b]
        rs = [r for r in moe if r["budget"] == b]
        win = [r for r in rs if r["shape"] in cell["fit"]]
        q = np.polyfit(np.log([r["n"] for r in win]), [r["y"]["macro"] for r in win], 2)
        pts = []
        for s in sorted({r["shape"] for r in rs}, key=lambda s: int(s.split("x")[1])):
            g = [r for r in rs if r["shape"] == s]
            pts.append(dict(shape=s, n=g[0]["n"], d=g[0]["d"], loss=sig(np.mean([r["y"]["macro"] for r in g])), seeds=len(g),
                            window=s in cell["fit"]))  # fmt: skip
        n_opt = cell["n_opt"]
        budgets.append(dict(c=sig(cell["c"]), points=pts, parabola=sigs(q, 6), n_opt=sig(n_opt), l_opt=sig(cell["l"]),
                            l_ci=sigs(cell["l_ci"])))  # fmt: skip

    ctx = dict(
        law=law,
        optimum=optimum,
        a=a,
        y=y,
        kappa=kappa,
        moe=moe,
        cells=[m["cells"]["s16"][b] for b in ("1e17", "3e17", "1e18")],
    )
    return dict(
        _src="results/recipe10x/sweep-readout-c8s200f0v4-w4.json: 45 runs (24 MoE, 21 dense); n active non-embedding parameters, d tokens, c training FLOPs, loss on the main test. law: its metrics[macro].law.s16.theta. Seeds' sd: .sigma. parabola: per budget, a quadratic in ln N through every run of the window of four sizes, as figures/make_figures.py's sweep. kappa: FLOPs per parameter and token, as its tokens_at()",
        runs=[
            dict(
                moe=r["fam"] == "s16",
                budget=r["budget"],
                shape=r["shape"],
                seed=r["seed"],
                n=r["n"],
                d=r["d"],
                c=sig(r["c"]),
                loss=sig(r["y"]["macro"]),
            )
            for r in runs
        ],  # fmt: skip
        compute_all=sig(sum(r["c"] for r in runs)),
        budgets=budgets,
        sigma=sig(m["seed_sd"], 3),
        law=dict(E=sig(E), A=sig(A), alpha=sig(al), B=sig(B), beta=sig(be), rmse=sig(m["law"]["s16"]["rmse"], 3)),
        kappa=dict(lnn=sigs(np.log(uniq), 6), k=sigs(kappa(np.array(uniq)), 5)),
    ), ctx


def backtest(ctx):
    """Rms error of the additive law refit on the two smaller budgets, on the third budget's runs."""
    moe, y = ctx["moe"], ctx["y"]
    n, d = ctx["a"]["n"], ctx["a"]["d"]
    sel = np.array([r["budget"] != "1e18" for r in moe])
    t, f = fit("additive", n[sel], d[sel], y[sel])
    err = f(t, n[~sel], d[~sel]) - y[~sel]
    return float(np.sqrt(np.mean(err**2)))


def fan(cells, cs):
    """Every E + k C^-g within chi2 <= 2.71 of the three best-size points (sd = 90% interval width / 3.29), E fixed."""
    C = np.array([t["c"] for t in cells])
    L = np.array([t["l"] for t in cells])
    S = np.array([t["l_ci"][1] - t["l_ci"][0] for t in cells]) / 3.29
    curves = []
    for E in np.r_[np.linspace(0.3, 1.25, 96), np.linspace(1.251, L.min() - 2e-4, 60)]:
        best = min(
            (least_squares(lambda p: (E + np.exp(p[0]) * (C / 1e18) ** -p[1] - L) / S, [np.log(max(L.min() - E, 1e-4)), g],
                           bounds=([-20, 0], [5, 3])) for g in (0.02, 0.1, 0.4)),
            key=lambda m: m.cost,
        )  # fmt: skip
        if 2 * best.cost <= 2.71:
            curves.append((float(E), float(np.exp(best.x[0])), float(best.x[1]), 2 * float(best.cost)))
    v = np.array([e + k * (cs / 1e18) ** -g for e, k, g, _ in curves])
    return curves, v.min(0), v.max(0)


# ---------------------------------------------------------------- the refit


def refit():
    """scale22's September fit (its P5): runs within x1.5 of the learning-rate law, each moved to that law's rate."""
    lr = jload(X / "scale22/lr_law-v2.json")
    nd, ref = lr["fits"]["NDq"], lr["ref"]
    tag = "-tema0.08g0.5"

    def best_lr(neff, d):
        f = dict(N=np.log2(neff / ref["N"]), D=np.log2(d / ref["D"]))
        f["D2"] = f["D"] ** 2
        return nd["q"][0] + sum(q * f[v] for q, v in zip(nd["q"][1:], nd["vars"]))

    def load(study):
        out = []
        for p in sorted(X.glob(f"{study}/results/*.json")):
            r = jload(p)
            g = LM / f"{r['name']}{tag}/strat-v2.json"
            if not g.exists():
                continue
            d, rate = r["steps"] * BATCH, r.get("lr", 1)
            neff = r["useful_training_flops"] / (6 * d)
            dx = float(np.log2(rate) - best_lr(neff, d))
            pen = (nd["A"][0] if dx < 0 else nd["A"][1]) * dx**2 / 1e3
            out.append(dict(name=r["name"], budget=r["budget"], shape=f"{r['layers']}x{r['width']}", seed=r["seed"], lr=rate,
                            n=float(counts(r["layers"], r["width"], 3)), d=float(d), c=r["useful_training_flops"], y=jload(g)["macro"],
                            neff=neff, dx=dx, pen=float(pen)))  # fmt: skip
        return out

    window = np.log2(1.5)
    seen = {}
    for x in load("s22*"):
        arm = x["name"].split("-")
        arm = arm[5] if len(arm) == 8 else "ctl"
        if (
            "-c4x-" in x["name"]
            and x["budget"] in ("1e17", "3e17", "1e18", "3e18")
            and (arm in ("ctl", "chk") or arm.startswith("lr"))
        ):
            seen.setdefault((x["budget"], x["shape"], x["seed"], x["lr"]), x)
    near = [p for p in seen.values() if abs(p["dx"]) <= window]
    anchors = [a for a in load("s22p4*") if abs(a["dx"]) <= window]

    best = []
    for b in ("1e17", "3e17", "1e18", "3e18"):
        pts = [p for p in near if p["budget"] == b]
        shapes = sorted({p["shape"] for p in pts}, key=lambda s: int(s.split("x")[1]))
        n = np.array([next(p["n"] for p in pts if p["shape"] == s) for s in shapes])
        y = np.array([np.mean([p["y"] - p["pen"] for p in pts if p["shape"] == s]) for s in shapes])
        q = np.polyfit(np.log(n), y, 2)
        u = -q[1] / (2 * q[0])
        best.append(dict(budget=b, c=sig(np.mean([p["c"] for p in pts])), n_opt=sig(np.exp(u)), l_opt=sig(np.polyval(q, u)),
                         edge=not n.min() < np.exp(u) < n.max(), shapes=shapes, n=sigs(n), loss=sigs(y)))  # fmt: skip

    n, d = np.array([p["n"] for p in near]), np.array([p["d"] for p in near])
    y = np.array([p["y"] - p["pen"] for p in near])
    n21, d21 = 1218183168.0, 213623.0 * BATCH
    ratio = {}
    for p in near + anchors:
        ratio.setdefault(p["n"], p["neff"] / p["n"])
    ratio[688521216.0] = 698.2e6 / 688521216.0
    ratio[n21] = 1230.6e6 / n21
    kn = np.array(sorted(ratio))
    neff = lambda x: x * np.interp(np.log(x), np.log(kn), [ratio[k] for k in kn])
    c21 = 6 * 1230.6e6 * 213623 * BATCH
    grid = np.geomspace(4e17, c21, 48)
    ngrid = np.geomspace(3e7, 3e10, 3000)
    best21 = best_lr(1230.6e6, 213623.0 * BATCH)
    dx21 = float(np.log2(LR21) - best21)
    price21 = float((nd["A"][0] if dx21 < 0 else nd["A"][1]) * dx21**2 / 1e3)
    forms = {}
    for name in (*SHAPES21, "no_floor"):
        t, f = fit(name, n, d, y)
        shift = float(np.mean([a["y"] - a["pen"] - f(t, np.array([a["n"]]), np.array([a["d"]]))[0] for a in anchors]))
        at = lambda c: float(np.min(f(t, ngrid, c / (6 * neff(ngrid)))) + shift)
        i = int(np.argmin(f(t, ngrid, c21 / (6 * neff(ngrid)))))
        forms[name] = dict(rmse=sig(rmse(f, t, n, d, y), 3), floor=None if name == "no_floor" else sig(t[0]), shift=sig(shift, 3),
                           at21=sig(f(t, np.array([n21]), np.array([d21]))[0] + shift), opt_n=sig(ngrid[i]),
                           opt_d=sig(c21 / (6 * neff(ngrid[i]))), curve=sigs([at(c) for c in grid]))  # fmt: skip
    return dict(
        _src="results/recipe10x/s22*/results/*.json and results/lm-eval/NAME-tema0.08g0.5/strat-v2.json (September 2026 games, each run's final weights blended half and half with their tail EMA), moved to the best rate of results/recipe10x/scale22/lr_law-v2.json's NDq law; forms and anchor shift as results/recipe10x/scale22/p5.py, whose p5-v2.json this reproduces. A run above 1.41 is off the plot. The forecast 1.2325-1.2396 was fixed on October 17 (DETAILS). no_floor was left out of it for its fit and optimum. lr21: that forecast assumes the NDq law's best rate at Allie 2.1's size and tokens (best, a multiple of the base rate); the run trained at ran, which the same law prices at price; lo, hi: the four forms' forecasts plus price, for the run as trained",
        runs=[dict(budget=p["budget"], c=sig(p["c"]), n=p["n"], d=p["d"], loss=sig(p["y"] - p["pen"]), shape=p["shape"], lr=p["lr"], seed=p["seed"]) for p in near],
        n_runs=len(near),
        max_move=sig(max(p["pen"] for p in near), 2),
        best=best,
        anchors=[dict(n=a["n"], d=a["d"], c=sig(a["c"]), loss=sig(a["y"] - a["pen"]), raw=sig(a["y"]), shape=a["shape"], lr=a["lr"]) for a in anchors],
        forms=forms,
        lr21=dict(best=sig(2**best21, 3), ran=LR21, price=sig(price21, 3),
                  lo=sig(min(forms[k]["at21"] for k in SHAPES21) + price21, 6), hi=sig(max(forms[k]["at21"] for k in SHAPES21) + price21, 6)),
        grid=sigs(grid),
        c21=sig(c21),
        n21=n21,
        d21=d21,
    )  # fmt: skip


def checks(ctx, traj, lr5, f20, y20, n20, d20):
    """What the miss was checked against."""
    sw = {(r["budget"], r["shape"], r["seed"]): r["y"]["macro"] for r in ctx["moe"]}
    gaps = {}
    for s in ("14x640", "18x896", "20x1024"):
        ref = np.mean([v for k, v in sw.items() if k[0] == "1e18" and k[1] == s])
        gaps[f"1e18/{s}"] = macro(f"v3g4-1e18-s16-{s}-c2-c8s200f0v4-s42") - ref
    ref = np.mean([v for k, v in sw.items() if k[0] == "1e17" and k[1] == "12x512"])
    gaps["1e17/12x512"] = np.mean([macro(f"v3g-1e17-s16-12x512-c2-c8s200f0v4-s{s}") for s in (42, 43)]) - ref
    tl, ts = [r["golden_macro"] for r in traj], [r["step"] for r in traj]
    sel = ctx["a"]["d"] / ctx["a"]["n"] >= 30
    t, f = fit("additive", ctx["a"]["n"][sel], ctx["a"]["d"][sel], ctx["y"][sel])
    decay = float(np.interp(lr5, ts, tl)) - y20
    starved = f20 - float(f(t, np.array([n20]), np.array([d20]))[0])
    return dict(
        _src="lr_end_sweep, lr_end_big: where each schedule stops, as a share of peak (docs/anim/data.json scaling.miss at commit 3c844055, DETAILS). moves: results/lm-eval/BIG/strat-v1.json counts. count_shift: the forecast with total or all parameters as N, moves by at most this (docs/anim/data.json scaling.miss.ruled_out at commit 3c844055). recipe: results/lm-eval/v3g*-c2-* (the big run's recipe at sweep sizes) minus the sweep's same size, main test; the top of the range, 0.0122, is 14x640 with one seed each, so it is one pair of runs. decay: the big run's loss where its learning rate reached 5% of peak (where the sweep's runs stop), interpolated in its trajectory, minus its final loss. It is read off that curve, not measured as an effect: the sweep's stop cuts a nearly straight line, and the decay is part of the recipe. starved: the forecast minus the law refit on runs with at least 30 tokens per parameter, tentative. rest: what is left",
        moves=sum(jload(LM / BIG / "strat-v1.json")["counts"].values()),
        count_shift=0.0003,
        recipe={k: sig(v, 3) for k, v in gaps.items()},
        recipe_range=[sig(min(gaps.values()), 2), sig(max(gaps.values()), 3)],
        lr_end_sweep=0.05,
        lr_end_big=0.001,
        decay=sig(decay, 3),
        starved=sig(starved, 3),
        rest=sig(f20 - y20 - decay - starved, 3),
    )


def demo():
    o = jload(DATA / "release-2.1/video-refresh/demo-21.json")["outputs"]
    return dict(
        _src="Allie.analyze of Allie 2.1 (int8, CPU): 1. e4 e5 2. Nf3, 1800 v 1750, 180+2, top legal moves. release-2.1/video-refresh/demo-21.json under ALLIE_DATA",
        fen=o["fen"],
        moves=[dict(san=s, p=p, surprise=sig(-np.log(p), 3)) for s, _, p in o["moves"][:5]],
    )


def main():
    first, ctx = first_sweep()
    law, optimum = ctx["law"], ctx["optimum"]
    traj20 = sorted(jsonl(X / "maia3-bench/bigrun-trajectory-v2.jsonl"), key=lambda r: r["step"])
    run20 = jload(R / f"pretrain/{BIG}/resume-config.json")["config"]
    steps20 = run20["scheduled_steps"]
    n20, d20, c20 = 688521216.0, steps20 * BATCH, traj20[-1]["useful_training_flops"]
    f20 = float(law(n20, d20))
    y20 = macro(BIG)
    tl = np.array([r["golden_macro"] for r in traj20])
    ts = np.array([r["step"] for r in traj20], float)
    cross = lambda v: float(np.interp(-v, -tl[tl.argmax() :], ts[tl.argmax() :]))
    lr5 = 2013 + (0.95 / 0.999) * (steps20 - 2013)  # linear decay from step 2013 to 0.1% of peak: where it is 5%
    top = first["budgets"][-1]

    cs = np.geomspace(4e17, 1.5e21, 64)
    curves, lo, hi = fan(ctx["cells"], cs)
    pick = sorted({int(round(i)) for i in np.linspace(0, len(curves) - 1, 15)})
    n, d, y = ctx["a"]["n"], ctx["a"]["d"], ctx["y"]
    others = {}
    for name in ("coupled", "n_dep_data", "two_d_terms", "no_floor"):
        t, f = fit(name, n, d, y)
        others[name] = dict(rmse=sig(rmse(f, t, n, d, y), 3), f20=sig(f(t, np.array([n20]), np.array([d20]))[0]))
    good = [v["f20"] for v in others.values() if v["rmse"] < first["sigma"]] + [f20]

    out = dict(
        _src="Every number the video draws; build_data.py writes this. Loss is CE in nats: the mean of 16 format x rating cells. Main test: strat-v1 (July 2026). September: strat-v2",
        first=first,
        frontier=dict(
            _src="the law at its compute-optimal size, as figures/make_figures.py optimum()",
            c=sigs(cs),
            loss=sigs([optimum(c)[0] for c in cs]),
        ),
        forecast20=dict(
            _src="law(N, D) for the big run. N: as figures/make_figures.py run() counts it. ci90: noise bootstrap of the law, figures/law_ci.json. floor_ci: its E, DETAILS table",
            n=n20,
            d=d20,
            c=c20,
            f=sig(f20),
            ci90=jload(HERE.parent / "figures/law_ci.json")["f20"],
            floor_ci=[1.26, 1.28],
            distance=sig(c20 / top["c"], 3),
            gain_promised=sig(top["l_opt"] - f20, 3),
            gain_real=sig(top["l_opt"] - y20, 3),
        ),  # fmt: skip
        run20=dict(
            _src="results/recipe10x/maia3-bench/bigrun-trajectory-v2.jsonl: every checkpoint scored on the main test (the first at 22B tokens). final: results/lm-eval/BIG/strat-v1.json. released: after the second anneal. days, gpus: DETAILS",
            tokens=sigs([r["tokens"] for r in traj20]),
            c=sigs([r["useful_training_flops"] for r in traj20]),
            loss=sigs([r["golden_macro"] for r in traj20]),
            final=sig(y20),
            released=sig(macro(ANN)),
            days=6.6,
            gpus=8,
            pass_forecast=dict(
                step=round(cross(f20)), tokens=sig(cross(f20) * BATCH), frac=sig(cross(f20) / steps20, 3)
            ),
            pass_floor=dict(
                step=round(cross(first["law"]["E"])),
                tokens=sig(cross(first["law"]["E"]) * BATCH),
                frac=sig(cross(first["law"]["E"]) / steps20, 3),
            ),
            miss=sig(f20 - y20, 3),
            noise_multiple=sig((f20 - y20) / first["sigma"], 3),
        ),  # fmt: skip
        backtest=dict(
            _src="the additive law refit on the two smaller budgets: rms error on the third budget's runs, how far past the fit they are",
            distance_third=sig(top["c"] / first["budgets"][1]["c"], 3),
            rmse_third=sig(backtest(ctx), 3),
        ),
        forms20=dict(
            _src="other law forms fit to the same 24 runs. lo, hi: forms with rmse under the seeds' sd",
            additive=dict(rmse=first["law"]["rmse"], f20=sig(f20)),
            **others,
            lo=sig(min(good)),
            hi=sig(max(good)),
        ),  # fmt: skip
        fan=dict(
            _src="every E + k (C/1e18)^-g within chi2 <= 2.71 of the three best-size points, with E fixed; their sd is each point's 90% interval width / 3.29",
            floors=[sig(curves[0][0], 3), sig(curves[-1][0], 3)],
            curves=[
                dict(E=sig(curves[i][0]), k=sig(curves[i][1]), g=sig(curves[i][2]), chi2=sig(curves[i][3], 3))
                for i in pick
            ],
            c=sigs(cs),
            lo=sigs(lo),
            hi=sigs(hi),
            at20=[
                sig(float(np.interp(np.log(c20), np.log(cs), lo))),
                sig(float(np.interp(np.log(c20), np.log(cs), hi))),
            ],
        ),  # fmt: skip
        checks=checks(ctx, traj20, lr5, f20, y20, n20, d20),
        demo=demo(),
        refit=(rf := refit()),
        how_big=dict(
            _src="each law's compute-optimal split at Allie 2.1's compute. first: the first sweep's law. refit: the four forms' range. flat: largest gain any form sees from 2.1's shape to its optimum",
            first_n=sig(optimum(rf["c21"])[1]),
            first_d=sig(optimum(rf["c21"])[2]),
            refit_n=[min(rf["forms"][k]["opt_n"] for k in SHAPES21), max(rf["forms"][k]["opt_n"] for k in SHAPES21)],
            refit_d=[min(rf["forms"][k]["opt_d"] for k in SHAPES21), max(rf["forms"][k]["opt_d"] for k in SHAPES21)],
            flat=sig(max(rf["forms"][k]["at21"] - rf["forms"][k]["curve"][-1] for k in SHAPES21), 2),
        ),
        allie21=dict(
            _src="results/lm-eval/allie-2.1/strat-v1.json and strat-v2.json (released weights); final weights, the October 5 forecast (built up from the big run, for the same shape before its recipe changed, 90%, main test) and dates: DETAILS; n, d, finished: README",
            main=sig(macro(V21)),
            sept=sig(macro(V21, "v2")),
            final_main=1.2248,
            n=1.23e9,
            d=112e9,
            finished="October 28",
            refit_fixed="October 17",
            oct5=dict(date="October 5", central=1.241, lo=1.231, hi=1.251),
        ),  # fmt: skip
    )
    (HERE / "data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
