"""Final scaling sweep readout (SWEEP-PLAN.md section 2), per metric (golden macro first, then expert macro):
isoFLOP parabolas in log N per family and budget, N*(C) across budgets, a per-family L(N, D) refit with an
identifiability check, and the dense-to-match compute ratio of each family against the reference at the budgets and at a
node-week, with bootstrap errors and a hardware-offset test. Text on stdout, everything in --out JSON.

  readout.py [--ref dense] [--only FAM,..] [--metrics macro,expert_macro] [--nw 4e20] [--sigma 0.0015]
             [--boot 200] [--window W] [--out FILE] PATH..
  the sweep: readout.py 'sweep-[0-9]*-RECIPE' --out results/recipe10x/sweep-readout-RECIPE.json

PATH: result JSONs, or study dirs / globs relative to results/recipe10x (their results/*.json).
Family: a sweep arm's label prefix (s16-10x448 -> s16), else isoflop-v1's recipe, else from the arch: dense, or
E{experts}k{top-k}sh{shared % of the MLP}, q = quantile router.
N: active non-embedding matmul parameters from the study's frozen model.arch: 4 L d^2 attention + the SwiGLU MLP;
MoE layers (all but the first) count the shared expert, the k routed experts and the router (dense ~ 12 L d^2);
isoflop-v1 runs carry their own n_nonembed (12 L d^2). D = steps x 524288. C = logged useful training FLOPs
(6 N D if not logged).
Law: experiments.isoflop additive() (then polished, see fitlaw()), per family ("separate E") and per MoE /
reference pair with one floor ("shared E"). L*(C) maps (N, D) to FLOPs with kappa(N) = C / (N D) interpolated in
log N over the runs, flat beyond them. CM(C) = the reference's FLOPs to reach this family's compute-optimal loss,
over C (dmix.multiplier). The isoFLOP CM does the same on the parabola minima, the reference's L*(C) linear in
log C between its budgets (extended past its ends).
Errors, 90%: parametric bootstrap, loss + N(0, s^2), s = max(--sigma, pooled seed-repeat SD, the fit's
dof-corrected RMS); the law's also resamples runs within each budget (shown when every cell has >= 3 runs).
--nw 4e20 useful training FLOPs: one 8 x L40S node for 168 h at ~23% MFU (moe-perf/memmodel.md L40S fit at width
1536: dense 24.6% -> 4.3e20, S16 19.7% -> 3.5e20). The CM is FLOP-matched; throughput is separate.
Hardware offset test: one budget's losses shifted by -/+ s for both families (a per-GPU-type offset; each budget
runs on one type), refit, node-week CM.
"""

import argparse
import functools
import glob
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from allie.experiments import dmix
from allie.experiments import isoflop as fit
from allie.paths import ROOT

R, P, EVAL = (
    ROOT / "results/recipe10x",
    ROOT / "results/pretrain",
    ROOT / "results/lm-eval",
)

PNAMES = ("E", "A", "alpha", "B", "beta")
LOG = (True, True, False, True, False)  # experiments.isoflop's theta holds log E, log A, log B
BOUND = lambda y: [(-4, np.log(y.min())), (-6, 6), (0.02, 3), (-6, 6), (0.02, 3)]


@functools.cache
def arch(study):
    """A study's frozen model.arch module (pre-package studies froze it as modded_arch.py)."""
    f = study / "source/allie/model/arch.py"
    f = f if f.exists() else study / "modded_arch.py"
    if not f.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"arch{abs(hash(study))}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def files(paths):
    for p in paths:
        hits = sorted(glob.glob(str(Path(p) if Path(p).is_absolute() else R / p)))
        assert hits, f"no match: {p}"
        for h in map(Path, hits):
            yield from (
                [h] if h.suffix == ".json" else sorted((h / "results").glob("*.json"))
            )


def steady(out, steps):
    try:
        rows = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    except (OSError, ValueError):
        return None
    v = [
        x["tokens_per_second"]
        for x in rows
        if "tokens_per_second" in x and x["step"] > steps / 4
    ]
    return statistics.median(v) if v else None


def load(f, metrics):
    r = json.loads(f.read_text())
    if r.get("stop_after"):
        return None
    L, d, a = r["layers"], r["width"], arch(f.parent.parent)
    dims = a.moe_dims(d, r["arch"]) if a and r.get("arch") else None
    H = a.swiglu_hidden(d) if a else None
    if "n_nonembed" in r:
        n, fam = r["n_nonembed"], r.get("recipe", "dense")
    elif dims:
        e, k, routed, shared = dims[:4]
        n = (
            4 * L * d * d
            + 3 * d * H
            + (L - 1) * (3 * d * (shared + k * routed) + d * e)
        )
        fam = f"E{e}k{k}sh{round(100 * shared / H)}"
        fam += "q" * (r["arch"].get("moe_update") == "quantile")
    else:
        n, fam = L * (4 * d * d + 3 * d * H), "dense"
    if m := re.fullmatch(r"([a-z]\w*)-\d+x\d+", r.get("v", "")):
        fam = m[1]
    s = r.get("strat") or json.loads((EVAL / r["name"] / "strat-v1.json").read_text())
    out, D = P / r["name"], r.get("tokens") or r["steps"] * 524288
    gpu = (
        "orchard H100"
        if (out / "orchard.json").exists()
        else r.get("gpu_name") or r.get("gpu")
    )
    return dict(
        name=r["name"], fam=fam, budget=f"{float(r['budget']):.0e}".replace("e+", "e"), shape=f"{L}x{d}",
        seed=r.get("seed", 42), n=n, d=D, c=r.get("useful_training_flops") or 6 * n * D,
        gpu=re.sub(r"^NVIDIA (RTX )?", "", gpu or "?"), resumed=(out / "resume-config.json").exists(),
        tps=steady(out, r["steps"]), y={m: r["ce"]["move"] if m == "move" else s[m] for m in metrics},
    )  # fmt: skip


def pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, (5, 95))] if len(v) > 1 else None


def ci(v, f="{:.3g}"):
    return f"[{f.format(v[0])}, {f.format(v[1])}]" if v else "[-]"


def vertex(k):
    a, b, c = k
    return (-b / (2 * a), c - b * b / (4 * a)) if a > 0 else (np.nan, np.nan)


def cell(rs, m, s, reps, rng, w=0):
    """One family x budget: the parabola in log N, its vertex and draws (log N*, L*); without one (fewer than
    3 sizes, or N spanning under 1.5x) the measured best stands in, flagged. w >= 3: fit only the w sizes around
    the best seed-mean size (a wide grid is not parabolic in log N); each draw perturbs every run's loss by
    N(0, s), reselects the window and refits, so the intervals include the window choice."""
    n, y = np.array([r["n"] for r in rs], float), np.array([r["y"][m] for r in rs])
    ns = np.unique(n)

    def pick(y):
        if not w or len(ns) <= w:
            return np.ones(len(y), bool)
        j = int(np.argmin([y[n == v].mean() for v in ns]))
        lo = min(max(0, j - (w - 1) // 2), len(ns) - w)
        return np.isin(n, ns[lo : lo + w])

    sel, i = pick(y), int(np.argmin(y))
    j = int(np.argmin([y[n == v].mean() for v in ns]))
    x, yw = np.log(n[sel]), y[sel]
    span, sizes = float(np.exp(np.ptp(x))), len(np.unique(x))
    shape = lambda v: next(r["shape"] for r in rs if r["n"] == v)
    o = dict(
        runs=int(sel.sum()), sizes=sizes, grid_sizes=len(ns), window=w, fit=[shape(v) for v in np.unique(n[sel])],
        span=span, c=statistics.median(r["c"] for r in rs), best=rs[i]["shape"], best_y=float(y[i]),
        best_mean=shape(ns[j]), best_mean_y=float(y[n == ns[j]].mean()), edge=len(ns) > 1 and j in (0, len(ns) - 1),
        gpu=sorted({r["gpu"] for r in rs}), l=float(y[i]), parabola=False,
    )  # fmt: skip
    if sizes < 3 or span < 1.5:
        return o | dict(
            reps=np.stack([np.full(reps, np.log(n[i])), y[i] + rng.normal(0, s, reps)], 1)
        )
    k = np.polyfit(x, yw, 2)
    res, dof = yw - np.polyval(k, x), len(yw) - 3
    s = max(s, float(np.sqrt(res @ res / dof)) if dof > 0 else 0.0)
    xs, ls = vertex(k)
    if w:
        d = []
        for _ in range(reps):
            yd = y + rng.normal(0, s, len(y))
            sd = pick(yd)
            d.append(vertex(np.polyfit(np.log(n[sd]), yd[sd], 2)))
        d = np.array(d)
    else:
        d = [
            np.polyfit(x, np.polyval(k, x) + rng.normal(0, s, len(yw)), 2)
            for _ in range(reps)
        ]
        d = np.array([vertex(z) for z in d])
    inside = (d[:, 0] >= x.min()) & (d[:, 0] <= x.max())
    return o | dict(
        parabola=True, n_opt=float(np.exp(xs)), l=float(ls), s=s, resid=res.tolist(), dof=dof, reps=d,
        unbracketed=not x.min() <= xs <= x.max(), inside=float(inside.mean()), n_ci=pct(np.exp(d[:, 0])),
        l_ci=pct(d[:, 1]),
    )  # fmt: skip


def logc_at(L, Ls, lcs):
    """log C where the reference's L*(C) reaches L: piecewise linear in (L*, log C), extended past its ends."""
    i = np.argsort(Ls)
    Ls, lcs = np.asarray(Ls)[i], np.asarray(lcs)[i]
    j = int(np.clip(np.searchsorted(Ls, L) - 1, 0, len(Ls) - 2))
    return lcs[j] + (L - Ls[j]) * (lcs[j + 1] - lcs[j]) / (Ls[j + 1] - Ls[j])


def fitlaw(d, shared=False, start=None):
    """experiments.isoflop.additive (its 32 starts, or a warm start) on {"ours": a} or, shared E, {"ours": a, "ref": b},
    polished on its objective x1e6 at tight tolerances: the raw objective is ~1e-5, under L-BFGS-B's default
    gradient tolerance, so a warm start stops near where it began and bootstrap spreads come out 5-15x too narrow
    along the E-alpha / B-beta ridges. Returns {key: theta}."""
    b = BOUND(np.concatenate([x[:, 3] for x in d.values()]))
    b = [b[0], *b[1:] * len(d)] if shared else b
    t = (
        fit.additive(d, shared)[: len(b)]
        if start is None
        else np.clip(start, *np.array(b).T)
    )
    f = lambda t: 1e6 * fit.objective(t, d, shared)
    t = minimize(
        f, t, method="L-BFGS-B", bounds=b, options=dict(ftol=1e-15, gtol=1e-12)
    ).x
    return {k: v for k, v in fit.unpack(t, shared).items() if k in d} | {"theta": t}


def best(p, c, kap):
    f = lambda x: fit.predict(p, np.exp(x), c / (kap(np.exp(x)) * np.exp(x)))
    o = minimize_scalar(f, bounds=(np.log(1e5), np.log(c / 1e5)), method="bounded")
    return float(o.fun), float(np.exp(o.x))


def cm(pf, pr, c, kap):
    return dmix.multiplier(lambda x: best(pr, x, kap)[0], best(pf, c, kap)[0], c)


def ident(p, a, s, draws):
    """Local: Jacobian of the prediction (loss units) in experiments.isoflop's theta, SE at noise s, correlations.
    Global: 90% bootstrap range and the share of draws at a bound. Not identified: no dof, > 10% of draws at a
    bound, or a 90% range wider than the value (exponents) or than 3x (E, A, B)."""
    h, N, D = 1e-5, a[:, 0], a[:, 1]
    J = [
        (fit.predict(p + h * e, N, D) - fit.predict(p - h * e, N, D)) / (2 * h)
        for e in np.eye(5)
    ]
    J = np.stack(J, 1)
    sv = np.linalg.svd(J / np.linalg.norm(J, axis=0), compute_uv=False)
    cov = s * s * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    corr = cov / np.maximum(np.outer(se, se), 1e-300)
    bd, T = BOUND(a[:, 3]), np.array(draws)
    hit = [
        float(np.mean((T[:, j] - bd[j][0] < 1e-3) | (bd[j][1] - T[:, j] < 1e-3)))
        for j in range(5)
    ]
    r90 = [pct(np.exp(T[:, j]) if LOG[j] else T[:, j]) for j in range(5)]
    wide = lambda j: (
        r90[j] and (r90[j][1] > 3 * r90[j][0] if LOG[j] else np.ptp(r90[j]) > abs(p[j]))
    )
    return dict(
        points=len(a), dof=len(a) - 5, rank=int((sv > 1e-8 * sv[0]).sum()), cond=float(sv[0] / sv[-1]),
        se=se.tolist(), at_bound=hit, range90=r90,
        corr=[(PNAMES[i], PNAMES[j], float(corr[i, j])) for i in range(5) for j in range(i + 1, 5) if abs(corr[i, j]) > 0.9],
        not_identified=[PNAMES[j] for j in range(5) if len(a) <= 5 or hit[j] > 0.1 or wide(j)],
    )  # fmt: skip


class Readout:
    def __init__(self, runs, m, a):
        self.runs, self.m, self.a, self.rng = runs, m, a, np.random.default_rng(0)
        self.fams = sorted({r["fam"] for r in runs})
        self.budgets = sorted({r["budget"] for r in runs}, key=float)
        g = {}
        for r in runs:
            g.setdefault((r["fam"], r["budget"], r["shape"]), []).append(r["y"][m])
        rep = [v for v in g.values() if len(v) > 1]
        dof = sum(len(v) - 1 for v in rep)
        ss = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in rep)
        self.seed_sd = float(np.sqrt(ss / dof)) if dof else None
        self.s = max(a.sigma, self.seed_sd or 0)
        ks = sorted((r["n"], r["c"] / (r["n"] * r["d"])) for r in runs)
        self.kap = lambda n: np.interp(
            np.log(n), np.log([x for x, _ in ks]), [k for _, k in ks]
        )
        self.data = {
            f: np.array(
                [
                    (r["n"], r["d"], float(r["budget"]), r["y"][m])
                    for r in runs
                    if r["fam"] == f
                ]
            )
            for f in self.fams
        }
        self.noise = {
            f: self.rng.standard_normal((a.boot, len(x))) for f, x in self.data.items()
        }
        pick = lambda x: np.concatenate(
            [
                self.rng.choice(np.flatnonzero(x[:, 2] == b), (x[:, 2] == b).sum())
                for b in np.unique(x[:, 2])
            ]
        )
        self.picks = {f: [pick(x) for _ in range(a.boot)] for f, x in self.data.items()}
        self.out = dict(
            metric=m,
            window=a.window,
            sigma=self.s,
            seed_sd=self.seed_sd,
            seed_dof=dof,
            cells={},
            nstar={},
            law={},
            cm={},
        )
        print(f"\n== {m}: isoFLOP window {a.window or 'all sizes'}, noise s = {self.s:.4f} (floor {a.sigma}; seed repeats: {len(rep)} shapes, pooled sd "
              f"{'-' if self.seed_sd is None else f'{self.seed_sd:.4f}'})")  # fmt: skip

    def sample(self, f, j, mode, theta):
        """Bootstrap draw j of family f's runs: the fit's prediction + s_f x fixed normals, or runs resampled
        within each budget (the same draws for every law, so separate and shared E see the same data)."""
        x = self.data[f]
        if mode == "runs":
            return x[self.picks[f][j]]
        x = x.copy()
        x[:, 3] = fit.predict(theta, x[:, 0], x[:, 1]) + self.sd[f] * self.noise[f][j]
        return x

    def isoflop(self):
        m, out = self.m, self.out
        print(
            "isoFLOP parabola in log N per family and budget: N* and L* [90%], measured best, residuals x1e3"
        )
        self.cells = {f: {} for f in self.fams}
        for f in self.fams:
            for b in self.budgets:
                rs = [r for r in self.runs if r["fam"] == f and r["budget"] == b]
                if not rs:
                    continue
                c = self.cells[f][b] = cell(rs, m, self.s, self.a.boot, self.rng, self.a.window)
                sz = f"{c['sizes']} of {c['grid_sizes']} sizes {c['fit'][0]}..{c['fit'][-1]}" if c["window"] else f"{c['sizes']} sizes"
                head = f"  {f:12} {b:5} {c['runs']:2} runs {sz} (N x{c['span']:.2f})"
                tail = (f"best measured {c['best']} {c['best_y']:.4f}, seed-mean {c['best_mean']} {c['best_mean_y']:.4f}"
                        f"{' (grid edge)' * c['edge']}  {'/'.join(c['gpu'])}")  # fmt: skip
                if not c["parabola"]:
                    print(f"{head}  no parabola (needs 3 sizes over N x1.5)  {tail}")
                    continue
                nci = c["n_ci"] and [x / 1e6 for x in c["n_ci"]]
                res = " ".join(f"{1e3 * x:+.1f}" for x in c["resid"])
                print(f"{head}  N* {c['n_opt'] / 1e6:.1f}M {ci(nci, '{:.1f}')}{' UNBRACKETED' * c['unbracketed']} "
                      f"(in range in {c['inside']:.0%} of draws)  L* {c['l']:.4f} {ci(c['l_ci'], '{:.4f}')}  "
                      f"{tail}  resid {res} (dof {c['dof']})")  # fmt: skip
            out["cells"][f] = {
                b: {k: v for k, v in c.items() if k != "reps"}
                for b, c in self.cells[f].items()
            }
        print("N*(C) ~ C^a from the bracketed parabola minima")
        for f in self.fams:
            ok = [
                (b, c)
                for b, c in self.cells[f].items()
                if c["parabola"] and not c["unbracketed"]
            ]
            if len(ok) < 2:
                print(f"  {f:12} not identified: {len(ok)} bracketed budget(s)")
                continue
            lc = np.log([c["c"] for _, c in ok])
            e = float(np.polyfit(lc, [np.log(c["n_opt"]) for _, c in ok], 1)[0])
            d = np.array([c["reps"][:, 0] for _, c in ok])
            sl = pct(
                [
                    np.polyfit(lc, d[:, j], 1)[0]
                    for j in range(d.shape[1])
                    if np.isfinite(d[:, j]).all()
                ]
            )
            out["nstar"][f] = dict(exponent=e, ci=sl, budgets=[b for b, _ in ok])
            two = " (2 budgets: no residual check)" * (len(ok) == 2)
            print(
                f"  {f:12} a = {e:.3f} {ci(sl)} over {' '.join(b for b, _ in ok)}{two}"
            )

    def laws(self):
        a, out, kap = self.a, self.out, self.kap
        print(
            "L(N, D) = E + A (N/1e7)^-alpha + B (D/1e8)^-beta per family (separate E); node-week optimum N*, L*"
        )
        self.th, self.sd = {}, {}
        for f, x in self.data.items():
            if len(x) < 3:
                print(f"  {f:12} {len(x)} run(s): no fit")
                continue
            self.th[f] = fitlaw({"ours": x})["ours"]
            res = fit.predict(self.th[f], x[:, 0], x[:, 1]) - x[:, 3]
            rms = float(np.sqrt(np.mean(res**2)))
            self.sd[f] = max(
                self.s, rms * np.sqrt(len(x) / (len(x) - 5)) if len(x) > 5 else 0.0
            )
            out["law"][f] = dict(
                theta=self.th[f].tolist(), rmse=rms, s=self.sd[f], resid=res.tolist()
            )
        self.draws = {
            mode: [
                {
                    f: fitlaw({"ours": self.sample(f, j, mode, p)}, start=p)["ours"]
                    for f, p in self.th.items()
                }
                for j in range(a.boot)
            ]
            for mode in ("noise", "runs")
        }
        for f, p in self.th.items():
            x = self.data[f]
            idf = ident(p, x, self.sd[f], [t[f] for t in self.draws["noise"]])
            ex = pct([t[f][4] / (t[f][2] + t[f][4]) for t in self.draws["noise"]])
            tr, te = x[x[:, 2] < x[:, 2].max()], x[x[:, 2] == x[:, 2].max()]
            hold = None
            if len(tr) >= 6 and len(np.unique(tr[:, 2])) >= 2:
                q = (
                    fit.predict(fitlaw({"ours": tr})["ours"], te[:, 0], te[:, 1])
                    - te[:, 3]
                )
                hold = dict(rmse=float(np.sqrt(np.mean(q**2))), bias=float(q.mean()))
            lnw, nnw = best(p, a.nw, kap)
            nd = pct([best(t[f], a.nw, kap)[1] / 1e6 for t in self.draws["noise"]])
            out["law"][f] |= dict(ident=idf, n_exponent=float(p[4] / (p[2] + p[4])), n_exponent_ci=ex,
                                  holdout=hold, node_week=dict(n=nnw, l=lnw, n_ci_M=nd))  # fmt: skip
            v = [np.exp(p[j]) if LOG[j] else p[j] for j in range(5)]
            pr = "  ".join(
                f"{k} {w:.3g} {ci(r)}" for k, w, r in zip(PNAMES, v, idf["range90"])
            )
            print(f"  {f:12} {pr}  N* ~ C^{p[4] / (p[2] + p[4]):.3f} {ci(ex)}  node-week N* {nnw / 1e6:.0f}M "
                  f"{ci(nd, '{:.0f}')} L* {lnw:.4f}")  # fmt: skip
            cor = ", ".join(f"{i}-{j} {c:+.2f}" for i, j, c in idf["corr"]) or "none"
            bnd = (
                ", ".join(f"{k} {h:.0%}" for k, h in zip(PNAMES, idf["at_bound"]) if h)
                or "none"
            )
            print(f"  {'':12} {len(x)} runs, dof {idf['dof']}, rmse {1e3 * out['law'][f]['rmse']:.2f}e-3 (noise "
                  f"used {self.sd[f]:.4f}), rank {idf['rank']}/5, cond {idf['cond']:.1e}, SE alpha "
                  f"{idf['se'][2]:.2g} beta {idf['se'][4]:.2g}; |corr| > 0.9: {cor}; draws at a bound: {bnd}")  # fmt: skip
            hs = (
                f"; top budget held out: rmse {1e3 * hold['rmse']:.2f}e-3, bias {1e3 * hold['bias']:+.2f}e-3"
                if hold
                else ""
            )
            res = " ".join(f"{1e3 * w:+.1f}" for w in out["law"][f]["resid"])
            print(
                f"  {'':12} NOT IDENTIFIED: {', '.join(idf['not_identified']) or 'none'}{hs}; resid x1e3 {res}"
            )

    def pair(self, f, ref, data=None, start=None):
        d = data or self.data
        t = fitlaw({"ours": d[f], "ref": d[ref]}, True, start)
        return t["ours"], t["ref"], t["theta"]

    def multipliers(self):
        a, ref, kap, out, th = self.a, self.a.ref, self.kap, self.out, self.th
        if ref not in th:
            print(f"no CM: the reference family {ref} has no law")
            return
        cs = [
            (b, statistics.median(r["c"] for r in self.runs if r["budget"] == b))
            for b in self.budgets
        ]
        cs += [("1e19 FLOPs", 1e19), ("node-week", a.nw)]
        rc = list(self.cells[ref].values())
        for f in [f for f in th if f != ref]:
            few = min(c["runs"] for c in [*self.cells[f].values(), *rc]) < 3
            pf, pr, t0 = self.pair(f, ref)
            shared = {
                mode: [self.pair(f, ref, {f: self.sample(f, j, mode, pf), ref: self.sample(ref, j, mode, pr)}, t0)[:2]
                       for j in range(a.boot)]
                for mode in ("noise", "runs")
            }  # fmt: skip
            sep = {mode: [(t[f], t[ref]) for t in v] for mode, v in self.draws.items()}
            laws = dict(separate=((th[f], th[ref]), sep), shared=((pf, pr), shared))
            rms = {g: float(np.sqrt(np.mean((fit.predict(p, x[:, 0], x[:, 1]) - x[:, 3]) ** 2)))
                   for g, p, x in ((f, pf, self.data[f]), (ref, pr, self.data[ref]))}  # fmt: skip
            out["cm"][f] = dict(shared_fit=dict(E=float(np.exp(t0[0])), rmse=rms))
            print(f"CM of {f} over {ref} (FLOP-matched). Shared-E fit: E {np.exp(t0[0]):.3g}, rmse "
                  + ", ".join(f"{g} {1e3 * v:.2f}e-3 (separate {1e3 * out['law'][g]['rmse']:.2f}e-3)" for g, v in rms.items()))  # fmt: skip
            print(f"  {'':10} {'C':9} {'isoFLOP minima [90%]':54} | separate E [90% noise] [runs] | shared E [90% noise] [runs]"
                  + "  (runs [-]: a cell has < 3 runs)" * few)  # fmt: skip
            for b, c in cs:
                row, fc = {}, self.cells[f].get(b)
                if fc and len(rc) >= 2:
                    L, lcs = [x["l"] for x in rc], np.log([x["c"] for x in rc])
                    d = [
                        logc_at(fc["reps"][j, 1], [x["reps"][j, 1] for x in rc], lcs)
                        for j in range(a.boot)
                    ]
                    para = fc["parabola"] and all(x["parabola"] for x in rc)
                    row["isoflop"] = dict(
                        cm=float(np.exp(logc_at(fc["l"], L, lcs)) / fc["c"]), ci=pct(np.exp(d) / fc["c"]),
                        extrapolated=not min(L) <= fc["l"] <= max(L),
                        basis="parabolas" if para else "best measured, not optima",
                    )  # fmt: skip
                for k, ((p1, p2), dr) in laws.items():
                    row[k] = dict(
                        cm=cm(p1, p2, c, kap),
                        noise=pct([cm(u, v, c, kap) for u, v in dr["noise"]]),
                        runs=None
                        if few
                        else pct([cm(u, v, c, kap) for u, v in dr["runs"]]),
                    )
                out["cm"][f][b] = row | dict(c=c)
                iso = row.get("isoflop")
                left = "-"
                if iso:
                    left = f"{iso['cm']:.2f}x {ci(iso['ci'], '{:.2f}')}{' extrap.' * iso['extrapolated']} ({iso['basis']})"
                lw = [
                    f"{row[k]['cm']:.2f}x {ci(row[k]['noise'], '{:.2f}')} {ci(row[k]['runs'], '{:.2f}')}"
                    for k in laws
                ]
                print(f"  {b:10} {c:.2e} {left:54} | {lw[0]:29} | {lw[1]}")
            shift = {k: {} for k in laws}
            for b in self.budgets:
                for k in laws:
                    shift[k][b] = []
                for sgn in (-1, 1):
                    d = {g: self.data[g].copy() for g in (f, ref)}
                    for x in d.values():
                        x[x[:, 2] == float(b), 3] += sgn * self.s
                    u = self.pair(f, ref, d, t0)
                    shift["shared"][b].append(cm(u[0], u[1], a.nw, kap))
                    u = [fitlaw({"ours": d[g]}, start=th[g])["ours"] for g in (f, ref)]
                    shift["separate"][b].append(cm(u[0], u[1], a.nw, kap))
            out["cm"][f]["offset_test"] = shift
            for k, v in shift.items():
                print(f"  hardware offset test ({k} E): one budget's losses -/+ {self.s:.4f} for both families -> "
                      "node-week CM " + ", ".join(f"{b} {x[0]:.2f}x / {x[1]:.2f}x" for b, x in v.items()))  # fmt: skip


def header(runs, a, metrics):
    fams = sorted({r["fam"] for r in runs})
    ks = sorted((r["n"], r["c"] / (r["n"] * r["d"])) for r in runs)
    top = max(r["c"] for r in runs)
    print(f"readout of {len(runs)} runs: " + ", ".join(f"{f} {sum(r['fam'] == f for r in runs)}" for f in fams)
          + f"; budgets {' '.join(sorted({r['budget'] for r in runs}, key=float))}; reference {a.ref}")  # fmt: skip
    print("N = active non-embedding matmul params (MoE: shared + k routed experts + router); D = steps x 524288; "
          f"C = logged useful training FLOPs, kappa = C/(N D) {ks[0][1]:.2f} at {ks[0][0] / 1e6:.0f}M .. "
          f"{ks[-1][1]:.2f} at {ks[-1][0] / 1e6:.0f}M; node-week {a.nw:.2g} FLOPs = {a.nw / top:.0f}x the largest run")  # fmt: skip
    print(f"  {'family':12} {'budget':6} {'shape':8} seed {'N':>7} {'D':>6} {'C':>8}  {'gpu':13} tok/s  "
          + "  ".join(metrics))  # fmt: skip
    for r in sorted(
        runs, key=lambda r: (r["fam"], float(r["budget"]), r["n"], r["seed"])
    ):
        tps = f"{r['tps'] / 1e3:.0f}K" if r["tps"] else "-"
        print(f"  {r['fam']:12} {r['budget']:6} {r['shape']:8} {r['seed']:4} {r['n'] / 1e6:6.1f}M {r['d'] / 1e9:5.2f}B "
              f"{r['c']:.2e}  {r['gpu'] + '*' * r['resumed']:13} {tps:>5}  "
              + "  ".join(f"{r['y'][m]:.4f}" for m in metrics))  # fmt: skip
    print(
        "hardware per budget (training GPU; * = resumed, earlier incarnations' GPU not verified); steady tok/s ratios"
    )
    for b in sorted({r["budget"] for r in runs}, key=float):
        rs = [r for r in runs if r["budget"] == b]
        gp = {
            f: sorted({r["gpu"] + "*" * r["resumed"] for r in rs if r["fam"] == f})
            for f in sorted({r["fam"] for r in rs})
        }
        tps = {(r["fam"], r["shape"]): r["tps"] for r in rs if r["tps"]}
        ratio = [
            f"{f} {sh} {v / tps[a.ref, sh]:.2f}"
            for (f, sh), v in sorted(tps.items())
            if f != a.ref and (a.ref, sh) in tps
        ]
        print(f"  {b:5} " + "; ".join(f"{f}: {', '.join(v)}" for f, v in gp.items())
              + " MIXED GPU TYPES" * (len({r["gpu"] for r in rs}) > 1)
              + (f"; tok/s over {a.ref} at the same shape: {', '.join(ratio)}" if ratio else ""))  # fmt: skip


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("paths", nargs="+")
    p.add_argument("--ref", default="dense")
    p.add_argument("--only", default="")
    p.add_argument("--metrics", default="macro,expert_macro")
    p.add_argument("--nw", type=float, default=4e20)
    p.add_argument("--sigma", type=float, default=0.0015)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()
    assert a.window == 0 or a.window >= 3, "--window is 0 (all sizes) or >= 3"
    metrics = a.metrics.split(",")
    runs = [r for f in files(a.paths) if (r := load(f, metrics))]
    runs = [r for r in runs if not a.only or r["fam"] in a.only.split(",")]
    if not runs:
        sys.exit("no scored runs yet")
    header(runs, a, metrics)
    res = []
    for m in metrics:
        x = Readout(runs, m, a)
        x.isoflop()
        x.laws()
        x.multipliers()
        res.append(x.out)
    if a.out:
        Path(a.out).write_text(
            json.dumps(
                dict(runs=runs, nw=a.nw, ref=a.ref, metrics=res),
                indent=1,
                default=float,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
