"""Calibration eval: human vs candidate move quality by format and mover-Elo bin (CALIBRATION.md).

Every legal move has a Stockfish eval (calib_mpv.py, MultiPV, one depth). Per move: Lichess win%
loss, Lichess move accuracy, Lichess inaccuracy / mistake / blunder (win% drops of 5 / 10 / 15
points), engine top-1 / top-3 agreement, eval percentile among the legal moves, and capped ACPL
(undecided positions only). A candidate's value at a position is its exact expectation under its
move distribution; humans contribute the move played. Win%-loss distributions are compared per bin
by quantiles, Wasserstein-1 and KS. Each headline metric maps to Elo through that format's human
curve (from the large human-only set); calibration error = implied Elo - mover Elo.

python calib_fit.py CALIB_DIR [--dists policy search ...]
"""

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

FORMATS = ("bullet", "blitz", "rapid", "classical")
METRICS = (
    "wploss",
    "accuracy",
    "blunder",
    "mistake",
    "inaccuracy",
    "top1",
    "top3",
    "percentile",
    "acpl",
)
LOG = {"wploss", "blunder", "mistake", "inaccuracy", "acpl"}  # fitted on a log scale
HEADLINE = ("accuracy", "blunder", "top1")  # top1 is reported, but nearly flat in Elo
EDGES = np.linspace(0, 100, 1001)  # win%-loss histogram, 0.1-point bins
LIVE = 500


def solid(f, b):
    """Bins with at least 100 games (bullet also 2800)."""
    return 800 <= b <= 2600 or (f == 0 and b == 2800)


def chunks(d):
    for f in sorted(Path(d).glob("[0-9]*.npz")) + sorted(Path(d).glob("c[0-9]*.npz")):
        if ".partial" not in f.name:
            with np.load(f) as z:
                yield {k: z[k] for k in z.files}


def load_mpv(d):
    """position -> {move token: cp}."""
    out = {}
    for z in chunks(d):
        for n, i in enumerate(z["index"]):
            a, b = z["offsets"][n], z["offsets"][n + 1]
            out[int(i)] = dict(zip(z["tokens"][a:b].tolist(), z["cp"][a:b].tolist()))
    return out


def load_dists(dirs):
    """position -> {field: array over its legal moves, 'legal': tokens, 'heads': root heads}."""
    out = defaultdict(dict)
    for d in dirs:
        for z in chunks(d):
            per = [
                k
                for k in z
                if k not in ("index", "legal", "offsets", "heads", "seconds")
            ]
            for n, i in enumerate(z["index"]):
                a, b = z["offsets"][n], z["offsets"][n + 1]
                o = out[int(i)]
                legal = z["legal"][a:b].astype(int)
                if "legal" in o and not np.array_equal(
                    o["legal"], legal
                ):  # align to the first order
                    order = {t: k for k, t in enumerate(legal)}
                    perm = np.array([order[t] for t in o["legal"]])
                else:
                    o["legal"], perm = legal, slice(None)
                if "heads" in z:
                    o["heads"] = z["heads"][n].copy()
                for k in per:
                    o[k] = z[k][a:b][perm].copy()
    return out


def winp(cp):
    """Lichess win% of a centipawn eval."""
    return 50 + 50 * (2 / (1 + np.exp(-0.00368208 * np.asarray(cp, float))) - 1)


def move_metrics(cp):
    """[metric, move] for every legal move given their evals (mover's view)."""
    cp = np.asarray(cp, float)
    best = cp.max()
    w = winp(best) - winp(cp)
    acc = np.clip(103.1668 * np.exp(-0.04354 * w) - 3.1669, 0, 100)
    better = (cp[None, :] > cp[:, None]).sum(1)
    worse = (cp[None, :] < cp[:, None]).sum(1)
    ties = len(cp) - better - worse - 1
    pct = (worse + 0.5 * ties) / max(len(cp) - 1, 1)
    acpl = np.where(abs(best) < LIVE, best - cp, np.nan)
    rows = [
        w,
        acc,
        w >= 15,
        (w >= 10) & (w < 15),
        (w >= 5) & (w < 10),
        better == 0,
        better < 3,
        pct,
        acpl,
    ]
    return np.stack(rows).astype(float)


def table(meta, mpv, dists, candidates, hist=None):
    """Per position: metrics of the human move and each candidate's expectation; win%-loss
    histograms per (format, bin) and candidate accumulate into `hist`."""
    rows = []
    for i, cps in mpv.items():
        m = meta[i]
        d = dists.get(i, {})
        legal = d.get("legal", np.array(list(cps)))
        if int(m[8]) not in cps or set(legal.tolist()) != set(cps):
            continue
        mm = move_metrics([cps[t] for t in legal])
        h = mm[:, list(legal).index(int(m[8]))]
        r = dict(
            i=i,
            f=int(m[2]),
            bin=int(m[3]),
            elo=int(m[4]),
            game=(int(m[0]), int(m[1])),
            clock=int(m[7]),
            human=h,
        )
        key = (r["f"], r["bin"])
        if hist is not None:
            hist[key]["human"] += np.histogram(h[0], EDGES)[0]
        for name, fn in candidates.items():
            p = fn(d, m)
            if p is None:
                continue
            p = np.asarray(p, float) / np.sum(p)
            r[name] = np.where(np.isnan(mm), 0, mm) @ p
            r[name][-1] = np.nan if np.isnan(h[-1]) else r[name][-1]
            if hist is not None:
                hist[key][name] += np.histogram(mm[0], EDGES, weights=p)[0]
        rows.append(r)
    return rows


def summarize(rows, names):
    """(format, bin) -> n, games, mean mover elo, human mean/SE per metric, and per candidate its
    means and SE."""
    by = defaultdict(list)
    for r in rows:
        by[r["f"], r["bin"]].append(r)
    se = lambda x: np.nanstd(x, 0) / np.sqrt(np.maximum((~np.isnan(x)).sum(0), 1))
    out = {}
    for key, rs in sorted(by.items()):
        h = np.array([r["human"] for r in rs])
        s = dict(n=len(rs), games=len({r["game"] for r in rs}), elo=float(np.mean([r["elo"] for r in rs])),
                 human=np.nanmean(h, 0), human_se=se(h))  # fmt: skip
        for n in names:
            x = np.array([r[n] for r in rs if n in r])
            if len(x):
                hh = np.array([r["human"] for r in rs if n in r])
                s[n], s[n + "_se"], s[n + "_dse"] = np.nanmean(x, 0), se(x), se(x - hh)
        out[key] = s
    return out


def quantile(counts, q):
    c = np.cumsum(counts) / max(counts.sum(), 1e-12)
    return float(EDGES[1:][np.searchsorted(c, q)])


def distances(hist, names):
    """(format, bin) -> {name: (W1, KS, median, p90)} of win% loss, humans included."""
    out = {}
    for key, hs in hist.items():
        hc = hs["human"] / max(hs["human"].sum(), 1)
        o = {
            "human": (0.0, 0.0, quantile(hs["human"], 0.5), quantile(hs["human"], 0.9))
        }
        for n in names:
            if hs[n].sum() > 0:
                diff = np.cumsum(hc) - np.cumsum(hs[n] / hs[n].sum())
                o[n] = (
                    float(np.abs(diff).sum() * 0.1),
                    float(np.abs(diff).max()),
                    quantile(hs[n], 0.5),
                    quantile(hs[n], 0.9),
                )
        out[key] = o
    return out


def curve(human, f, k, key="human"):
    """Series `key`'s metric k against mean mover Elo over the solid bins: a weighted quadratic (on
    a log scale for losses and rates), made monotone in the direction of its overall trend."""
    pts = [
        (s["elo"], s[key][k], s["n"])
        for (g, b), s in human.items()
        if g == f and solid(g, b) and key in s
    ]
    x, y, w = (np.array(v, float) for v in zip(*pts))
    log = METRICS[k] in LOG
    c = np.polyfit((x - 1700) / 1000, np.log(y) if log else y, 2, w=np.sqrt(w))
    grid = np.linspace(x.min() - 600, x.max() + 600, 3001)
    fit = np.polyval(c, (grid - 1700) / 1000)
    fit = np.exp(fit) if log else fit
    up = np.polyfit(x, y, 1)[0] > 0
    fit = np.maximum.accumulate(fit) if up else np.minimum.accumulate(fit)
    return grid, fit, up


def implied(grid, fit, up, value):
    """Elo where the human curve takes `value` (clamped to the grid's ends)."""
    return float(np.interp(value, fit, grid) if up else np.interp(-value, -fit, grid))


def slope(human, f, k, elo):
    """d metric / d Elo of format f's human curve at elo (of the log metric for log metrics): the
    quadratic fit's derivative, kept to the overall trend's sign and at least a quarter of it."""
    pts = [
        (s["elo"], s["human"][k], s["n"])
        for (g, b), s in human.items()
        if g == f and solid(g, b)
    ]
    x, y, w = (np.array(v, float) for v in zip(*pts))
    y = np.log(y) if METRICS[k] in LOG else y
    c = np.polyfit((x - 1700) / 1000, y, 2, w=np.sqrt(w))
    d = np.polyval(np.polyder(c), (elo - 1700) / 1000) / 1000
    trend = np.polyfit(x, y, 1, w=np.sqrt(w))[0]
    return float(np.sign(trend) * max(np.sign(trend) * d, abs(trend) / 4))


def errors(human, summary, names, k):
    """Per candidate, format and solid bin: the paired gap (candidate minus human on the same
    positions) in Elo, through the human curve's local slope; per format the mean gap (bias) and the
    noise-debiased RMS, sqrt(mean(gap^2) - mean(se^2)); overall: formats equally weighted."""
    present = [
        f for f in range(4) if sum(g == f and solid(g, b) for g, b in summary) >= 3
    ]
    log = METRICS[k] in LOG
    out = {}
    for n in names:
        per, rms, bias = {}, {}, {}
        for f in present:
            gaps = []
            for (g, b), s in summary.items():
                if g != f or not solid(g, b) or n not in s:
                    continue
                h, x, dse = s["human"][k], s[n][k], s[n + "_dse"][k]
                d, e = (np.log(x / h), dse / h) if log else (x - h, dse)
                sl = slope(human, f, k, s["elo"])
                gaps.append((b, d / sl, abs(e / sl)))
            if gaps:
                g_, e_ = np.array([x[1] for x in gaps]), np.array([x[2] for x in gaps])
                per[FORMATS[f]] = {b: (round(v), round(e)) for b, v, e in gaps}
                rms[FORMATS[f]] = float(
                    np.sqrt(max(np.mean(g_**2) - np.mean(e_**2), 0))
                )
                bias[FORMATS[f]] = float(np.mean(g_))
        if rms:
            rms["overall"] = float(np.sqrt(np.mean([v**2 for v in rms.values()])))
        out[n] = dict(per_bin=per, rms=rms, bias=bias)
    return out


def power(p, t):
    p = np.asarray(p, float)
    if t == 0:
        q = (p == p.max()).astype(float)
        return q / q.sum()
    q = np.exp(np.log(np.maximum(p, 1e-30)) / t)
    return q / q.sum()


def temperature(d, m, t):
    return power(d["prior"], t) if "prior" in d else None


def think_seconds(heads):
    """Expected human think time (s) from the time head (allie.lichess.engine.Game.think's bins)."""
    z = np.asarray(heads[:63], float)
    p = np.exp(z - z.max())
    p /= p.sum()
    b = np.arange(63)
    sec = np.where(b < 16, b + 0.5, 16 * np.exp((b - 16) / 7.06))
    return float(p @ sec)


BUDGETS = (8, 32, 128, 512)


def budget(d, c):
    """Largest searched budget at most c * predicted think time (0: no search)."""
    want = c * think_seconds(d["heads"])
    have = [b for b in BUDGETS if b <= want and f"mcts_n{b}" in d]
    return have[-1] if have else 0


def visits(d, m, n, t=1.0):
    """Sample proportional to MCTS visits at budget n (the policy at n = 0)."""
    if "prior" not in d or (n and f"mcts_n{n}" not in d):
        return None
    v = d[f"mcts_n{n}"].astype(float) if n else d["prior"]
    return power(v / v.sum(), t) if v.sum() > 0 else d["prior"]


def tilt(d, m, n, beta, t=1.0, key="cov_q"):
    """pi ~ p^(1/t) exp(beta Q_n): the policy tilted by searched values (n = 0: the policy)."""
    if "prior" not in d or (n and f"{key}{n}" not in d):
        return None
    logit = np.log(np.maximum(d["prior"].astype(float), 1e-30)) / t
    if n:
        logit = logit + beta * d[f"{key}{n}"]
    q = np.exp(logit - logit.max())
    return q / q.sum()


def adaptive(d, m, c, beta=None):
    """Allie-paper budget: N = c * predicted think time, rounded down to a searched budget; sample
    the visits (beta None) or the coverage-Q tilt at that budget."""
    if "heads" not in d or "prior" not in d:
        return None
    n = budget(d, c)
    if beta is None:
        return visits(d, m, n)
    return tilt(d, m, n if f"cov_q{n}" in d else 0, beta)


TUNED_T = {
    0: (0.95, -0.05),
    1: (0.80, -0.35),
    2: (0.75, -0.30),
    3: (0.70, -0.50),
}  # temp family (calib_tune)
SPEEDS = ("bullet", "blitz", "rapid", "classical")


def tuned_temp(d, m):
    if "prior" not in d:
        return None
    a, b = TUNED_T[int(m[2])]
    return power(d["prior"], float(np.clip(a + b * (m[4] - 1700) / 1000, 0.2, 1.5)))


def rule_mode(d, m, par, hw=np.inf, ladder=(0, 8, 32, 128, 256), no_bullet=False):
    """The calibration rule (tau0, tau, k, gamma, cap[, x0]) on the stored outputs: the mixture over
    the budget ladder that the bot's random rounding draws from. x0 set: the hinge temperature
    exp(tau max(0, x - x0)), 1 below rating 1700 + 1000 x0."""
    import calibrated as c

    tau0, tau, k, gamma, cap, x0, alpha = (*par, None, 1.0)[:7] if len(par) < 7 else par
    if "prior" not in d or "heads" not in d:
        return None
    x = (int(m[4]) - 1700) / 1000
    t = float(np.exp(tau0 + tau * x if x0 is None else tau * max(x - x0, 0)))
    think = think_seconds(d["heads"])
    n = min(cap, k * think ** (alpha or 1.0) * np.exp(gamma * x), hw * think) if k and len(d["prior"]) > 1 else 0
    if no_bullet and int(m[2]) == 0:
        n = 0
    u = np.log2(1 + np.array(ladder))
    i = float(np.interp(np.log2(1 + n), u, np.arange(len(u))))
    lo, w = int(i), i - int(i)
    lad = [(ladder[lo], 1 - w)] + ([(ladder[lo + 1], w)] if w > 0 else [])
    most = hw * max(float(m[7]) - 1, 0) / 10 if m[7] >= 0 and np.isfinite(hw) else np.inf  # the clock's hard limit
    lad = [(max(b for b in ladder if b <= min(r, most)), w) for r, w in lad]
    if any(b and f"cov_p{b}" not in d for b, _ in lad):
        return None
    return sum(w * c.distribution(d[f"cov_p{b}"] if b else d["prior"], t) for b, w in lad)


def calibrated_mode(d, m):
    """calibrated.py's rule with its module parameters."""
    import calibrated as c

    return rule_mode(d, m, (0.0, c.TAU, c.K, c.GAMMA, c.CAP))


def candidates(search=False):
    """name -> fn(position's distributions, meta row) -> move distribution over its legal moves."""
    out = {
        f"T{t}": partial(temperature, t=t) for t in (0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.15)
    }
    out["tunedT"] = tuned_temp
    if search:
        out["calibrated"] = calibrated_mode
        out |= {
            f"cov{n}": (lambda n: lambda d, m: d.get(f"cov_p{n}"))(n)
            for n in (8, 32, 128)
        }
        out |= {f"vis{n}": partial(visits, n=n) for n in BUDGETS}
        out |= {
            f"tilt{n}b{b}": partial(tilt, n=n, beta=b)
            for n in (8, 32, 128)
            for b in (1, 2, 4, 8)
        }
        out |= {f"avis_c{c}": partial(adaptive, c=c) for c in (1, 2, 4, 8, 16, 32)}
        out |= {
            f"atilt_c{c}b{b}": partial(adaptive, c=c, beta=b)
            for c in (2, 8, 32)
            for b in (2, 4, 8)
        }
    return out


def human_figure(human, path):
    """Headline metrics by mover Elo, one line per format (solid bins), direct labels."""
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
    ink2, muted, grid, axis = "#52525b", "#71717a", "#e5e7eb", "#d4d4d8"
    colors = ("#2563eb", "#0d9488", "#d97706", "#7c3aed")
    plt.rcParams.update({
        "font.family": ["Nimbus Sans", "DejaVu Sans"], "font.size": 13, "text.color": "#27272a",
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": grid, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False, "xtick.labelcolor": muted,
        "ytick.labelcolor": muted, "xtick.major.size": 0, "ytick.major.size": 0, "svg.fonttype": "path",
    })  # fmt: skip
    panels = (("accuracy", "Mean move accuracy (Lichess, %)", "{:.0f}", None),
              ("blunder", "Blunder rate (win% drop of 15+)", "{:.1%}", (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)),
              ("top1", "Engine top-1 agreement", "{:.0%}", None))  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(left=0.05, right=0.93, top=0.95, bottom=0.14, wspace=0.3)
    for ax, (name, label, fmt, ticks) in zip(axes, panels, strict=True):
        k = METRICS.index(name)
        for f in range(4):
            keys = sorted(b for g, b in human if g == f and solid(g, b))
            x = [human[f, b]["elo"] for b in keys]
            y = [human[f, b]["human"][k] for b in keys]
            ax.plot(x, y, "-o", color=colors[f], lw=2, ms=4)
            ax.annotate(
                FORMATS[f],
                (x[-1], y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=colors[f],
                va="center",
            )
        if ticks:
            ax.set_yscale("log")
            ax.set_yticks(ticks)
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _, fmt=fmt: fmt.format(v).replace(".0%", "%"))
        )
        ax.set_ylabel(label)
        ax.set_xlabel("Mover's rating (Lichess)")
        ax.set_xlim(800, 3100)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def figure(human, summary, series, path):
    """Rows: accuracy, blunder rate; columns: formats. Humans (large set: points and fitted curve)
    and candidate series [(name, label, colour, dashes)] over the solid bins, at each bin's mean
    mover rating (the rating each candidate is conditioned on). Direct labels on the right."""
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
    ink, ink2, muted, grid, axis = "#27272a", "#52525b", "#71717a", "#e5e7eb", "#d4d4d8"
    plt.rcParams.update({
        "font.family": ["Nimbus Sans", "DejaVu Sans"], "font.size": 13, "text.color": ink,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": grid, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
        "xtick.labelcolor": muted, "ytick.labelcolor": muted, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.major.size": 0, "ytick.major.size": 0, "svg.fonttype": "path",
    })  # fmt: skip
    rows = (
        ("accuracy", "Move accuracy (Lichess, %)", None),
        ("blunder", "Blunder rate", (0.01, 0.02, 0.05, 0.1)),
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), sharex=True)
    fig.subplots_adjust(
        left=0.06, right=0.83, top=0.95, bottom=0.09, wspace=0.1, hspace=0.12
    )
    for row, (name, ylabel, ticks) in enumerate(rows):
        k = METRICS.index(name)
        for f in range(4):
            ax = axes[row, f]
            keys = sorted(b for g, b in human if g == f and solid(g, b))
            x = np.array([human[f, b]["elo"] for b in keys])
            h = np.array([human[f, b]["human"][k] for b in keys])
            g_, fit, _ = curve(human, f, k)
            inside = (g_ >= x.min()) & (g_ <= x.max())
            ax.plot(g_[inside], fit[inside], color=ink, lw=1.2, zorder=3)
            he = 1.96 * np.array([human[f, b]["human_se"][k] for b in keys])
            ax.vlines(x, h - he, h + he, color=ink, lw=1, zorder=4)
            ax.plot(x, h, "o", ms=5, color=ink, zorder=4)
            ends = [("Humans", fit[inside][-1], ink2)]
            for cand, label, color, dash in series:
                ks = [b for b in keys if (f, b) in summary and cand in summary[f, b]]
                xs = np.array([summary[f, b]["elo"] for b in ks])
                y = np.array([summary[f, b][cand][k] for b in ks])
                e = 1.96 * np.array([summary[f, b][cand + "_se"][k] for b in ks])
                ax.vlines(xs, y - e, y + e, color=color, lw=1, alpha=0.7, zorder=5)
                ax.plot(xs, y, "o", ms=4, color=color, zorder=6)
                cg, cfit, _ = curve(summary, f, k, cand)
                inside = (cg >= xs.min()) & (cg <= xs.max())
                ax.plot(cg[inside], cfit[inside], color=color, lw=2, ls=dash, zorder=5)
                ends.append((label, cfit[inside][-1], color))
            if ticks:
                ax.set_yscale("log")
                ax.set_yticks(ticks)
                ax.yaxis.set_minor_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            if row == 0:
                ax.set_title(
                    FORMATS[f].capitalize(), fontsize=14, color=ink2, loc="left"
                )
            if f == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
            ax.set_xticks([1000, 1600, 2200, 2800])
            ax.set_xlim(800, 2950)
        lo = min(a.get_ylim()[0] for a in axes[row])
        hi = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(lo, hi)
        tr = (np.log, np.exp) if ticks else (lambda v: v, lambda v: v)
        span, prev = tr[0](hi) - tr[0](lo), -np.inf
        for text, y0, color in sorted(ends, key=lambda t: t[1]):
            yv = max(tr[0](y0), prev + span / 13)
            prev = yv
            axes[row, -1].annotate(text, (1.0, y0), xycoords=("axes fraction", "data"), xytext=(1.05, tr[1](yv)),
                                   textcoords=("axes fraction", "data"), color=color, va="center", fontsize=13,
                                   arrowprops=dict(arrowstyle="-", color=axis, lw=0.8, shrinkA=2, shrinkB=2))  # fmt: skip
    fig.supxlabel(
        "Rating: the human's (Lichess), or the one Allie is conditioned on",
        fontsize=13,
        color=ink2,
    )
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def report(summary, names, k):
    print(f"\n{METRICS[k]} by bin: positions / games / mean elo / human / candidates")
    print(
        "format     bin     n games    elo   human "
        + " ".join(f"{n:>7s}" for n in names)
    )
    for (f, b), s in summary.items():
        vals = " ".join(f"{s[n][k]:7.4g}" for n in names if n in s)
        print(f"{FORMATS[f]:9s} {b:4d} {s['n']:5d} {s['games']:5d} {s['elo']:6.0f} {s['human'][k]:7.4g} {vals}"
              + ("" if solid(f, b) else "  (thin)"))  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--dists", nargs="+", default=["policy"])
    p.add_argument("--mpv", default="mpv")
    p.add_argument(
        "--human",
        default="positions-human.npz:mpv-human",
        help="large human-only set: positions:mpv dir",
    )
    p.add_argument("--out", default="calib.json")
    p.add_argument(
        "--positions",
        default="positions.npz",
        help="the paired set (candidates scored on it)",
    )
    p.add_argument(
        "--search",
        action="store_true",
        help="also the search candidates (needs search outputs)",
    )
    p.add_argument(
        "--figure",
        action="store_true",
        help="candidates.png: humans, T=1, tuned temperature, calibrated",
    )
    p.add_argument(
        "--rule", help="calib_rule.py output: its fitted families become candidates"
    )
    p.add_argument(
        "--rules",
        help="json {name: {params, hw, label, colour, dash}}: rule candidates, drawn in order",
    )
    a = p.parse_args()
    d = Path(a.dir)
    hp, hm = a.human.split(":")
    human = summarize(table(np.load(d / hp)["meta"], load_mpv(d / hm), {}, {}), [])
    print("human curves: format bin n games elo " + " ".join(METRICS))
    for (f, b), s in human.items():
        print(
            f"  {FORMATS[f]:9s} {b:4d} {s['n']:5d} {s['games']:5d} {s['elo']:6.0f} "
            + " ".join(f"{v:.4g}" for v in s["human"])
        )
    human_figure(human, d / "human-curves")
    cands = candidates(a.search)
    rules = json.loads((d / a.rules).read_text()) if a.rules else {}
    for name, r in rules.items():
        cands[name] = partial(rule_mode, par=tuple(r["params"]), hw=r.get("hw", np.inf),
                              ladder=tuple(r.get("ladder", (0, 8, 32, 128, 256))), no_bullet=r.get("no_bullet", False))
    if a.rule:
        fitted = json.loads((d / a.rule).read_text())
        for fam in ("temp", "search", "both"):
            if fam in fitted:
                cands[f"rule-{fam}"] = partial(
                    rule_mode, par=tuple(fitted[fam]["all"]["params"])
                )
    hist = defaultdict(lambda: defaultdict(lambda: np.zeros(len(EDGES) - 1)))
    rows = table(
        np.load(d / a.positions)["meta"],
        load_mpv(d / a.mpv),
        load_dists([d / x for x in a.dists]),
        cands,
        hist,
    )
    summary = summarize(rows, list(cands))
    print(f"\n{len(rows)} paired positions")
    for k in [METRICS.index(m) for m in HEADLINE]:
        report(summary, list(cands), k)
    err = {
        m: errors(human, summary, list(cands), METRICS.index(m))
        for m in (*HEADLINE, "wploss", "acpl")
    }
    dist = distances(hist, list(cands))
    print(
        "\nCalibration error (Elo): debiased RMS (mean signed gap) per format, overall RMS"
    )
    for m in HEADLINE:
        print(f"  {m}")
        for n in cands:
            e = err[m][n]
            print(f"    {n:14s} " + " ".join(f"{f[:4]} {e['rms'][f]:4.0f} ({e['bias'][f]:+5.0f})" for f in e["bias"])
                  + f"  all {e['rms'].get('overall', float('nan')):4.0f}")  # fmt: skip
    print("\nWin%-loss W1 (points), mean over solid bins per format:")
    for n in cands:
        w = {
            FORMATS[f]: np.mean(
                [
                    v[n][0]
                    for (g, b), v in dist.items()
                    if g == f and solid(g, b) and n in v
                ]
            )
            for f in range(4)
        }
        print(
            f"{n:10s} "
            + " ".join(f"{k[:4]} {v:.2f}" for k, v in w.items() if np.isfinite(v))
        )
    js = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
    out = dict(
        metrics=METRICS,
        human={
            f"{FORMATS[f]}/{b}": {k: js(v) for k, v in s.items()}
            for (f, b), s in human.items()
        },
        summary={
            f"{FORMATS[f]}/{b}": {k: js(v) for k, v in s.items()}
            for (f, b), s in summary.items()
        },
        errors=err,
        distances={f"{FORMATS[f]}/{b}": v for (f, b), v in dist.items()},
    )
    (d / a.out).write_text(json.dumps(out, indent=1) + "\n")
    if a.figure:
        series = [("T1.0", "Allie now (T = 1)", "#a1a1aa", "-")]
        series += [
            (n, r["label"], r["colour"], r.get("dash", "-")) for n, r in rules.items()
        ]
        if a.rule:
            series += [("rule-temp", "Temperature only", "#d97706", "--"), ("rule-search", "Search only", "#0d9488", "--"),
                       ("rule-both", "Temperature + search", "#2563eb", "-")]  # fmt: skip
        elif not rules:
            series.append(("tunedT", "Temperature by rating", "#d97706", "--"))
            if a.search:
                series.append(("calibrated", "Calibrated mode", "#2563eb", "-"))
        figure(human, summary, series, d / "candidates")


if __name__ == "__main__":
    main()
