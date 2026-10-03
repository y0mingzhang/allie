"""Regenerate every figure in README.md from the result files under results/.

usage: .venv/bin/python docs/make_figures.py [name ...]

Writes docs/figures/NAME.{png,svg} (all figures by default) and prints the numbers each
figure shows, so the README can be checked against this output.
"""

import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

plt.switch_backend("agg")

ROOT = Path(__file__).resolve().parents[1]

R = ROOT / "results"
X = R / "recipe10x"
OUT = ROOT / "docs" / "figures"
FINAL = (
    R / "pretrain/bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42"
)  # Allie-v3.0
OLD = R / "pretrain/bigrun-24x1536d75m4ship-s16-24x1536-c8s200f0v4-s42"
REPORT = X / "distill-v2/report.json"
SWEEP = X / "sweep-readout-c8s200f0v4-w4.json"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, PALE, EDGE = "#e9e8e2", "#c3c2b7", "#f3f2ee", "#d8d6cf"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
WASH, WASH_EDGE = "#eaf2fc", "#c9daf3"
MAIA = {"maia3-5m": "#b4b2aa", "maia3-23m": "#7d7b74", "maia3-79m": "#3d3c38"}
NAME = {"maia3-5m": "Maia-3 5M", "maia3-23m": "Maia-3 23M", "maia3-79m": "Maia-3 79M"}
RING = {"mec": SURFACE, "mew": 1.6}
HOLLOW = {"mfc": SURFACE, "mec": BLUE, "mew": 1.6}
MINUS = str.maketrans("-", "−")

plt.rcParams.update(
    {
        "font.family": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "svg.fonttype": "path",
    }
)


def jload(p):
    return json.loads(Path(p).read_text())


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def md_rows(path):
    """Cells of every markdown table row in path (separator rows skipped)."""
    for line in Path(path).read_text().splitlines():
        if line.startswith("|") and not set(line) <= set("|-: "):
            yield [c.strip() for c in line.strip("|").split("|")]


def ci(s):
    """'+0.0055 [+0.0038, +0.0070]' -> (0.0055, 0.0038, 0.0070)."""
    return tuple(float(v) for v in re.findall(r"[-+]?\d*\.\d+", s)[:3])


def H(label, **kw):
    return Line2D([], [], label=label, **kw)


def note(ax, text, xy, dx=0, dy=0, **kw):
    kw = {"fontsize": 8.5, "color": INK2, "linespacing": 1.2} | kw
    return ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", **kw)


def titled(fig, title, sub, y=0.975, gap=0.07):
    fig.text(0.012, y, title, fontsize=13.5, color=INK, weight="bold", va="top")
    fig.text(0.012, y - gap, sub, fontsize=9.5, color=INK2, va="top", linespacing=1.3)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote docs/figures/{name}.png")


def logx(ax, ticks, fmt="{:g}".format):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))


def allie_n():
    """Active and total non-embedding matmul parameters of Allie-v3.0, and its tokens."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import modded_arch

    c = jload(FINAL / "resume-config.json")["config"]
    L, d = c["layers"], c["width"]
    e, k, routed, shared = modded_arch.moe_dims(d, c["arch"])[:4]
    base = 4 * L * d * d + 3 * d * modded_arch.swiglu_hidden(d) + (L - 1) * d * e
    act = base + (L - 1) * 3 * d * (shared + k * routed)
    tot = base + (L - 1) * 3 * d * (shared + e * routed)
    return act, tot, c["scheduled_steps"] * 524288


def macro_metric():
    return next(m for m in jload(SWEEP)["metrics"] if m["metric"] == "macro")


def law(fam):
    """The sweep's L(N, D) = E + A (N/1e7)^-alpha + B (D/1e8)^-beta for the main eval."""
    t = macro_metric()["law"][fam]["theta"]
    E, A, B = math.exp(t[0]), math.exp(t[1]), math.exp(t[3])
    return lambda n, d: E + A * (n / 1e7) ** -t[2] + B * (d / 1e8) ** -t[4]


# ------------------------------------------------------------------ 1. Pareto vs Maia-3


def pareto():
    rep = jload(REPORT)
    final, ann = "bigrun-143051", "ann-all-p05-t1907"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.8, bottom=0.2, wspace=0.2)
    titled(
        fig,
        "Prediction quality against inference compute: Allie-v3.0 and Maia-3",
        "Held-out Lichess blitz, July 2026: the same 80,000 positions for every model. "
        "Up and to the left: closer to the moves played, for less compute.",
    )
    for ax, key in zip(axes, ("ce", "acc")):
        at = {k: (v["gflops"], v[key]) for k, v in rep.items()}
        ce = key == "ce"
        mx = [at[k] for k in MAIA]
        ax.plot(*zip(*mx), color=MAIA["maia3-23m"], lw=1.2, zorder=2)
        for k, xy in zip(MAIA, mx):
            ax.plot(*xy, "o", ms=9, color=MAIA[k], zorder=4, **RING)
            if k == "maia3-79m":
                note(ax, NAME[k], xy, 0, -16, ha="center")
            else:
                note(ax, NAME[k], xy, 9, -3, va="center")
        ax.plot(*at[final], "o", ms=15, zorder=5, **HOLLOW)
        ax.plot(*at[ann], "*", ms=9, color=BLUE, mew=0, zorder=6)
        note(ax, "Allie-v3.0", at[final], 14, 0, va="center", color=INK)
        logx(ax, [0.5, 1, 2, 5, 10])
        ax.set_xlim(0.45, 14)
        ax.set_ylim(*((1.305, 1.195) if ce else (56.6, 59.6)))
        ax.set_xlabel("Inference compute per move (GFLOPs, log scale)")
        ylab = "Cross-entropy over legal moves (nats, flipped)"
        ax.set_ylabel(ylab if ce else "Top-1 accuracy (%)")
    handles = [
        H("Allie-v3.0", marker="o", ls="", ms=11, **HOLLOW),
        H("Allie-v3.0 (annealed)", marker="*", ls="", ms=9, color=BLUE, mew=0),
        H("Maia-3 5M / 23M / 79M", marker="o", ms=7, color=MAIA["maia3-23m"]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, columnspacing=2)
    save(fig, "pareto")
    for k in [*MAIA, final, ann]:
        r = rep[k]
        print(f"  {k:20s} {r['gflops']:.2f} GF  CE {r['ce']:.4f}  top-1 {r['acc']:.2f}")


# ------------------------------------------------------- 2. accuracy and CE by game rating


def rating():
    path = X / "bigrun-progress/acc-by-game-rating-final.json"
    rows = [r for r in jload(path) if r["n"]]
    x = np.array([np.mean([float(v) for v in r["bin"].split("-")]) for r in rows])
    thin = np.array([r["n"] < 3000 for r in rows])
    big = "bigrun-143051"
    series = [
        ("maia3-5m", "Maia-3 5M", MAIA["maia3-5m"], 1.6),
        ("maia3-23m", "Maia-3 23M", MAIA["maia3-23m"], 1.6),
        ("maia3-79m", "Maia-3 79M", MAIA["maia3-79m"], 2.0),
        (big, "Allie-v3.0 (1.39 GFLOPs)", BLUE, 2.6),
    ]
    grid = {"height_ratios": [2.1, 1]}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, gridspec_kw=grid)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.835, bottom=0.15)
    fig.subplots_adjust(wspace=0.22, hspace=0.08)
    titled(
        fig,
        "Accuracy and cross-entropy across the rating range",
        "All 402,108 scored blitz moves of the July 2026 eval, by game rating (mean of both "
        "players), reweighted to the natural player mix.\nBands: 95% intervals, "
        "bootstrapping whole games. Hollow markers: bins with fewer than 3,000 positions.",
        gap=0.05,
    )
    cols = (
        ("acc", 100, "Top-1 accuracy (%)", "minus 79M (pp)"),
        ("ce", 1, "CE, legal moves (nats, flipped)", "minus 79M (nats, flipped)"),
    )
    for (top, bottom), (key, scale, ylab, dlab) in zip(axes.T, cols):
        for name, label, color, lw in series:
            v = np.array([r["models"][name][key] for r in rows]) * scale
            z = 3 + (name == big)
            top.plot(x, v[:, 0], color=color, lw=lw, label=label, zorder=z)
            if name == big:
                top.plot(x[thin], v[thin, 0], "o", ls="", ms=5.5, zorder=6, **HOLLOW)
        top.set_ylabel(ylab)
        d = np.array([r["models"][big][f"d_{key}_79m"] for r in rows]) * scale
        bottom.axhline(0, color=MAIA["maia3-79m"], lw=1.2, zorder=2)
        bottom.fill_between(x, d[:, 1], d[:, 2], color=BLUE, alpha=0.14, lw=0)
        bottom.plot(x, d[:, 0], color=BLUE, lw=2, zorder=3)
        bottom.plot(x[thin], d[thin, 0], "o", ls="", ms=5, zorder=4, **HOLLOW)
        bottom.set_ylabel(dlab)
        bottom.set_xticks(np.arange(600, 3000, 400))
        bottom.set_xlim(600, 2900)
        bottom.set_xlabel("Game rating (Lichess blitz)")
    axes[0, 0].set_ylim(40, 66)
    axes[0, 1].set_ylim(1.9, 0.92)
    axes[1, 0].set_ylim(-4, 5)
    axes[1, 1].set_ylim(0.13, -0.13)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4)
    save(fig, "rating")
    for key, scale in (("ce", 1), ("acc", 100)):
        d = {r["bin"]: r["models"][big][f"d_{key}_79m"] for r in rows}
        worse = [(b, v[0] * scale) for b, v in d.items() if (v[0] > 0) == (key == "ce")]
        spans = [b for b, v in d.items() if v[1] < 0 < v[2]]
        print(f"  {key}: {len(d)} bins; 79M closer: {worse}; interval spans 0: {spans}")


# ------------------------------------------------------------------ 3. model diagram


def box(ax, x, y, w, h, text="", fc=PALE, ec=EDGE, fs=8.5):
    style = "round,pad=0,rounding_size=0.012"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.1))
    kw = {"ha": "center", "va": "center", "fontsize": fs, "color": INK}
    ax.text(x + w / 2, y + h / 2, text, linespacing=1.3, **kw)


def arrow(ax, a, b):
    style = {"arrowstyle": "-|>", "mutation_scale": 9, "color": MUTED, "lw": 1.1}
    ax.add_patch(FancyArrowPatch(a, b, **style))


def model():
    act, tot, _ = allie_n()
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 0.46))
    ax.axis("off")
    title = "Allie-v3.0: a mixture-of-experts transformer over the whole game"
    ax.text(0.012, 0.449, title, fontsize=13.5, color=INK, weight="bold", va="top")
    sub = (
        f"MoE {act / 1e9:.2f}B active / {tot / 1e9:.1f}B total parameters. Every block "
        "after the first sends each token to 16 of 256 small experts plus one shared expert."
    )
    ax.text(0.012, 0.417, sub, fontsize=9.5, color=INK2, va="top")
    head = {"fontsize": 7.5, "color": MUTED, "weight": "bold"}
    small = {"ha": "center", "va": "top", "color": INK2}
    ax.text(0.02, 0.365, "INPUTS AT EVERY MOVE", **head)
    for y, h, text in (
        (0.255, 0.095, "Tokens: header (time control,\nboth ratings as digits)\n+ every move so far"),
        (0.155, 0.085, "Clock: both players' time left,\nthe mover's last thinking time"),
        (0.055, 0.085, "Board: current position,\n8×8 planes through a small CNN"),
    ):  # fmt: skip
        box(ax, 0.02, y, 0.2, h, text)
        arrow(ax, (0.222, y + h / 2), (0.247, 0.2 + (y + h / 2 - 0.2) * 0.12))
    ax.add_patch(Circle((0.262, 0.2), 0.016, fc=SURFACE, ec=MUTED, lw=1.1))
    ax.text(0.262, 0.2, "+", ha="center", va="center", fontsize=11, color=INK2)
    trunk = "round,pad=0,rounding_size=0.014"
    ax.add_patch(
        FancyBboxPatch((0.295, 0.04), 0.365, 0.325, boxstyle=trunk, fc="#f7f9fd",
                       ec=WASH_EDGE, lw=1.2)
    )  # fmt: skip
    head_blue = head | {"color": BLUE}
    ax.text(0.308, 0.345, "24 TRANSFORMER BLOCKS, WIDTH 1536", **head_blue)
    arrow(ax, (0.278, 0.2), (0.313, 0.2))
    block1 = "Block 1\n\nattention\n+\ndense MLP"
    box(ax, 0.315, 0.07, 0.085, 0.26, block1, SURFACE, WASH_EDGE, 8)
    box(ax, 0.42, 0.07, 0.225, 0.26, "", SURFACE, WASH_EDGE)
    arrow(ax, (0.4, 0.2), (0.42, 0.2))
    cx = 0.5325
    ax.text(cx, 0.302, "Blocks 2-24", ha="center", fontsize=8.5, color=INK)
    ax.text(cx, 0.292, "attention over this game's moves", fontsize=7.5, **small)
    ax.text(cx, 0.274, "+ mixture-of-experts layer:", fontsize=7.5, **small)
    ax.text(cx, 0.247, "a router scores all 256 experts per token", fontsize=6.8,
            ha="center", va="top", color=MUTED)  # fmt: skip
    gx, gy, s = 0.438, 0.103, 0.0082
    on = set(np.random.default_rng(3).choice(256, 16, replace=False).tolist())
    for i in range(256):
        c = BLUE if i in on else "#dfe3ea"
        xy = (gx + i % 16 * s, gy + i // 16 * s)
        ax.add_patch(Rectangle(xy, s * 0.78, s * 0.78, fc=c, lw=0))
    ax.text(
        gx + 8 * s, gy - 0.012, "256 experts, 16 used per token", fontsize=7, **small
    )
    box(ax, 0.583, 0.103, 0.05, 0.131, "shared\nexpert", WASH, WASH_EDGE, 7)
    ax.text(0.695, 0.365, "OUTPUTS FROM ONE HEAD", **head)
    outputs = (
        (
            0.245,
            0.105,
            "Next move: a probability for each\nof 1,968 possible moves",
            WASH,
        ),
        (0.15, 0.075, "How long the move took (auxiliary)", PALE),
        (0.055, 0.075, "Game result: win, draw or loss (auxiliary)", PALE),
    )
    for y, h, text, fc in outputs:
        box(ax, 0.695, y, 0.285, h, text, fc, WASH_EDGE if fc == WASH else EDGE, 8.3)
        arrow(ax, (0.66, 0.2 + (y + h / 2 - 0.2) * 0.15), (0.695, y + h / 2))
    save(fig, "model")


# ------------------------------------------------------------------ 4. training curve


def training():
    path = X / "maia3-bench/bigrun-trajectory-v2.jsonl"
    rows = sorted(jsonl(path), key=lambda r: r["step"])
    act, _, D = allie_n()
    f = law("s16")
    forecast = f(act, D)
    ann = jload(REPORT)["ann-all-p05-t1907"]["golden"]["macro"]
    sweep_best = min(r["y"]["macro"] for r in jload(SWEEP)["runs"])
    t = np.array([r["tokens"] for r in rows]) / 1e9
    gm = np.array([r["golden_macro"] for r in rows])
    gap = np.array([r["paired_ce_79m"] for r in rows])
    grid = {"width_ratios": [1.15, 1]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw=grid)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.12, wspace=0.24)
    titled(
        fig,
        "Training Allie-v3.0: 75B tokens on one 8-GPU node, below its scaling-law forecast",
        "Left: main eval (16-cell macro CE, all formats), scored every second checkpoint "
        "from 22B tokens on. Right: blitz-benchmark CE\nminus Maia-3 79M's on the same "
        "80,000 positions, with 95% intervals.",
    )
    ax = axes[0]
    ds = np.linspace(8e9, 80e9, 200)
    ax.plot(ds / 1e9, [f(act, d) for d in ds], color=AXIS, lw=1.6, zorder=2)
    text = (
        "scaling law: where a finished run of D tokens\nshould land, at this model size"
    )
    note(ax, text, (12, f(act, 12e9)), 0, -7, color=MUTED, va="top", fontsize=8)
    ax.axhline(sweep_best, color=AQUA, lw=1.2, zorder=1)
    text = f"best model of the scaling sweep: {sweep_best:.4f}"
    note(ax, text, (1.5, sweep_best), 0, 3, fontsize=8)
    ax.plot(t, gm, color=BLUE, lw=2, zorder=3)
    ax.plot(t, gm, "o", ms=2.6, color=BLUE, mew=0, zorder=4)
    ax.plot(D / 1e9, forecast, "o", ms=9, mfc=SURFACE, mec=INK2, mew=1.6, zorder=5)
    note(ax, f"forecast {forecast:.4f}", (D / 1e9, forecast), 0, 9, ha="center")
    ax.plot(D / 1e9 + 1, ann, "*", ms=12, color=BLUE, zorder=6, **RING)
    right = {"ha": "right", "color": INK}
    note(ax, f"Allie-v3.0 {gm[-1]:.4f}", (t[-1], gm[-1]), -10, 1, va="bottom", **right)
    text = f"annealed {ann:.4f}"
    note(ax, text, (D / 1e9 + 1, ann), -12, -2, va="top", **right)
    ax.set(xlim=(0, 80), ylim=(1.235, 1.42))
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Main eval CE (nats)")
    ax = axes[1]
    ax.axhline(0, color=MAIA["maia3-79m"], lw=1.2, zorder=2)
    note(ax, "Maia-3 79M", (22, 0), 0, 4)
    ax.fill_between(t, gap[:, 1], gap[:, 2], color=BLUE, alpha=0.14, lw=0, zorder=2)
    ax.plot(t, gap[:, 0], color=BLUE, lw=2, zorder=3)
    cross = t[np.argmax(gap[:, 2] < 0)]
    nums = f"{gap[-1, 0]:+.4f} [{gap[-1, 1]:+.4f}, {gap[-1, 2]:+.4f}]".translate(MINUS)
    text = f"CE below Maia-3 79M's, beyond the interval,\nfrom {cross:.0f}B tokens; final {nums}"
    note(ax, text, (21.5, -0.006), va="top", color=INK)
    ax.set(xlim=(20, 78), ylim=(-0.03, 0.14))
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("CE minus Maia-3 79M (nats)")
    save(fig, "training")
    print(f"  N {act / 1e6:.1f}M, D {D / 1e9:.1f}B: forecast {forecast:.4f}")
    print(f"  final {gm[-1]:.4f} ({gm[-1] - forecast:+.4f}), annealed {ann:.4f}")
    print(f"  best sweep model {sweep_best:.4f}; below 79M from {cross:.1f}B tokens")
    print(f"  final gap to 79M {gap[-1]}")


# ------------------------------------------------------------------ 5. scaling sweep


def scaling():
    runs = jload(SWEEP)["runs"]
    cells = macro_metric()["cells"]
    act, _, D = allie_n()
    final = jload(R / f"lm-eval/{FINAL.name}/strat-v1.json")["macro"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.2, wspace=0.08)
    titled(
        fig,
        "Scaling sweep: mixture-of-experts reaches lower loss than dense at every budget",
        "45 runs, main-eval CE. Curves: quadratic in log N through the four sizes around "
        "each minimum (hollow points: outside that window). Same recipe\nfor both; the "
        "MoE sends each token to 16 of 256 experts plus a shared one. Fitted law's "
        f"forecast for Allie-v3.0: {law('s16')(act, D):.4f} (measured {final:.4f}).",
    )
    names = {"1e17": "6.3e17", "3e17": "1.9e18", "1e18": "6.2e18"}
    for ax, b in zip(axes, names):
        opt = {}
        for fam, color in (("dense", ORANGE), ("s16", AQUA)):
            by = {}
            for r in runs:
                if r["fam"] == fam and r["budget"] == b:
                    by.setdefault(r["shape"], (r["n"], []))[1].append(r["y"]["macro"])
            shapes = sorted(by, key=lambda s: by[s][0])
            pts = np.array([(by[s][0], np.mean(by[s][1])) for s in shapes])
            cell = cells[fam][b]
            fit = np.array([s in cell["fit"] for s in shapes])
            lx = np.log(pts[fit, 0])
            c = np.polyfit(lx, pts[fit, 1], 2)
            xs = np.linspace(lx.min() - 0.12, lx.max() + 0.12, 80)
            ax.plot(np.exp(xs), np.polyval(c, xs), color=color, lw=2, zorder=3)
            ax.plot(*pts[fit].T, "o", ms=5.5, color=color, mew=0, zorder=4)
            out = {"mfc": SURFACE, "mec": color, "mew": 1.3}
            ax.plot(*pts[~fit].T, "o", ls="", ms=5.5, zorder=4, **out)
            out["mfc"] = "none"
            ax.plot(cell["n_opt"], cell["l"], "o", ms=12, zorder=5, **out)
            opt[fam] = cell["l"]
            print(
                f"  {names[b]} {fam:5s} N* {cell['n_opt'] / 1e6:6.1f}M L* {cell['l']:.4f}"
            )
        ax.set_title(f"{names[b]} training FLOPs", fontsize=10, color=INK, loc="left")
        text = f"MoE optimum {opt['dense'] - opt['s16']:.3f} nats lower"
        ax.text(
            0.04, 0.95, text, transform=ax.transAxes, fontsize=8.5, color=INK2, va="top"
        )
        logx(ax, [1e7, 3e7, 1e8, 3e8], lambda v: f"{v / 1e6:g}M")
        ax.set_xlim(1.2e7, 5.5e8)
        ax.set_xlabel("Active parameters N")
    axes[0].set_ylim(1.295, 1.535)
    axes[0].set_ylabel("Main eval CE (nats)")
    handles = [
        H("MoE", marker="o", lw=2, ms=5.5, color=AQUA, mew=0),
        H("dense", marker="o", lw=2, ms=5.5, color=ORANGE, mew=0),
        H("fitted optimum", marker="o", ls="", ms=10, mfc="none", mec=INK2, mew=1.3),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    save(fig, "scaling")


# ------------------------------------------------------------------ 6. cheaper inference


def inference():
    rep = jload(REPORT)
    ks = {16: "bigrun-143051"} | {k: f"final-k{k}-trunc" for k in (12, 8, 6, 4, 2)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.16, wspace=0.26)
    titled(
        fig,
        "Inference cost: most experts can be skipped; search adds little",
        "Left: Allie-v3.0 computing only its best K of 16 routed experts per token, no retraining "
        "(blitz benchmark, 80,000 positions). Right: what\n128 simulations of tree search "
        "add, against the model's training compute (20,000 positions, 95% intervals).",
    )
    ax = axes[0]
    for k in ("maia3-23m", "maia3-79m"):
        ax.axhline(rep[k]["ce"], color=MAIA[k], lw=1.2, zorder=1)
        text = f"{NAME[k]} ({rep[k]['gflops']:.2f} GFLOPs)"
        note(ax, text, (16.6, rep[k]["ce"]), 0, -3, ha="right", va="top", fontsize=8)
    kk = sorted(ks)
    ce = [rep[ks[k]]["ce"] for k in kk]
    ax.plot(kk, ce, color=BLUE, lw=1.3, alpha=0.55, zorder=2)
    ax.plot(kk, ce, "o", ls="", ms=7, zorder=3, **HOLLOW)
    ax.set_xticks(kk, [f"{k}\n{rep[ks[k]]['gflops']:.2f}" for k in kk])
    ax.set(xlim=(1, 17), ylim=(1.268, 1.198))
    ax.set_xlabel("Routed experts per token, and GFLOPs per move")
    ax.set_ylabel("Cross-entropy over legal moves (nats, flipped)")
    ax = axes[1]
    flops = {r["name"]: r["c"] for r in jload(SWEEP)["runs"]}
    search = X / "maia3-bench/search"
    dense = [
        (flops[f"{r[2]}-c8s200f0v4-s42"], ci(r[5]))
        for r in md_rows(search / "ladder.md")
        if r[0] == "dense"
    ]
    moe = [
        (float(r[1]), ci(r[3]))
        for r in md_rows(search / "bigrun-bigrun.md")
        if len(r) == 5 and r[1][0].isdigit()
    ]
    for pts, color, name in (
        (dense, ORANGE, "dense sweep"),
        (moe[:3], AQUA, "MoE sweep"),
    ):
        c, g = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        err = [g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]]
        ax.plot(c, g[:, 0], color=color, lw=1.6, zorder=2)
        ax.errorbar(c, g[:, 0], err, fmt="o", ms=6.5, color=color, mew=0, zorder=3)
        note(ax, name, (c[0], g[0, 0]), -10, ha="right", va="center")
    (c2, g2), (c, g) = moe[2], moe[3]
    ax.plot([c2, c], [g2[0], g[0]], color=BLUE, lw=1.2, alpha=0.5, zorder=2)
    err = [[g[0] - g[1]], [g[2] - g[0]]]
    ax.errorbar([c], [g[0]], err, fmt="*", ms=14, color=BLUE, mew=0, zorder=4)
    text = f"Allie-v3.0: {g[0]:+.4f} nats"
    note(ax, text, (c, g[0]), -12, -4, ha="right", va="top", color=INK)
    logx(ax, [1e18, 1e19, 1e20], lambda v: f"1e{round(math.log10(v))}")
    ax.set(xlim=(8e16, 6e20), ylim=(0, 0.036))
    ax.set_xlabel("Training compute of the model (FLOPs, log scale)")
    ax.set_ylabel("CE gain from 128-simulation search (nats)")
    save(fig, "inference")
    for k, v in sorted(ks.items()):
        r = rep[v]
        print(
            f"  K={k:2d} {v:24s} {r['gflops']:.2f} GF  CE {r['ce']:.4f}  {r['acc']:.2f}%"
        )
    for name, pts in (("dense", dense), ("MoE", moe)):
        print(f"  search gain, {name}: " + ", ".join(f"{c:.3g}: {g}" for c, g in pts))


# ------------------------------------------------------------------ 7. small-scale ablations


def ablations():
    def macro(study, run):
        p = X / f"{study}-c8s200f0v4/results/{run}-c8s200f0v4-s42.json"
        return jload(p)["strat"]["macro"]

    run, elo = "nga-1e17-s16-12x512", "elo-1e17-s16-12x512"
    base = macro("elo-ab-1e17", elo)
    sigma = macro_metric()["seed_sd"]
    arms = [
        ("Decay the LR to 0.1% of peak, not 5%", "nga-l", f"{run}-d2z", True),
        ("No multi-token prediction", "nga-l", f"{run}-mtp0", True),
        ("Both ratings at every token (best of 5)", "elo-ab-1e17lr", f"{elo}-elo-lr0p1", False),
        ("Half the weight decay", "nga-l", f"{run}-wd0p5", False),
        ("No time-control tokens", "nga-l2", f"{run}-notc", False),
        ("Adam on every step, not every other", "nga-l", f"{run}-adamevery", True),
        ("Router-input centring", "nga-l4", f"{run}-center", True),
    ]  # fmt: skip
    d = [(name, macro(s, r) - base, ship) for name, s, r, ship in arms]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    fig.subplots_adjust(left=0.31, right=0.975, top=0.77, bottom=0.14)
    titled(
        fig,
        "One change at a time, at small scale",
        "Each bar: one change to an MoE with 39M active parameters trained on 2.5B tokens, "
        f"minus the unchanged run ({base:.4f});\nsame seed and data order. Gray band: ±1 "
        f"seed-to-seed standard deviation of a run ({sigma:.4f}). Blue: used in Allie-v3.0.",
    )
    ax.axvspan(-sigma, sigma, color="#f0efec", zorder=0)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    y = np.arange(len(d))[::-1]
    for yi, (_, v, ship) in zip(y, d):
        ax.barh(yi, v, height=0.5, color=BLUE if ship else "#bdbbb4", zorder=2)
        side = {"ha": "left"} if v > 0 else {"ha": "right"}
        note(
            ax,
            f"{v:+.4f}".translate(MINUS),
            (v, yi),
            5 if v > 0 else -5,
            va="center",
            **side,
        )
    ax.set_yticks(y, [n for n, *_ in d], fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.0045, 0.0072)
    text = "the two costly blue changes are\nkept: at width 1536 they prevent\nrouter collapse"
    ax.text(0.0071, 2.0, text, ha="right", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Change in main-eval CE (nats; negative is better)")
    save(fig, "ablations")
    for name, v, ship in d:
        print(f"  {v:+.4f}  {'adopted' if ship else '       '}  {name}")


# ------------------------------------------------------------------ 8. router collapse


def router():
    fig, ax = plt.subplots(figsize=(10, 4.0))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.14)
    titled(
        fig,
        "Router collapse at width 1536, and the fix",
        "Experts receiving under 10% of the mean load in the first mixture-of-experts layer "
        "(of 256), logged every 25 steps. Halving the router's\nlearning rate only delayed "
        "the collapse; centring the router's input and stepping Adam on every step "
        "stopped it.",
    )
    for run, color, label in ((OLD, RED, "first attempt"), (FINAL, BLUE, "Allie-v3.0")):
        rows = [r for r in jsonl(run / "train.jsonl") if r["step"] <= 3125]
        s = np.array([r["step"] for r in rows])
        v = np.array([r["moe_starved"][0] for r in rows])
        ax.plot(s, v, color=color, lw=2, zorder=3, label=label)
        print(f"  {label}: max {v.max():.0f}, last {v[-1]:.0f} at step {s[-1]}")
    ax.axvline(2013, color=AXIS, lw=1, zorder=1)
    text = "end of learning-rate warmup"
    note(ax, text, (2013, 112), -5, color=MUTED, fontsize=8, ha="right")
    note(
        ax,
        "first attempt stopped",
        (3125, 70),
        0,
        -10,
        fontsize=8,
        ha="center",
        va="top",
    )
    ax.set(xlim=(0, 3300), ylim=(-3, 125))
    ax.set_xlabel("Training step (524,288 tokens each)")
    ax.set_ylabel("Starved experts (of 256)")
    ax.legend(loc="upper left")
    save(fig, "router")


FIGS = {
    "pareto": pareto,
    "rating": rating,
    "model": model,
    "training": training,
    "scaling": scaling,
    "inference": inference,
    "ablations": ablations,
    "router": router,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        print(name)
        FIGS[name]()
