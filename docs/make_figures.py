"""Regenerate the README figures from the result files.

usage: python docs/make_figures.py [name ...]

Writes docs/figures/NAME.{png,svg} (all figures by default) and prints the numbers each figure shows, so the
README can be checked against this output. The earlier Allie models' scores are read from ALLIE_ORIGINAL (a
directory of their benchmark and rating-set scores); without them their marks are left out.
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
from scipy.optimize import minimize_scalar

plt.switch_backend("agg")

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
X = R / "recipe10x"
DATA = Path(os.environ.get("ALLIE_DATA", "/data/group_data/dei-group/yimingz3/allie"))
ORIGINAL = Path(
    os.environ.get("ALLIE_ORIGINAL", "/home/yimingz3/allie-equiv/old-allie")
)
OUT = ROOT / "docs" / "figures"
FINAL = R / "pretrain/bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42"  # Allie 2.0
BENCH = (
    X / "distill-v2/report.json"
)  # every scored model on the 80,000 benchmark positions
SEARCH = DATA / "maia3-bench/search"
SWEEP = X / "sweep-readout-c8s200f0v4-w4.json"
PREVIOUS = {"allie1-medium": "Original Allie"}

INK, INK2, MUTED = "#27272a", "#52525b", "#71717a"
GRID, AXIS, SURFACE = "#e5e7eb", "#d4d4d8", "#ffffff"
BLUE, WASH, CHOSEN = "#2563eb", "#f3f7fe", "#dbe6fb"
MAIA = {"maia3-5m": "#b4b4bc", "maia3-23m": "#85858d", "maia3-79m": "#52525b"}
NAME = {"maia3-5m": "Maia-3 5M", "maia3-23m": "Maia-3 23M", "maia3-79m": "Maia-3 79M"}
DENSE = "#85858d"
SIZE = (7.2, 4.6)
C_ALLIE = (
    3.142088554411623e20
)  # Allie 2.0's useful training FLOPs (bigrun-trajectory-v2)
TEXT = 14
MINUS = str.maketrans("-", "−")

plt.rcParams.update(
    {
        "font.family": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": TEXT,
        "text.color": INK,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.labelsize": TEXT,
        "axes.labelpad": 9,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.size": 0,
        "xtick.major.pad": 7,
        "ytick.major.size": 0,
        "ytick.major.pad": 7,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "svg.fonttype": "path",
    }
)


def jload(p):
    return json.loads(Path(p).read_text())


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def figure(size=SIZE, **adjust):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(
        **({"left": 0.12, "right": 0.96, "top": 0.95, "bottom": 0.17} | adjust)
    )
    return fig, ax


def label(ax, text, xy, dx=0, dy=0, color=INK2, size=TEXT, **kw):
    kw = {"ha": "left", "va": "center", "fontsize": size, "color": color} | kw
    return ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", **kw)


def dot(ax, x, y, color, ms=9, hollow=False, z=5):
    face, edge = (SURFACE, color) if hollow else (color, SURFACE)
    ax.plot(x, y, "o", ms=ms, mfc=face, mec=edge, mew=2 if hollow else 1.2, zorder=z)


def save(fig, name, dpi=200):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote docs/figures/{name}.png")


def logx(ax, ticks, fmt="{:g}".format):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))


def power(v):
    """6.29e17 -> 6.3×10¹⁷, 1e18 -> 10¹⁸ (mathtext)."""
    e = math.floor(math.log10(v) + 1e-9)
    m = float(f"{v / 10**e:.2g}")
    return f"$10^{{{e}}}$" if m == 1 else rf"${m:g}\times10^{{{e}}}$"


def ends(ax, items, gap):
    """Direct labels right of the plot for (text, y) series ending at y: stacked at least gap apart, with a short
    leader to the series' own end when a label had to move."""
    items = sorted(items, key=lambda t: t[1])
    ys = [items[0][1]]
    for _, y in items[1:]:
        ys.append(max(y, ys[-1] + gap))
    for (text, y0), y in zip(items, ys):
        leader = {
            "arrowstyle": "-",
            "color": AXIS,
            "lw": 0.8,
            "shrinkA": 1,
            "shrinkB": 0,
        }
        ax.annotate(
            text, (1.0, y0), xytext=(1.03, y), xycoords=("axes fraction", "data"),
            textcoords=("axes fraction", "data"), fontsize=TEXT, color=INK2, va="center", ha="left",
            arrowprops=leader if abs(y - y0) > gap / 4 else None,
        )  # fmt: skip


def allie_n():
    """Active and total non-embedding matmul parameters of Allie 2.0, and its tokens."""
    sys.path[:0] = [
        str(ROOT / "src"),
        str(ROOT / "scripts"),
    ]  # this checkout's package, or its flat modules
    try:
        from allie.model import arch as model_arch
    except ImportError:
        import modded_arch as model_arch

    c = jload(FINAL / "resume-config.json")["config"]
    L, d = c["layers"], c["width"]
    e, k, routed, shared = model_arch.moe_dims(d, c["arch"])[:4]
    base = 4 * L * d * d + 3 * d * model_arch.swiglu_hidden(d) + (L - 1) * d * e
    act = base + (L - 1) * 3 * d * (shared + k * routed)
    tot = base + (L - 1) * 3 * d * (shared + e * routed)
    return act, tot, c["scheduled_steps"] * 524288


# ------------------------------------------------------------------------- the sweep's scaling laws


def macro():
    return next(m for m in jload(SWEEP)["metrics"] if m["metric"] == "macro")


def law(fam):
    """The sweep's L(N, D) = E + A (N/1e7)^-alpha + B (D/1e8)^-beta for the main eval."""
    t = macro()["law"][fam]["theta"]
    E, A, B = math.exp(t[0]), math.exp(t[1]), math.exp(t[3])
    return lambda n, d: E + A * (n / 1e7) ** -t[2] + B * (d / 1e8) ** -t[4]


def flops_per():
    """Training FLOPs per parameter and token, by size, as the sweep's runs measured it (experiments.readout);
    held at the end values outside the sweep's sizes."""
    ks = sorted((r["n"], r["c"] / (r["n"] * r["d"])) for r in jload(SWEEP)["runs"])
    return lambda n: np.interp(
        np.log(n), np.log([x for x, _ in ks]), [k for _, k in ks]
    )


def optimum(fam, c):
    """The law's compute-optimal (L*, N*, D*) at c training FLOPs, as experiments.readout.best finds it."""
    f, kap = law(fam), flops_per()
    o = minimize_scalar(
        lambda x: f(math.exp(x), c / (kap(math.exp(x)) * math.exp(x))),
        bounds=(math.log(1e5), math.log(c / 1e5)),
        method="bounded",
    )
    n = math.exp(o.x)
    return float(o.fun), n, c / (kap(n) * n)


def training():
    """Allie 2.0's main-eval CE over training, and the scaling law's forecast for the finished run."""
    rows = sorted(
        jsonl(X / "maia3-bench/bigrun-trajectory-v2.jsonl"), key=lambda r: r["step"]
    )
    t = np.array([r["tokens"] for r in rows]) / 1e9
    main = np.array([r["golden_macro"] for r in rows])
    act, _, D = allie_n()
    forecast = law("s16")(act, D)
    fig, ax = figure()
    ax.plot(t, main, color=BLUE, lw=2.6, zorder=4)
    dot(ax, t[-1], main[-1], BLUE)
    label(
        ax,
        f"Allie 2.0  {main[-1]:.4f}",
        (t[-1], main[-1]),
        -12,
        -2,
        color=INK,
        ha="right",
        va="top",
    )
    dot(ax, D / 1e9, forecast, MUTED, hollow=True)
    text = f"forecast\n{forecast:.4f}"
    label(
        ax, text, (D / 1e9, forecast), 0, 12, ha="center", va="bottom", linespacing=1.15
    )
    ax.set_xlim(20, 79)
    ax.set_ylim(1.24, 1.42)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Main-evaluation CE (nats)")
    save(fig, "training")
    print(
        f"  N {act / 1e6:.1f}M, D {D / 1e9:.1f}B: forecast {forecast:.4f}; final {main[-1]:.4f} from {t[0]:.0f}B"
    )


def isoflop():
    """Isoflop curves, MoE against dense, one panel per budget: quadratic fits in log N through each budget's
    selected four-size window (as the sweep readout fits them, every run), the window's seed means and the
    fitted minima."""
    sweep = jload(SWEEP)
    cells = macro()["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.0), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.87, bottom=0.2, wspace=0.08)
    for ax, b in zip(axes, ("1e17", "3e17", "1e18")):
        for fam, color in (("dense", DENSE), ("s16", BLUE)):
            cell = cells[fam][b]
            runs = [
                r
                for r in sweep["runs"]
                if r["fam"] == fam and r["budget"] == b and r["shape"] in cell["fit"]
            ]
            n = np.array([r["n"] for r in runs])
            y = np.array([r["y"]["macro"] for r in runs])
            c = np.polyfit(np.log(n), y, 2)
            xs = np.linspace(np.log(n).min() - 0.15, np.log(n).max() + 0.15, 80)
            ax.plot(np.exp(xs), np.polyval(c, xs), color=color, lw=2.2, zorder=3)
            means = np.array([(m, y[n == m].mean()) for m in np.unique(n)])
            ax.plot(*means.T, "o", ms=5.5, color=color, mew=0, zorder=4)
            v = -c[1] / (2 * c[0])
            ax.plot(
                math.exp(v),
                np.polyval(c, v),
                "o",
                ms=10,
                mfc="none",
                mec=color,
                mew=1.8,
                zorder=5,
            )
            print(
                f"  {b} {fam:5s} N* {math.exp(v) / 1e6:6.1f}M L* {np.polyval(c, v):.4f}"
            )
            if b == "1e17" and fam == "s16":
                label(
                    ax,
                    "MoE",
                    (math.exp(v), np.polyval(c, v)),
                    0,
                    -16,
                    ha="center",
                    color=color,
                )
            elif b == "1e17":
                label(
                    ax,
                    "dense",
                    (np.exp(xs[0]), np.polyval(c, xs[0])),
                    -2,
                    6,
                    ha="left",
                    va="bottom",
                    color=color,
                )
        ax.set_title(f"{power(cells['s16'][b]['c'])} FLOPs", fontsize=TEXT, color=INK2)
        logx(ax, [3e7, 1e8, 3e8], lambda v: f"{v / 1e6:g}M")
        ax.set_xlim(1.2e7, 5e8)
    axes[0].set_ylim(1.3, 1.46)
    axes[0].yaxis.set_major_locator(FixedLocator([1.3, 1.35, 1.4, 1.45]))
    axes[0].set_ylabel("Main-evaluation CE (nats)")
    axes[1].set_xlabel("Active parameters (log scale)")
    save(fig, "isoflop")


def law_curves(ax, value, top):
    """Each family's law at its compute-optimal size, solid over the sweep and dashed up to Allie 2.0's compute,
    with its isoflop minima and their 90% intervals; value(fam, c) and the cell's (point, interval) keys."""
    cells = macro()["cells"]
    point, ci = top
    for fam, color, name in (("dense", DENSE, "dense"), ("s16", BLUE, "MoE")):
        last = cells[fam]["1e18"]["c"]
        for lo, hi, ls in ((5e17, last, "-"), (last, C_ALLIE, (0, (4, 3)))):
            cs = np.geomspace(lo, hi, 40)
            ax.plot(cs, [value(fam, c) for c in cs], color=color, lw=2, ls=ls, zorder=3)
        for b in ("1e17", "3e17", "1e18"):
            cl = cells[fam][b]
            ax.errorbar(cl["c"], cl[point], yerr=[[cl[point] - cl[ci][0]], [cl[ci][1] - cl[point]]],
                        fmt="o", ms=7, color=color, mec=SURFACE, mew=1, elinewidth=1.4, capsize=0, zorder=4)  # fmt: skip


def frontier():
    """Main-eval CE against training compute: the isoflop minima, each family's law at its compute-optimal size,
    and Allie 2.0 against the law's forecast for its own size and tokens."""
    act, _, D = allie_n()
    c_allie = C_ALLIE
    final = jload(R / f"lm-eval/{FINAL.name}/strat-v1.json")["macro"]
    forecast = law("s16")(act, D)
    fig, ax = figure(right=0.97)
    law_curves(ax, lambda fam, c: optimum(fam, c)[0], ("l", "l_ci"))
    label(
        ax,
        "dense",
        (1.1e18, optimum("dense", 1.1e18)[0]),
        6,
        6,
        va="bottom",
        color=DENSE,
    )
    label(
        ax,
        "MoE",
        (1.1e18, optimum("s16", 1.1e18)[0]),
        -6,
        -8,
        ha="right",
        va="top",
        color=BLUE,
    )
    E = math.exp(macro()["law"]["s16"]["theta"][0])
    ax.axhline(E, color=AXIS, lw=1.2, ls=(0, (1.5, 2.5)), zorder=2)
    label(ax, "MoE fitted floor", (5e17, E), 0, -6, va="top", color=MUTED)
    ax.plot([c_allie, c_allie], [final, forecast], color=AXIS, lw=1.2, zorder=3)
    dot(ax, c_allie, forecast, BLUE, hollow=True)
    label(
        ax,
        f"forecast at its\nshape  {forecast:.4f}",
        (c_allie, forecast),
        12,
        0,
        linespacing=1.15,
    )
    dot(ax, c_allie, final, BLUE, ms=11, z=6)
    label(
        ax,
        f"Allie 2.0\n{final:.4f}",
        (c_allie, final),
        12,
        0,
        color=INK,
        linespacing=1.15,
    )
    logx(ax, [1e18, 1e19, 1e20], power)
    ax.set_xlim(4e17, 3e21)
    ax.set_ylim(1.24, 1.42)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("Training compute (FLOPs, log scale)")
    ax.set_ylabel("Main-evaluation CE (nats)")
    save(fig, "frontier")
    l_opt, n_opt, d_opt = optimum("s16", c_allie)
    print(
        f"  Allie 2.0 {c_allie:.3e} FLOPs: {final:.4f}, forecast {forecast:.4f}, law optimum {l_opt:.4f}"
    )
    print(
        f"  law optimum at that compute: N* {n_opt / 1e9:.2f}B, D* {d_opt / 1e9:.1f}B; floor E {E:.4f}"
    )


def moe_vs_dense():
    """Dense compute needed to match the MoE, as a ratio, from the two fitted laws, with 90% noise-bootstrap intervals."""
    cm = macro()["cm"]["s16"]
    fig, ax = figure(right=0.9)
    ax.axhline(1, color=DENSE, lw=1.4, zorder=2)
    label(ax, "dense", (1.0, 1), 6, 0, xycoords=("axes fraction", "data"))
    pts = [(cm[b]["c"], cm[b]["separate"], False) for b in ("1e17", "3e17", "1e18")]
    pts.append((cm["1e19 FLOPs"]["c"], cm["1e19 FLOPs"]["separate"], True))
    for c, s, ext in pts:
        ax.plot(
            [c, c], s["noise"], color=BLUE, lw=2.4, alpha=0.18 if ext else 0.3, zorder=3
        )
        dot(ax, c, s["cm"], BLUE, hollow=ext)
        text = f"{s['cm']:.1f}×" + ("\nextrapolated" if ext else "")
        label(
            ax,
            text,
            (c, s["cm"]),
            12 if ext else -12,
            0,
            ha="left" if ext else "right",
            linespacing=1.15,
        )
    logx(ax, [1e18, 1e19], power)
    ax.set_xlim(2.5e17, 4e19)
    ax.set_ylim(0.8, 4)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("Training compute (FLOPs, log scale)")
    ax.set_ylabel("Dense compute to match MoE (×)")
    save(fig, "moe-vs-dense")
    for c, s, _ in pts:
        print(f"  {c:.2e}: {s['cm']:.2f}x [{s['noise'][0]:.2f}, {s['noise'][1]:.2f}]")


def optimal():
    """Compute-optimal active parameters against training compute: the isoflop minima, each law's optimum, and
    Allie 2.0's size against the law's choice at its compute."""
    act, _, D = allie_n()
    c_allie = C_ALLIE
    fig, ax = figure(right=0.97)
    law_curves(ax, lambda fam, c: optimum(fam, c)[1], ("n_opt", "n_ci"))
    label(
        ax,
        "dense",
        (3e18, optimum("dense", 3e18)[1]),
        -8,
        10,
        ha="right",
        va="bottom",
        color=DENSE,
    )
    label(ax, "MoE", (1.1e18, optimum("s16", 1.1e18)[1]), 6, -8, va="top", color=BLUE)
    _, n_opt, d_opt = optimum("s16", c_allie)
    dot(ax, c_allie, n_opt, BLUE, hollow=True)
    text = f"extrapolated optimum\n{n_opt / 1e9:.1f}B, {d_opt / 1e9:.0f}B tokens"
    label(ax, text, (c_allie, n_opt), 12, 0, linespacing=1.15)
    dot(ax, c_allie, act, BLUE, ms=11, z=6)
    label(
        ax,
        f"Allie 2.0\n{act / 1e9:.2f}B, {D / 1e9:.0f}B tokens",
        (c_allie, act),
        12,
        0,
        color=INK,
        linespacing=1.15,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([3e7, 1e8, 3e8, 1e9, 3e9]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1e9:g}B" if v >= 1e9 else f"{v / 1e6:g}M")
    )
    logx(ax, [1e18, 1e19, 1e20], power)
    ax.set_xlim(4e17, 1.1e22)
    ax.set_ylim(1.5e7, 5e9)
    ax.set_xlabel("Training compute (FLOPs, log scale)")
    ax.set_ylabel("Active parameters")
    save(fig, "optimal")
    print(
        f"  law optimum at {c_allie:.3e}: {n_opt / 1e9:.2f}B on {d_opt / 1e9:.1f}B; Allie 2.0 {act / 1e9:.3f}B"
    )


# ------------------------------------------------------------------------------------ the model


def box(ax, x, y, w, h, text="", ec=AXIS, fc=SURFACE, size=12.5, color=INK, lw=1.2):
    style = "round,pad=0,rounding_size=0.08"
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2)
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            zorder=3,
        )


def arrow(ax, a, b, color=AXIS):
    style = {"arrowstyle": "-|>", "mutation_scale": 12, "color": color, "lw": 1.3}
    ax.add_patch(FancyArrowPatch(a, b, shrinkA=0, shrinkB=0, zorder=4, **style))


def model():
    """Inputs, one mixture-of-experts block (schematic), the next-move output; coordinates in inches."""
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 7.2), ylim=(0, 3.6))
    ax.axis("off")
    mid = 1.7
    small = {
        "ha": "center",
        "va": "center",
        "fontsize": 11.5,
        "color": INK2,
        "zorder": 4,
    }
    for y, text in ((2.55, "Game +\nratings"), (1.7, "Clock"), (0.85, "Board")):
        box(ax, 0.1, y - 0.3, 1.0, 0.6, text)
        arrow(ax, (1.1, y), (1.4, mid + (y - mid) * 0.35))
    box(ax, 1.4, 0.2, 4.0, 3.2, ec="#c9d8f5", fc=WASH)
    ax.text(3.4, 3.15, "MoE block  (23 of 24 blocks)", **small)
    box(ax, 1.55, mid - 0.3, 1.15, 0.6, "Attention")
    ax.plot(
        [2.7, 2.95, 2.95, 3.75], [mid, mid, 2.6, 2.6], color=BLUE, lw=1.3, zorder=3
    )  # the shared expert
    box(ax, 3.75, 2.45, 1.05, 0.3, ec=BLUE, fc=CHOSEN)
    ax.text(4.275, 2.82, "shared expert", **small | {"va": "bottom"})
    ax.plot([4.8, 5.15, 5.15], [2.6, 2.6, mid + 0.09], color=BLUE, lw=1.3, zorder=3)
    ax.plot(
        [2.7, 3.25], [mid, mid], color=MUTED, lw=1.3, zorder=3
    )  # the router and its routed experts
    ax.add_patch(plt.Circle((3.3, mid), 0.07, fc=MUTED, lw=0, zorder=5))
    ax.text(3.0, mid - 0.12, "router", **small | {"va": "top", "color": MUTED})
    for on, y in zip(
        [False, True, None, False, True, False], np.linspace(2.05, 0.75, 6)
    ):
        if on is None:
            ax.text(4.275, y, "⋮", **small | {"fontsize": 14, "color": MUTED})
            continue
        box(
            ax,
            3.75,
            y - 0.09,
            1.05,
            0.18,
            ec=BLUE if on else AXIS,
            fc=CHOSEN if on else SURFACE,
        )
        color, z = (BLUE, 3) if on else ("#e4e4e7", 2.5)
        ax.plot([3.3, 3.75], [mid, y], color=color, lw=1.3, zorder=z)
        ax.plot([4.8, 5.08], [y, mid], color=color, lw=1.3, zorder=z)
    ax.text(4.275, 0.42, "16 of 256 routed experts", **small)
    ax.add_patch(plt.Circle((5.15, mid), 0.09, fc=SURFACE, ec=MUTED, lw=1.2, zorder=5))
    ax.text(5.15, mid, "+", **small | {"fontsize": 10, "color": MUTED, "zorder": 6})
    arrow(ax, (5.24, mid), (5.7, mid), color=MUTED)
    box(ax, 5.7, mid - 0.4, 1.4, 0.8, "Next-move\ndistribution", ec=BLUE, lw=1.6)
    ax.text(
        6.4,
        mid - 0.55,
        "also: time spent,\ngame result",
        **small | {"va": "top", "color": MUTED, "linespacing": 1.2},
    )
    save(fig, "model")


# -------------------------------------------------------------------- comparison with earlier models


def previous(name, positions):
    """An earlier Allie's raw-policy legal-move CE per position on a set (bench: the 80,000 benchmark positions,
    rating: every scored blitz move), in that set's order; None if not scored."""
    f = ORIGINAL / positions / f"{name}.npz"
    if not f.exists():
        print(f"  (no scores at {f})")
        return None
    with np.load(f) as z:
        return np.asarray(z["ce_legal"], float)


def gflops(name):
    """An earlier Allie's GFLOPs per move with a key-value cache, counted as for the other models."""
    return (
        jload(ORIGINAL / f"flops-{name.split('-')[1]}.json")[
            "flops_per_move_incremental"
        ]
        / 1e9
    )


def search_index(n):
    """Benchmark positions the search ran on (legal.npz order), as run.py drew them."""
    files = sorted((SEARCH / "bigrun/legal").glob("[0-9]*.npz"))
    return np.concatenate([np.load(f)["index"] for f in files])[:n]


def pareto():
    """Legal-move CE against GFLOPs per move, on the 20,000 benchmark positions that search ran on."""
    rep = jload(SEARCH / "report-bigrun.json")
    pts = rep["points"]
    fig, ax = figure(left=0.13, right=0.97)
    maia = [(pts[f"maia/{m}"]["gflops"], pts[f"maia/{m}"]["ce"]) for m in MAIA]
    ax.plot(*zip(*maia), color=MAIA["maia3-23m"], lw=1.8, zorder=2)
    for m, xy, (dx, dy, va) in zip(
        MAIA, maia, ((10, 8, "bottom"), (10, 8, "bottom"), (10, 8, "bottom"))
    ):
        dot(ax, *xy, MAIA[m])
        label(ax, NAME[m], xy, dx, dy, va=va)
    line = [pts[k] for k in ("frozen/legal", "devcal/5", "devcal/128")]
    gx, gy = [p["gflops"] for p in line], [p["ce"] for p in line]
    ax.plot(gx, gy, color=BLUE, lw=2.6, zorder=3)
    for x, y, n in zip(gx[1:], gy[1:], ("5", "128")):
        dot(ax, x, y, BLUE, ms=7)
        label(
            ax, f"{n} sims", (x, y), 0, -11, ha="center", va="top", color=MUTED, size=13
        )
    dot(ax, gx[0], gy[0], BLUE, ms=11, z=6)
    label(
        ax, "Allie 2.0 (raw)", (gx[0], gy[0]), 0, -13, ha="center", va="top", color=INK
    )
    label(
        ax,
        "Allie 2.0 + search",
        (gx[2], gy[2]),
        0,
        12,
        ha="right",
        va="bottom",
        color=INK,
    )
    ix = search_index(rep["n"])
    for name, text in PREVIOUS.items():
        ce = previous(name, "bench")
        if ce is not None:
            xy = gflops(name), ce[ix].mean()
            dot(ax, *xy, INK2, hollow=True)
            label(ax, text, xy, -11, 0, ha="right")
            print(f"  {text}: {xy[0]:.2f} GF, CE {xy[1]:.4f} on the search positions")
    logx(ax, [0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xlim(0.2, 320)
    ax.set_ylim(1.2, 1.32)
    ax.yaxis.set_major_locator(FixedLocator([1.2, 1.24, 1.28, 1.32]))
    ax.set_xlabel("Inference compute per move (GFLOPs, log scale)")
    ax.set_ylabel("Legal-move CE (nats)")
    save(fig, "pareto")
    for k, p in pts.items():
        print(
            f"  {k:16s} {p['gflops']:7.2f} GF  CE {p['ce']:.4f}  top-1 {p['acc']:.2f}"
        )


def rating():
    """Legal-move CE minus Maia-3 79M's per 100-point bin of game rating, on every scored blitz move."""
    rows = [
        r for r in jload(X / "bigrun-progress/acc-by-game-rating-final.json") if r["n"]
    ]
    x = np.array([np.mean([float(v) for v in r["bin"].split("-")]) for r in rows])
    big = "bigrun-143051"
    fig, ax = figure(right=0.76)
    ref = np.array([r["models"]["maia3-79m"]["ce"][0] for r in rows])
    ax.axhline(0, color=MAIA["maia3-79m"], lw=1.4, zorder=2)
    items = [("Maia-3 79M", 0.0)]
    for m in ("maia3-5m", "maia3-23m"):
        v = np.array([r["models"][m]["ce"][0] for r in rows]) - ref
        ax.plot(x, v, color=MAIA[m], lw=1.8, zorder=3)
        items.append((NAME[m], v[-1]))
    for (name, text), ls in zip(PREVIOUS.items(), ((0, (5, 2.5)), (0, (1.5, 2)))):
        ce = previous(name, "rating")
        if ce is not None:
            v = binned(ce) - ref
            ax.plot(x, v, color=INK2, lw=1.6, ls=ls, zorder=3)
            items.append((text, v[-1]))
            print(f"  {text} minus 79M per bin: {np.round(v, 3).tolist()}")
    d = np.array([r["models"][big]["d_ce_79m"] for r in rows])
    ax.fill_between(x, d[:, 1], d[:, 2], color=BLUE, alpha=0.12, lw=0, zorder=2)
    ax.plot(x, d[:, 0], color=BLUE, lw=2.6, zorder=4)
    i = int(np.argmin(np.abs(x - 1950)))
    label(ax, "Allie 2.0", (x[i], d[i, 1]), 0, -8, ha="center", va="top", color=INK)
    lo, hi = (
        min(-0.11, *(y - 0.01 for _, y in items)),
        max(0.13, *(y + 0.012 for _, y in items)),
    )
    ax.set_ylim(lo, hi)
    ends(ax, items, (hi - lo) / 15)
    ax.set_xticks(np.arange(800, 2900, 400))
    ax.set_xlim(600, 2850)
    ax.set_xlabel("Game rating (Lichess blitz)")
    ax.set_ylabel("Δ legal-move CE (nats)")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:+.2f}".translate(MINUS) if v else "0")
    )
    save(fig, "rating")
    worse = [r["bin"] for r, v in zip(rows, d[:, 0]) if v > 0]
    print(
        f"  {len(rows)} bins; CE above 79M in {worse}; interval below 0 in {(d[:, 2] < 0).sum()}"
    )


def binned(values):
    """values (rating-set order) averaged per 100-point game-rating bin, weighted to the natural mover mix."""
    with np.load(DATA / "maia3-bench/rating/games.npz") as z:
        sel, meta = z["sel"][z["keep"]], z["meta"]
    m = jload(DATA / "strat-eval-v1/manifest.json")
    w = (np.array(m["population_moves"]) / np.array(m["scored_moves"]))[sel[:, 2]]
    rating = meta[sel[:, 0], 2:4].mean(1)
    edges = np.arange(600, 2901, 100)
    keep = [(rating >= lo) & (rating < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return np.array([np.average(values[k], weights=w[k]) for k in keep if k.any()])


FIGS = {
    "training": training,
    "isoflop": isoflop,
    "frontier": frontier,
    "moe-vs-dense": moe_vs_dense,
    "optimal": optimal,
    "model": model,
    "pareto": pareto,
    "rating": rating,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        print(name)
        FIGS[name]()
