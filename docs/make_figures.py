"""Regenerate the README figures from the result files.

usage: python docs/make_figures.py [name ...]

Writes docs/figures/NAME.{png,svg} (all figures by default) and prints the numbers each figure shows, so the
README can be checked against this output. The original Allie's scores are read from ALLIE_ORIGINAL (a directory
of its benchmark and rating-set scores); without them its marks are left out.
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

plt.switch_backend("agg")

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
X = R / "recipe10x"
DATA = Path(os.environ.get("ALLIE_DATA", "/data/group_data/dei-group/yimingz3/allie"))
ORIGINAL = Path(
    os.environ.get("ALLIE_ORIGINAL", "/home/yimingz3/allie-equiv/old-allie")
)
OUT = ROOT / "docs" / "figures"
FINAL = (
    R / "pretrain/bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42"
)  # Allie-v3.0
BENCH = (
    X / "distill-v2/report.json"
)  # every scored model on the 80,000 benchmark positions
SEARCH = DATA / "maia3-bench/search"
SWEEP = X / "sweep-readout-c8s200f0v4-w4.json"

INK, INK2, MUTED = "#27272a", "#52525b", "#71717a"
GRID, AXIS, SURFACE = "#e5e7eb", "#d4d4d8", "#ffffff"
BLUE, PALE_BLUE, WASH = "#2563eb", "#93b4f0", "#f3f7fe"
MAIA = {"maia3-5m": "#b4b4bc", "maia3-23m": "#85858d", "maia3-79m": "#52525b"}
NAME = {"maia3-5m": "Maia-3 5M", "maia3-23m": "Maia-3 23M", "maia3-79m": "Maia-3 79M"}
SIZE = (7.2, 4.3)
MINUS = str.maketrans("-", "−")

plt.rcParams.update(
    {
        "font.family": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.size": 0,
        "ytick.major.pad": 6,
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
        **({"left": 0.1, "right": 0.97, "top": 0.95, "bottom": 0.15} | adjust)
    )
    return fig, ax


def label(ax, text, xy, dx=0, dy=0, color=INK2, size=10.5, **kw):
    kw = {"ha": "left", "va": "center", "fontsize": size, "color": color} | kw
    return ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", **kw)


def dot(ax, x, y, color, ms=6.5, hollow=False, z=5):
    face = SURFACE if hollow else color
    edge = color if hollow else SURFACE
    ax.plot(x, y, "o", ms=ms, mfc=face, mec=edge, mew=1.4 if ms >= 6 else 0, zorder=z)


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


def law(fam):
    """The sweep's L(N, D) = E + A (N/1e7)^-alpha + B (D/1e8)^-beta for the main eval."""
    m = next(m for m in jload(SWEEP)["metrics"] if m["metric"] == "macro")
    t = m["law"][fam]["theta"]
    E, A, B = math.exp(t[0]), math.exp(t[1]), math.exp(t[3])
    return lambda n, d: E + A * (n / 1e7) ** -t[2] + B * (d / 1e8) ** -t[4]


def original(name):
    """The original Allie's raw policy on a position set (bench: the 80,000 benchmark positions, rating: every
    scored blitz move), as per-position (legal-move CE, top-1) in that set's order; None if not scored."""
    f = ORIGINAL / name / "allie1-medium.npz"
    if not f.exists():
        print(f"  (no original Allie scores at {f})")
        return None
    with np.load(f) as z:
        return np.asarray(z["ce_legal"], float), np.asarray(z["top1"], float)


def original_gflops():
    """The original Allie's GFLOPs per move with a key-value cache, counted as for the other models."""
    return jload(ORIGINAL / "flops-medium.json")["flops_per_move_incremental"] / 1e9


# ------------------------------------------------------------------------------------- shared pieces


ORIG = "Original Allie"


def ends(ax, items, gap):
    """Direct labels right of the plot for (text, y, color) series ending at y: stacked at least gap apart, with
    a short leader to the series' own end when a label had to move."""
    items = sorted(items, key=lambda t: t[1])
    ys = [items[0][1]]
    for _, y, _ in items[1:]:
        ys.append(max(y, ys[-1] + gap))
    for (text, y0, _), y in zip(items, ys):
        leader = {
            "arrowstyle": "-",
            "color": AXIS,
            "lw": 0.7,
            "shrinkA": 1,
            "shrinkB": 0,
        }
        ax.annotate(
            text, (1.0, y0), xytext=(1.03, y), xycoords=("axes fraction", "data"),
            textcoords=("axes fraction", "data"), fontsize=10.5, color=INK2, va="center", ha="left",
            arrowprops=leader if abs(y - y0) > gap / 4 else None,
        )  # fmt: skip


# --------------------------------------------------------------- 1. quality against inference compute


def pareto():
    """Legal-move CE against GFLOPs per move, on the 20,000 benchmark positions that search ran on."""
    rep = jload(SEARCH / "report-bigrun.json")
    pts = rep["points"]
    fig, ax = figure(right=0.95)
    ce = [pts[f"maia/{m}"]["ce"] for m in MAIA]
    for m in MAIA:
        xy = pts[f"maia/{m}"]["gflops"], pts[f"maia/{m}"]["ce"]
        dot(ax, *xy, MAIA[m])
        label(ax, NAME[m], xy, 9, 0)
    line = [pts[k] for k in ("frozen/legal", "devcal/5", "devcal/128")]
    gx, gy = [p["gflops"] for p in line], [p["ce"] for p in line]
    ax.plot(gx, gy, color=BLUE, lw=1.4, zorder=3)
    for x, y, n in zip(gx[1:], gy[1:], ("5", "128")):
        dot(ax, x, y, BLUE, ms=5)
        label(
            ax, f"{n} sims", (x, y), 0, -9, ha="center", va="top", color=MUTED, size=9.5
        )
    dot(ax, gx[0], gy[0], BLUE, ms=8, z=6)
    label(
        ax, "Allie-v3.0 (raw)", (gx[0], gy[0]), 0, -11, ha="center", va="top", color=INK
    )
    label(
        ax,
        "Allie-v3.0 + search",
        (gx[2], gy[2]),
        0,
        10,
        ha="center",
        va="bottom",
        color=INK,
    )
    orig = original("bench")
    if orig:
        xy = original_gflops(), orig[0][search_index(rep["n"])].mean()
        dot(ax, *xy, MAIA["maia3-23m"], hollow=True)
        label(ax, ORIG, xy, 9, 0)
        ce.append(xy[1])
        print(
            f"  original Allie {xy[0]:.2f} GF  CE {xy[1]:.4f} on the search positions"
        )
    logx(ax, [0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xlim(0.4, 300)
    top = math.ceil((max(ce) + 0.008) * 50) / 50
    ax.set_ylim(1.2, max(1.32, top))
    ax.yaxis.set_major_locator(
        FixedLocator(np.arange(1.2, max(1.32, top) + 1e-9, 0.02))
    )
    ax.set_xlabel("Inference compute per move (GFLOPs, log scale)")
    ax.set_ylabel("Legal-move cross-entropy (nats)")
    save(fig, "pareto")
    for k, p in pts.items():
        print(
            f"  {k:16s} {p['gflops']:7.2f} GF  CE {p['ce']:.4f}  top-1 {p['acc']:.2f}"
        )


def search_index(n):
    """Benchmark positions the search ran on (legal.npz order), as run.py drew them."""
    files = sorted((SEARCH / "bigrun/legal").glob("[0-9]*.npz"))
    return np.concatenate([np.load(f)["index"] for f in files])[:n]


# ------------------------------------------------------------------------- 2. across the rating range


def rating():
    """Legal-move CE minus Maia-3 79M's per 100-point bin of game rating, on every scored blitz move."""
    rows = [
        r for r in jload(X / "bigrun-progress/acc-by-game-rating-final.json") if r["n"]
    ]
    x = np.array([np.mean([float(v) for v in r["bin"].split("-")]) for r in rows])
    big = "bigrun-143051"
    fig, ax = figure(right=0.8)
    ref = np.array([r["models"]["maia3-79m"]["ce"][0] for r in rows])
    ax.axhline(0, color=MAIA["maia3-79m"], lw=1, zorder=2)
    items = [("Maia-3 79M", 0.0, MAIA["maia3-79m"])]
    for m in ("maia3-5m", "maia3-23m"):
        v = np.array([r["models"][m]["ce"][0] for r in rows]) - ref
        ax.plot(x, v, color=MAIA[m], lw=1.1, zorder=3)
        items.append((NAME[m], v[-1], MAIA[m]))
    orig = original("rating")
    if orig:
        v = binned(orig[0]) - ref
        ax.plot(x, v, color=MAIA["maia3-23m"], lw=1.1, ls=(0, (4, 2)), zorder=3)
        items.append((ORIG, v[-1], MAIA["maia3-23m"]))
    d = np.array([r["models"][big]["d_ce_79m"] for r in rows])
    ax.fill_between(x, d[:, 1], d[:, 2], color=BLUE, alpha=0.12, lw=0, zorder=2)
    ax.plot(x, d[:, 0], color=BLUE, lw=1.8, zorder=4)
    i = int(np.argmin(np.abs(x - 1950)))
    label(ax, "Allie-v3.0", (x[i], d[i, 1]), 0, -6, ha="center", va="top", color=INK)
    lo, hi = (
        min(-0.11, *(y - 0.01 for _, y, _ in items)),
        max(0.13, *(y + 0.01 for _, y, _ in items)),
    )
    ax.set_ylim(lo, hi)
    ends(ax, items, (hi - lo) / 22)
    ax.set_xticks(np.arange(800, 2900, 400))
    ax.set_xlim(600, 2850)
    ax.set_xlabel("Game rating (Lichess blitz, mean of both players)")
    ax.set_ylabel("Δ legal-move cross-entropy (nats)")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:+.2f}".translate(MINUS) if v else "0")
    )
    save(fig, "rating")
    worse = [r["bin"] for r, v in zip(rows, d[:, 0]) if v > 0]
    print(
        f"  {len(rows)} bins; CE above 79M in {worse}; interval below 0 in {(d[:, 2] < 0).sum()}"
    )
    if orig:
        print(f"  original Allie minus 79M per bin: {np.round(v, 3).tolist()}")


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


# ------------------------------------------------------------------------------------ 3. the model


def box(ax, x, y, w, h, text="", ec=AXIS, fc=SURFACE, size=10, color=INK, lw=1):
    style = "round,pad=0,rounding_size=0.06"
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
    style = {"arrowstyle": "-|>", "mutation_scale": 9, "color": color, "lw": 1}
    ax.add_patch(FancyArrowPatch(a, b, shrinkA=0, shrinkB=0, zorder=4, **style))


def model():
    """Inputs, one mixture-of-experts block (schematic), the next-move output."""
    fig = plt.figure(figsize=(7.2, 3.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 7.2), ylim=(0, 3.2))
    ax.axis("off")
    mid = 1.5
    for y, text in ((2.3, "Game so far + ratings"), (1.5, "Clock"), (0.7, "Board")):
        box(ax, 0.1, y - 0.24, 1.5, 0.48, text, size=9.5)
        arrow(ax, (1.6, y), (1.95, mid + (y - mid) * 0.3))
    box(ax, 1.95, 0.25, 3.45, 2.65, ec="#c9d8f5", fc=WASH)
    ax.text(
        3.675,
        2.66,
        "MoE block (23 of the 24 blocks)",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK2,
    )
    box(ax, 2.1, mid - 0.25, 0.85, 0.5, "Attention", size=9.5)
    # the shared expert: always on, beside the router
    ax.plot([2.95, 3.15, 3.15, 4.0], [mid, mid, 2.2, 2.2], color=BLUE, lw=1, zorder=3)
    box(ax, 4.0, 2.2 - 0.12, 0.8, 0.24, ec=BLUE, fc="#dbe6fb", lw=1)
    ax.text(
        4.4, 2.38, "shared expert", ha="center", va="bottom", fontsize=9.5, color=INK2
    )
    ax.plot([4.8, 5.1, 5.1], [2.2, 2.2, mid + 0.07], color=BLUE, lw=1, zorder=3)
    # the router and its routed experts
    ax.plot([2.95, 3.45], [mid, mid], color=MUTED, lw=1, zorder=3)
    ax.add_patch(plt.Circle((3.5, mid), 0.06, fc=MUTED, lw=0, zorder=4))
    ax.text(3.42, mid - 0.12, "router", ha="right", va="top", fontsize=9, color=MUTED)
    rows = [False, True, None, False, True, False]
    ys = np.linspace(1.75, 0.65, len(rows))
    for on, y in zip(rows, ys):
        if on is None:
            ax.text(
                4.4,
                y,
                "⋮",
                ha="center",
                va="center",
                fontsize=11,
                color=MUTED,
                zorder=4,
            )
            continue
        ec, fc = (BLUE, "#dbe6fb") if on else (AXIS, SURFACE)
        box(ax, 4.0, y - 0.08, 0.8, 0.16, ec=ec, fc=fc, lw=1)
        color = BLUE if on else "#e4e4e7"
        ax.plot([3.5, 4.0], [mid, y], color=color, lw=1, zorder=3)
        ax.plot([4.8, 5.03], [y, mid], color=color, lw=1, zorder=3)
    ax.text(
        4.4,
        0.42,
        "16 of 256 routed experts",
        ha="center",
        va="center",
        fontsize=9,
        color=INK2,
    )
    ax.add_patch(plt.Circle((5.1, mid), 0.07, fc=SURFACE, ec=MUTED, lw=1, zorder=4))
    ax.text(5.1, mid, "+", ha="center", va="center", fontsize=8, color=MUTED, zorder=5)
    arrow(ax, (5.17, mid), (5.7, mid), color=MUTED)
    box(
        ax,
        5.7,
        mid - 0.32,
        1.4,
        0.64,
        "Next-move\ndistribution",
        ec=BLUE,
        lw=1.4,
        size=9.5,
    )
    ax.text(
        6.4,
        mid - 0.45,
        "also: move time, result",
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "model")


# -------------------------------------------------------------------------------------- 4. training


def training():
    """Allie-v3.0's benchmark CE over training, against Maia-3, the original Allie and the scaling-law forecast."""
    rows = sorted(
        jsonl(X / "maia3-bench/bigrun-trajectory-v2.jsonl"), key=lambda r: r["step"]
    )
    t = np.array([r["tokens"] for r in rows]) / 1e9
    bench = np.array([r["ce"] for r in rows])
    main = np.array([r["golden_macro"] for r in rows])
    act, _, D = allie_n()
    forecast = law("s16")(act, D)
    # the law forecasts main-eval CE; mapped to benchmark CE through this run's own checkpoints (post hoc)
    fit = np.polyfit(main, bench, 1)
    rep = jload(BENCH)
    fig, ax = figure(right=0.69)
    refs = [(NAME[m], rep[m]["ce"], MAIA[m], "-") for m in MAIA]
    refs.append(
        ("Forecast (post-hoc mapping)", np.polyval(fit, forecast), MUTED, (0, (4, 3)))
    )
    orig = original("bench")
    if orig:
        refs.append((ORIG, orig[0].mean(), MAIA["maia3-23m"], (0, (1.5, 2))))
    for _, y, color, ls in refs:
        ax.axhline(y, color=color, lw=1.1, ls=ls, zorder=2)
    top = max(1.37, *(y + 0.012 for _, y, _, _ in refs))
    ax.set_ylim(1.19, top)
    ends(ax, [(text, y, color) for text, y, color, _ in refs], (top - 1.19) / 26)
    ax.plot(t, bench, color=BLUE, lw=1.8, zorder=4)
    dot(ax, t[-1], bench[-1], BLUE, ms=6)
    label(ax, "Allie-v3.0", (t[-1], bench[-1]), -8, -2, color=INK, ha="right", va="top")
    ax.set_xlim(20, 77)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Legal-move cross-entropy (nats)")
    save(fig, "training")
    resid = bench - np.polyval(fit, main)
    print(
        f"  forecast {forecast:.4f} main eval -> {np.polyval(fit, forecast):.4f} benchmark (sd {resid.std():.4f})"
    )
    print(
        f"  final {bench[-1]:.4f}; below 79M from {t[np.argmax(bench < rep['maia3-79m']['ce'])]:.1f}B tokens"
    )


# ------------------------------------------------------------------------------------ 5. scaling sweep


def scaling():
    """Isoflop curves, MoE against dense, one panel per budget: quadratic fits in log N through each budget's
    selected four-size window (as the sweep readout fits them, every run), with the window's seed means."""
    sweep = jload(SWEEP)
    cells = next(m for m in sweep["metrics"] if m["metric"] == "macro")["cells"]
    names = {
        "1e17": r"$6.3\times10^{17}$ FLOPs",
        "3e17": r"$1.9\times10^{18}$ FLOPs",
        "1e18": r"$6.2\times10^{18}$ FLOPs",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2, wspace=0.08)
    for ax, (b, flops) in zip(axes, names.items()):
        for fam, color in (("dense", MAIA["maia3-23m"]), ("s16", BLUE)):
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
            ax.plot(np.exp(xs), np.polyval(c, xs), color=color, lw=1.5, zorder=3)
            means = np.array([(m, y[n == m].mean()) for m in np.unique(n)])
            ax.plot(*means.T, "o", ms=4, color=color, mec=SURFACE, mew=0, zorder=4)
            v = -c[1] / (2 * c[0])
            print(
                f"  {b} {fam:5s} N* {math.exp(v) / 1e6:6.1f}M L* {np.polyval(c, v):.4f} (readout {cell['n_opt'] / 1e6:.1f}M {cell['l']:.4f})"
            )
            if b == "1e17":
                if fam == "s16":
                    label(
                        ax,
                        "MoE",
                        (math.exp(v), np.polyval(c, v)),
                        0,
                        -12,
                        ha="center",
                        color=color,
                    )
                else:
                    start = np.exp(xs[0]), np.polyval(c, xs[0])
                    label(ax, "dense", start, -4, 0, ha="right", color=color)
        ax.set_title(flops, fontsize=10, color=INK2, loc="center")
        logx(ax, [3e7, 1e8, 3e8], lambda v: f"{v / 1e6:g}M")
        ax.set_xlim(1.2e7, 5e8)
    axes[0].set_ylim(1.3, 1.45)
    axes[0].set_ylabel("Main-evaluation CE (nats)")
    axes[1].set_xlabel("Active parameters (log scale)")
    save(fig, "scaling")


FIGS = {
    "pareto": pareto,
    "rating": rating,
    "model": model,
    "training": training,
    "scaling": scaling,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        print(name)
        FIGS[name]()
