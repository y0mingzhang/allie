"""Plots for the IsoFLOP v1 study: per-budget curves and the compute frontier."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isoflop_fit as f

OUT = f.STUDY / "plots"
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
COLOR = dict(ours="#2a78d6", qwen="#eb6834")
NAME = dict(ours="Ours", qwen="Qwen (chess-v2 recipe)")
METRIC = dict(move="Move CE", expert2400="≥2400 move CE")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
millions = FuncFormatter(lambda v, _: f"{v / 1e9:g}B" if v >= 1e9 else f"{v / 1e6:g}M")


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, which="major", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, which="both")
    for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
        label.set_color(INK2)


def dots(ax, x, y, color, **kw):
    ax.scatter(x, y, s=64, c=color, edgecolors=SURFACE, linewidths=2, zorder=3, **kw)


def curves():
    data = {m: f.load(m) for m in METRIC}
    budgets = sorted(set(next(iter(data.values()))["ours"][:, 2]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), facecolor=SURFACE)
    for row, metric in enumerate(METRIC):
        iso = f.isoflop(data[metric])
        for col, c in enumerate(budgets):
            ax = axes[row, col]
            style(ax)
            for r in f.RECIPES:
                s = data[metric][r][data[metric][r][:, 2] == c]
                x = np.log(s[:, 0])
                a, b, k = np.polyfit(x, s[:, 3], 2)
                xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 100)
                ax.plot(
                    np.exp(xs),
                    a * xs**2 + b * xs + k,
                    color=COLOR[r],
                    lw=2,
                    solid_capstyle="round",
                )
                dots(ax, s[:, 0], s[:, 3], COLOR[r], label=NAME[r])
                m = next(m for m in iso[r]["minima"] if m["budget"] == c)
                if m["interior"]:
                    ax.scatter(
                        [m["n_opt"]],
                        [m["loss_opt"]],
                        s=70,
                        facecolors=SURFACE,
                        edgecolors=COLOR[r],
                        linewidths=2,
                        zorder=4,
                    )
                    ax.annotate(
                        f"{m['n_opt'] / 1e6:.0f}M · {m['loss_opt']:.3f}",
                        (m["n_opt"], m["loss_opt"]),
                        textcoords="offset points",
                        xytext=(0, -16 if r == "ours" else 9),
                        ha="center",
                        fontsize=8,
                        color=INK2,
                    )
            ax.set_xscale("log")
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.12 * (hi - lo), hi)
            ax.xaxis.set_major_formatter(millions)
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            if row == 0:
                ax.set_title(f"ND = {c:.0e}".replace("+", ""), fontsize=11)
            if row == 1:
                ax.set_xlabel("Non-embedding parameters")
            if col == 0:
                ax.set_ylabel(METRIC[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", frameon=False, labelcolor=INK2, ncol=2
    )
    fig.suptitle(
        "IsoFLOP curves: loss vs model size at fixed compute (hollow = fitted optimum)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "isoflop-curves.png", dpi=200, facecolor=SURFACE)


def opt_n(p, c):
    g = lambda x: f.predict(p, np.exp(x), c / np.exp(x))
    return float(
        np.exp(
            minimize_scalar(
                g, bounds=(np.log(1e4), np.log(c / 1e4)), method="bounded"
            ).x
        )
    )


def model_free_multiplier(iso, c):
    ours = next(m["loss_opt"] for m in iso["ours"]["minima"] if m["budget"] == c)
    q = [(m["budget"], m["loss_opt"]) for m in iso["qwen"]["minima"]]
    for (c0, l0), (c1, l1) in zip(q, q[1:]):
        if l1 <= ours <= l0:
            return float(
                np.exp(np.log(c0) + (l0 - ours) / (l0 - l1) * np.log(c1 / c0)) / c
            )
    return None


def frontier(metric="move", boot=200):
    d = f.load(metric)
    iso = f.isoflop(d)
    shared, separate = f.additive(d, True), f.additive(d, False)
    ps = f.unpack(shared, True)
    rng = np.random.default_rng(0)
    samples = [
        f.additive(
            {r: x[rng.integers(len(x), size=len(x))] for r, x in d.items()},
            True,
            starts=[shared],
        )
        for _ in range(boot)
    ]
    grid = np.logspace(16.4, 19, 12)
    measured = (3e16, 3e17)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE)

    ax = axes[0]
    style(ax)
    ax.axvspan(*measured, color=GRID, alpha=0.35, lw=0)
    for r in f.RECIPES:
        ax.plot(
            grid,
            [f.best_loss(ps[r], c) for c in grid],
            color=COLOR[r],
            lw=2,
            label=f"{NAME[r]}, fitted law",
        )
        mins = [m for m in iso[r]["minima"] if m["interior"]]
        dots(ax, [m["budget"] for m in mins], [m["loss_opt"] for m in mins], COLOR[r])
    ax.set_title("Best achievable loss")
    ax.set_ylabel(METRIC[metric])
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)

    ax = axes[1]
    style(ax)
    ax.axvspan(*measured, color=GRID, alpha=0.35, lw=0)
    cm = np.array(
        [[f.multiplier(t, True, c) or np.nan for c in grid] for t in samples], float
    )
    lo, hi = np.nanpercentile(cm, [5, 95], axis=0)
    ax.fill_between(
        grid, lo, hi, color=INK2, alpha=0.10, lw=0, label="90% bootstrap band"
    )
    ax.plot(
        grid,
        [f.multiplier(shared, True, c) for c in grid],
        color=INK2,
        lw=2,
        label="Fitted law, shared floor",
    )
    ax.plot(
        grid,
        [f.multiplier(separate, False, c) or np.nan for c in grid],
        color=MUTED,
        lw=2,
        ls="--",
        label="Fitted law, separate floors",
    )
    free = [
        (c, model_free_multiplier(iso, c))
        for c in sorted({m["budget"] for m in iso["ours"]["minima"]})
    ]
    free = [(c, v) for c, v in free if v]
    dots(
        ax,
        [c for c, _ in free],
        [v for _, v in free],
        INK,
        label="Measured optima (no law)",
    )
    for c, v in free:
        ax.annotate(
            f"{v:.2f}×",
            (c, v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=INK2,
        )
    ax.axhline(1, color=AXIS, lw=1)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 3])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_title("Qwen compute needed to match ours")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower left")

    ax = axes[2]
    style(ax)
    ax.axvspan(*measured, color=GRID, alpha=0.35, lw=0)
    for r in f.RECIPES:
        ax.plot(
            grid,
            [opt_n(ps[r], c) for c in grid],
            color=COLOR[r],
            lw=2,
            label=f"{NAME[r]}, fitted law",
        )
        mins = [m for m in iso[r]["minima"] if m["interior"]]
        dots(
            ax,
            [m["budget"] for m in mins],
            [m["n_opt"] for m in mins],
            COLOR[r],
            label=f"{NAME[r]}, measured optimum",
        )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(millions)
    ax.set_title("Compute-optimal model size")
    ax.set_ylabel("Non-embedding parameters")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Compute (N·D, non-embedding)")
    fig.suptitle(
        f"Compute frontier ({METRIC[metric]}); grey band = measured range",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"frontier-{metric}.png", dpi=200, facecolor=SURFACE)


NODE_DAY = 1.46e19  # 8 L40S x 24h at 35% MFU, 362 TFLOP/s per GPU, 6ND
HISTORICAL = dict(n=1.409e9, d=44.4e9, move=1.3422, expert2400=1.1622)


def forecast(metric="move", boot=200):
    d = f.load(metric)
    iso = f.isoflop(d)
    shared = f.additive(d, True)
    ps = f.unpack(shared, True)
    rng = np.random.default_rng(0)
    samples = [
        f.unpack(
            f.additive(
                {r: x[rng.integers(len(x), size=len(x))] for r, x in d.items()},
                True,
                starts=[shared],
            ),
            True,
        )
        for _ in range(boot)
    ]
    grid = np.logspace(16.4, 20, 18)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    style(ax)
    ax.axvspan(3e16, 3e17, color=GRID, alpha=0.35, lw=0)
    for r in f.RECIPES:
        band = np.array([[f.best_loss(p[r], c) for c in grid] for p in samples])
        lo, hi = np.percentile(band, [5, 95], axis=0)
        ax.fill_between(grid, lo, hi, color=COLOR[r], alpha=0.10, lw=0)
        ax.plot(
            grid,
            [f.best_loss(ps[r], c) for c in grid],
            color=COLOR[r],
            lw=2,
            label=f"{NAME[r]}, compute-optimal forecast",
        )
        mins = [m for m in iso[r]["minima"] if m["interior"]]
        dots(ax, [m["budget"] for m in mins], [m["loss_opt"] for m in mins], COLOR[r])
    hx = HISTORICAL["n"] * HISTORICAL["d"]
    ax.scatter(
        [hx],
        [HISTORICAL[metric]],
        s=70,
        facecolors=SURFACE,
        edgecolors=COLOR["qwen"],
        linewidths=2,
        zorder=4,
        label="Historical 1.4B Qwen run (not compute-optimal)",
    )
    for k, label in ((1, "1 node-day"), (2, "2 node-days")):
        ax.axvline(k * NODE_DAY, color=AXIS, lw=1)
        ax.annotate(
            label,
            (k * NODE_DAY, ax.get_ylim()[0]),
            textcoords="offset points",
            xytext=(-12, 8),
            rotation=90,
            fontsize=8,
            color=INK2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Compute (N·D, non-embedding)")
    ax.set_ylabel(METRIC[metric])
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower left")
    fig.suptitle(
        f"Forecast: best achievable {METRIC[metric]} vs compute (shared-floor law, 90% bootstrap band)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"forecast-{metric}.png", dpi=200, facecolor=SURFACE)

    exp = np.polyfit(
        np.log([m["budget"] for m in iso["ours"]["minima"]]),
        np.log([m["n_opt"] for m in iso["ours"]["minima"]]),
        1,
    )
    for k in (1, 2):
        c = k * NODE_DAY
        band = np.array([f.best_loss(p["ours"], c) for p in samples])
        n_law, n_meas = opt_n(ps["ours"], c), float(np.exp(np.polyval(exp, np.log(c))))
        print(
            f"{metric} {k} node-day ND={c:.2e}: ours L*={f.best_loss(ps['ours'], c):.4f} [{np.percentile(band, 5):.4f}, {np.percentile(band, 95):.4f}]"
            f" qwen L*={f.best_loss(ps['qwen'], c):.4f}; N* law {n_law / 1e6:.0f}M (D {c / n_law / 1e9:.0f}B) vs measured-allocation"
            f" {n_meas / 1e6:.0f}M (D {c / n_meas / 1e9:.0f}B, law loss there {float(f.predict(ps['ours'], n_meas, c / n_meas)):.4f})"
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["forecast"]:
        OUT.mkdir(exist_ok=True)
        forecast("move")
        forecast("expert2400")
        sys.exit()
    OUT.mkdir(exist_ok=True)
    curves()
    frontier("move")
    frontier("expert2400")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
