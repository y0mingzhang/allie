"""acc-by-game-rating-final.png / .md: top-1 move-matching accuracy and legal-move CE against game rating
(mean of both players' ratings, 100-point bins), as on Maia-3's homepage, for Maia-3 3M / 5M / 23M / 79M,
the best 1e18 sweep MoE and Allie 2.0, on every scored blitz move of the golden eval
(maia3-bench/rating: 402,108 positions in 6,247 games, eval.maia3.rating_sample).

The golden eval samples each mover-Elo band at its own rate, so every position is weighted by its
cell's population / sampled moves (strat-eval-v1 manifest): within a bin the mix of movers is then
the natural July 2026 one. Intervals: 95%, 2,000 bootstrap draws of whole games.

usage: plot_rating.py BIGRUN_SCORES_NAME   (e.g. bigrun-143051, a file in rating/scores/)
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from allie import paths  # noqa: E402

DATA = paths.DATA / "maia3-bench"
G = paths.DATA / "strat-eval-v1"
HERE = paths.ROOT / "results/recipe10x/bigrun-progress"  # outputs
EDGES = np.arange(600, 2901, 100)
REPS, MIN_N = 2000, 3000
# name -> (label, color, line width)
MODELS = {
    "maia3-3m": ("Maia-3 3M (ablation)", "#e87ba4", 1.5),
    "maia3-5m": ("Maia-3 5M", "#f2d27a", 2),
    "maia3-23m": ("Maia-3 23M", "#9bd59b", 2),
    "maia3-79m": ("Maia-3 79M", "#2a78d6", 2),
    "sw-moe-s42": ("our best 1e18 MoE", "#a9a8a3", 2),
}


def scores(name):
    with np.load(DATA / "rating" / "scores" / f"{name}.npz") as z:
        ce = z["ce_legal"] if "ce_legal" in z else z["ce"]
        return np.asarray(ce, float), np.asarray(z["top1"], float)


def boot(games, w, xs, rng):
    """Bootstrap draws of the weighted mean of each x in xs (games resampled whole)."""
    _, g = np.unique(games, return_inverse=True)
    sw = np.bincount(g, w)
    sx = [np.bincount(g, w * x) for x in xs]
    pick = rng.integers(0, len(sw), (REPS, len(sw)))
    den = sw[pick].sum(1)
    return [s[pick].sum(1) / den for s in sx]


def main():
    big = sys.argv[1]
    rng = np.random.default_rng(7)
    with np.load(DATA / "rating" / "games.npz") as z:
        sel, meta = z["sel"][z["keep"]], z["meta"]
    m = json.loads((G / "manifest.json").read_text())
    w = (np.array(m["population_moves"]) / np.array(m["scored_moves"]))[sel[:, 2]]
    game = sel[:, 0]
    rating = meta[game, 2:4].mean(1)
    info = json.loads((DATA / "rating" / "scores" / f"{big}.json").read_text())
    final = info["step"] == 143051
    tag = "final" if final else f"step {info['step']:,}"
    models = dict(MODELS) | {big: (f"our big run ({tag})", "#0b0b0b", 3)}
    sc = {k: scores(k) for k in models}
    rows, centers = [], (EDGES[:-1] + EDGES[1:]) / 2
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = (rating >= lo) & (rating < hi)
        r = dict(
            bin=f"{lo}-{hi}", n=int(k.sum()), games=len(np.unique(game[k])), models={}
        )
        for name, (ce, top1) in sc.items():
            dce, dacc = (
                ce[k] - sc["maia3-79m"][0][k],
                top1[k] - sc["maia3-79m"][1][k],
            )
            b = boot(game[k], w[k], [top1[k], ce[k], dacc, dce], rng)
            est = [np.average(x, weights=w[k]) for x in (top1[k], ce[k], dacc, dce)]
            r["models"][name] = {
                key: [float(e), *map(float, np.percentile(d, [2.5, 97.5]))]
                for key, e, d in zip(("acc", "ce", "d_acc_79m", "d_ce_79m"), est, b)
            }
        rows.append(r)

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150, facecolor=surface)
    thin = np.array([r["n"] < MIN_N for r in rows])
    for ax, key, scale, ylabel in (
        (axes[0], "acc", 100, "Move-matching accuracy (%)"),
        (axes[1], "ce", 1, "Cross-entropy over legal moves (nats)"),
    ):
        ax.set_facecolor(surface)
        for name, (label, color, lw) in models.items():
            v = np.array([r["models"][name][key] for r in rows]) * scale
            ax.plot(
                centers,
                v[:, 0],
                color=color,
                lw=lw,
                label=label,
                zorder=3 + (name == big),
            )
            if name in (big, "maia3-79m"):
                ax.fill_between(
                    centers, v[:, 1], v[:, 2], color=color, alpha=0.12, lw=0, zorder=2
                )
            if name == big:
                ax.plot(centers[thin], v[thin, 0], ls="none", marker="o", ms=7, mfc=surface, mec=color, mew=1.5,
                        zorder=5, label=f"bin with < {MIN_N:,} positions")  # fmt: skip
        ax.set_xticks(np.arange(600, 2901, 200))
        ax.set_xlim(600, 2900)
        ax.set_xlabel(
            "Game rating (mean of both players)", color=ink, fontweight="bold"
        )
        ax.set_ylabel(ylabel, color=ink, fontweight="bold")
        ax.grid(True, color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=muted, loc="lower right")
    fig.suptitle(
        "Lichess blitz, July 2026 (held out): all 402K scored moves of our golden eval, reweighted to the natural "
        "mover mix. Bands: 95% game-bootstrap intervals (big run, Maia-3 79M)",
        color=muted, fontsize=9.5, x=0.01, ha="left",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "acc-by-game-rating-final.png", facecolor=surface)

    f = lambda t, s=100, fmt="{:+.2f}": (
        f"{fmt.format(t[0] * s)} [{fmt.format(t[1] * s)}, {fmt.format(t[2] * s)}]"
    )
    names = list(models)
    lines = [
        f"Top-1 accuracy % by game rating (big run = {tag}, model sha256 {info['model_sha256'][:12]}); weighted to the natural mover mix. "
        f"Bins with < {MIN_N:,} positions are marked *.",
        "",
        "| game rating | positions | games | " + " | ".join(models[k][0] for k in names)
        + " | big run - 79M acc (pp) | big run - 79M CE |",
        "|---|---:|---:|" + "---:|" * (len(names) + 2),
    ]  # fmt: skip
    for r in rows:
        if not r["n"]:
            continue
        mm = r["models"]
        lines.append(
            f"| {r['bin']}{' *' if r['n'] < MIN_N else ''} | {r['n']:,} | {r['games']:,} | "
            + " | ".join(f"{mm[k]['acc'][0] * 100:.1f}" for k in names)
            + f" | {f(mm[big]['d_acc_79m'])} | {f(mm[big]['d_ce_79m'], 1, '{:+.4f}')} |"
        )
    (HERE / "acc-by-game-rating-final.md").write_text("\n".join(lines) + "\n")
    (HERE / "acc-by-game-rating-final.json").write_text(
        json.dumps(rows, indent=1) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
