"""Pareto tables and plot: our raw and searched points against Maia-3 on the same positions.

usage: report.py <tag>   (a folder under maia3-bench/search/ written by run.py, plus fit.py's
devcal-*.npz when present). Positions are the prefix of run.py's stratified order that every
point finished, so each band has the same n and pooled = macro; Maia-3 and the raw scorer are the
cached 80K arrays at those indices.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate import boot  # noqa: E402

DATA = Path("/data/group_data/dei-group/yimingz3/allie/maia3-bench")
HERE = Path(__file__).resolve().parent
FLOPS = json.loads((HERE.parent / "aggregate-sweep1e18.json").read_text())["flops"]
BUDGETS = ["5", "8", "25", "128", "460", "adaptive"]
BANDS = {4: "<1400", 5: "1400-2000", 6: "2000-2400", 7: ">=2400"}
MAIA = {"maia3-5m": "Maia-3 5M", "maia3-23m": "Maia-3 23M", "maia3-79m": "Maia-3 79M"}
MODEL = {
    "sw-dense-s42": ("sweep dense 1e18", 18, 896),
    "sw-moe-s42": ("sweep MoE 1e18", 18, 896),
    "bigrun132k": ("big run step 132K", 24, 1536),
    "bigrun": ("big run", 24, 1536),
}
# the big run (MoE 0.69B active / 5.6B total): 2 x active matmul parameters (report_sweep1e18.sweep_flops with
# its source's extra_flops); its raw reference is the Maia-3 watcher's score_moe.py output of the same checkpoint
FLOPS |= {t: dict(flops_per_move=1389084288) for t in ("bigrun132k", "bigrun")}
RAW = {
    "bigrun132k": DATA / "bigrun-v2/step-00132096/scores.npz",
    "bigrun": DATA / "bigrun-v2/step-00143051/scores.npz",
}
KEYS = ("index", "ce", "top1", "nodes")


def cached(name):
    z = np.load(RAW.get(name, DATA / "scores" / f"{name}.npz"))
    return (z["ce_legal"] if "ce_legal" in z else z["ce"]), z["top1"]


def load(tag):
    """(group, budget) -> per-position arrays; group frozen (run.py) or devcal (fit.py)."""
    runs = {}
    for b in ["legal"] + BUDGETS:
        files = sorted((DATA / "search" / tag / b).glob("[0-9]*[0-9].npz"))
        if files:
            z = [np.load(f) for f in files]
            runs["frozen", b] = {
                k: np.concatenate([x[k] for x in z]) for k in KEYS + ("prefill",)
            }
            runs["frozen", b]["seconds"] = np.concatenate(
                [np.full(len(x["ce"]), x["seconds"] / len(x["ce"])) for x in z]
            )
        f = DATA / "search" / tag / f"devcal-{'temperature' if b == 'legal' else b}.npz"
        if f.exists():
            runs["devcal", b] = dict(np.load(f))
    n = min(len(r["index"]) for r in runs.values())
    ix = runs["frozen", "legal"]["index"][:n]
    for r in runs.values():
        assert np.array_equal(r["index"][:n], ix)
    return {k: {kk: v[:n] for kk, v in r.items()} for k, r in runs.items()}, ix


def ci(games, x, rng):
    lo, hi = np.percentile(boot(games, x, rng), [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def points(tag, runs, ix):
    label, layers, width = MODEL[tag]
    per_eval = FLOPS[tag]["flops_per_move"] / 1e9
    attn = (
        4 * layers * width * (runs["frozen", "legal"]["prefill"] + 1).mean() / 1e9
    )  # QK + AV, cached token
    pts = {
        ("maia", m): dict(
            label=l,
            gflops=FLOPS[m]["flops_per_move"] / 1e9,
            nodes=0.0,
            ce=cached(m)[0][ix],
            top1=cached(m)[1][ix].astype(float),
        )
        for m, l in MAIA.items()
    }
    for (group, b), r in runs.items():
        name = {"legal": "raw", "adaptive": "search Elo-adaptive"}.get(b, f"search {b}")
        if group == "devcal":
            name = (
                "raw + dev temperature" if b == "legal" else name + ", dev-calibrated"
            )
        pts[group, b] = dict(
            label=f"Ours {label}, {name}",
            nodes=float(r["nodes"].mean()),
            gflops=(1 + r["nodes"].mean()) * (per_eval + attn),
            ce=r["ce"],
            top1=r["top1"].astype(float),
            ms=1e3 * runs["frozen", b]["seconds"].mean()
            if ("frozen", b) in runs
            else None,
        )
    return pts, attn


def table(tag):
    rng = np.random.default_rng(7)
    with np.load(DATA / "games.npz") as z:
        s = z["sel"][z["keep"]]
    runs, ix = load(tag)
    game, cell = s[ix, 0], s[ix, 2]
    pts, attn = points(tag, runs, ix)
    raw_ce, raw_top1 = cached(tag)
    parity = runs["frozen", "legal"]["ce"] - raw_ce[ix]
    n = len(ix)
    f = lambda t, fmt: f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"
    lines = [
        f"n = {n:,} positions ({n // 4:,} per band). Root parity: search-oracle legal CE minus the cached raw "
        f"scorer, mean {parity.mean():+.5f} nats (per-position p99 |diff| {np.percentile(abs(parity), 99):.3f}, "
        f"BF16); same top-1 correctness (right or wrong) on {100 * (raw_top1[ix] == runs['frozen', 'legal']['top1']).mean():.1f}% of positions. "
        f"Attention: {attn:.4f} GFLOPs per evaluation at the mean context.",
        "",
        "| Point | new NN evals/move | GFLOPs/move | ms/move | CE | acc % | CE >=2400 | acc >=2400 |"
        " dCE vs 79M | dCE vs 23M | dAcc vs 79M (pp) | dAcc vs 23M (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    res = {}
    for k, p in pts.items():
        e = cell == 7
        r = res[k] = dict(
            label=p["label"],
            nodes=p["nodes"],
            gflops=p["gflops"],
            ms=p.get("ms"),
            ce=float(p["ce"].mean()),
            acc=float(100 * p["top1"].mean()),
            d_ce={m: ci(game, p["ce"] - pts["maia", m]["ce"], rng) for m in MAIA},
            d_acc={
                m: ci(game, 100 * (p["top1"] - pts["maia", m]["top1"]), rng)
                for m in MAIA
            },
            bands={
                BANDS[c]: dict(
                    ce=ci(game[cell == c], p["ce"][cell == c], rng),
                    acc=ci(game[cell == c], 100 * p["top1"][cell == c], rng),
                )
                for c in BANDS
            },
        )
        ms = f"{r['ms']:.2f}" if r["ms"] else "-"
        lines.append(
            f"| {r['label']} | {r['nodes']:.1f} | {r['gflops']:.2f} | {ms} | {r['ce']:.4f} | {r['acc']:.2f} | "
            f"{p['ce'][e].mean():.4f} | {100 * p['top1'][e].mean():.2f} | "
            + " | ".join(f(r["d_ce"][m], "{:+.4f}") for m in ("maia3-79m", "maia3-23m"))
            + " | "
            + " | ".join(
                f(r["d_acc"][m], "{:+.2f}") for m in ("maia3-79m", "maia3-23m")
            )
            + " |"
        )
    lines += [
        "",
        "Per band, CE ± 95% half-width / accuracy %:",
        "",
        "| Point | " + " | ".join(BANDS.values()) + " |",
        "|---|" + "---:|" * 4,
    ]
    for r in res.values():
        lines.append(
            f"| {r['label']} | "
            + " | ".join(
                f"{b['ce'][0]:.4f} ±{(b['ce'][2] - b['ce'][1]) / 2:.4f} / {b['acc'][0]:.1f}"
                for b in r["bands"].values()
            )
            + " |"
        )
    lines += [
        "",
        "Pareto check (a point of ours with less compute, lower CE and higher accuracy):",
        "",
    ]
    ours = [r for k, r in res.items() if k[0] != "maia"]
    for m, l in MAIA.items():
        mm = res["maia", m]
        cheaper = [r for r in ours if r["gflops"] < mm["gflops"]]
        both = [r for r in cheaper if r["ce"] < mm["ce"] and r["acc"] > mm["acc"]]
        if not cheaper:
            lines.append(
                f"- {l} ({mm['gflops']:.2f} GFLOPs): no point of ours is cheaper"
            )
            continue
        best = min(cheaper, key=lambda r: r["ce"])
        lines.append(
            f"- {l} ({mm['gflops']:.2f} GFLOPs, {mm['ce']:.4f} / {mm['acc']:.2f}%): "
            + (
                "cheaper with lower CE and higher accuracy: "
                + "; ".join(
                    f"{r['label']} ({r['gflops']:.2f}, {r['ce']:.4f} / {r['acc']:.2f}%)"
                    for r in both
                )
                if both
                else f"none; lowest-CE cheaper point is {best['label']} ({best['gflops']:.2f} GFLOPs, "
                f"{best['ce']:.4f} / {best['acc']:.2f}%)"
            )
            + f"; lowest-CE cheaper point minus {l}: dCE {f(best['d_ce'][m], '{:+.4f}')}, "
            f"dAcc {f(best['d_acc'][m], '{:+.2f}')} pp"
        )
    (DATA / "search" / f"report-{tag}.json").write_text(
        json.dumps(
            dict(
                n=n,
                parity=float(parity.mean()),
                attention_gflops=attn,
                points={"/".join(k): v for k, v in res.items()},
            ),
            indent=1,
        )
        + "\n"
    )
    (HERE / f"tables-{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res, n


def plot(results):
    """One figure: Maia-3, then per model its dev-calibrated line (solid) and frozen line (dashed)."""
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    hue = {
        "sw-moe-s42": "#2a78d6",
        "sw-dense-s42": "#1baf7a",
        "bigrun132k": "#4a3aa7",
        "bigrun": "#4a3aa7",
    }
    big = results[0][0].startswith("bigrun")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=surface)
    n = min(r[2] for r in results)
    # every tag is a prefix of run.py's one fixed order, so equal n means identical positions
    assert all(r[2] == n for r in results), [r[2] for r in results]
    for ax, key, title in (
        (axes[0], "ce", "CE over legal moves (nats, lower is better)"),
        (axes[1], "acc", "Top-1 accuracy % (higher is better)"),
    ):
        ax.set_facecolor(surface)
        res = results[0][1]
        maia = [("maia", m) for m in MAIA]
        ax.plot(
            [res[k]["gflops"] for k in maia],
            [res[k][key] for k in maia],
            color="#eb6834",
            lw=2,
            marker="o",
            ms=8,
            mec=surface,
            mew=2,
            zorder=3,
            label="Maia-3 5M / 23M / 79M (no search)",
        )
        for k in maia:
            ax.annotate(
                MAIA[k[1]].replace("Maia-3 ", ""),
                (res[k]["gflops"], res[k][key]),
                textcoords="offset points",
                xytext=(0, -15 if (key == "ce") != big else 9),
                ha="center",
                fontsize=8.5,
                color=muted,
            )
        for tag, res, _ in results:
            for group, ls, name in (
                ("devcal", "-", "dev-calibrated"),
                ("frozen", "--", "frozen calibration"),
            ):
                if (
                    big and tag != results[0][0] and group == "frozen"
                ):  # big-run figure: references dev-calibrated only
                    continue
                keys = [("frozen", "legal")] + [
                    k
                    for k in res
                    if k[0] == group and k[1] not in ("legal", "adaptive")
                ]
                if len(keys) < 2:
                    continue
                ax.plot(
                    [res[k]["gflops"] for k in keys],
                    [res[k][key] for k in keys],
                    color=hue[tag],
                    lw=2 if ls == "-" else 1.5,
                    ls=ls,
                    marker="o",
                    ms=7 if ls == "-" else 5,
                    mec=surface,
                    mew=1.5,
                    zorder=3,
                    label=f"Ours, {MODEL[tag][0]}: raw + search, {name}",
                )
                if (group, "adaptive") in res:
                    r = res[group, "adaptive"]
                    ax.plot(
                        r["gflops"],
                        r[key],
                        color=hue[tag],
                        marker="D",
                        ms=7 if ls == "-" else 5,
                        mec=surface,
                        mew=1.5,
                        ls="none",
                        zorder=3,
                    )
                if tag == results[0][0] and group == "devcal":
                    for k in keys:
                        text = "raw" if k[1] == "legal" else f"{k[1]}"
                        ax.annotate(
                            text,
                            (res[k]["gflops"], res[k][key]),
                            textcoords="offset points",
                            xytext=(0, 9 if (key == "ce") != big else -15),
                            ha="center",
                            fontsize=8,
                            color=muted,
                        )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30, 100, 300]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100", "300"])
        ax.set_xlabel("GFLOPs per move (log scale)", color=muted)
        ax.set_title(title, color=ink, fontsize=10.5, loc="left")
        ax.grid(True, color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=muted,
    )
    fig.suptitle(
        f"Maia-3 blitz benchmark, {n:,} paired positions. Point labels: simulations per move; "
        "diamonds: Elo-adaptive budget",
        color=muted,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(
        HERE / (f"pareto-{results[0][0]}.png" if big else "pareto.png"),
        dpi=160,
        facecolor=surface,
    )


def main():
    plot([(tag, *table(tag)) for tag in sys.argv[1:]])


if __name__ == "__main__":
    main()
