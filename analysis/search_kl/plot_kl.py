"""The piKL track's scaling figure: per dev cell, CE at human-matched accuracy minus the raw policy's against leaf
evaluations (lower is better; below 0 passes the CE bar). Points where no tilt reaches the humans' accuracy sit
on the top strip as open markers.

python plot_kl.py OUT.png --series 'label=family: name' ...
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(
    "/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/search-eff"
)
CELLS = ["classical/2400", "classical/2600", "rapid/2600", "blitz/2400"]
COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#8a8984",
]  # reference slots 1-3, then a neutral for the shipped baseline
TOP = 0.20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--series", nargs="+", required=True)
    p.add_argument(
        "--curves", default=f"{R}/pikl/curves.jsonl,{R}/scaling/curves.jsonl"
    )
    a = p.parse_args()
    rows = {}
    for f in a.curves.split(","):
        for line in open(f):
            r = json.loads(line)
            rows[f"{r['family']}: {r['name']}", r["cell"], r["budget"]] = r
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#a1a1aa", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for j, cell in enumerate(CELLS):
        ax = axes[j]
        ax.axhline(0, color="#52514e", lw=1)
        ax.axhspan(TOP * 0.92, TOP * 1.08, color="#f1f0ec", lw=0)
        for k, spec in enumerate(a.series):
            label, key = spec.split("=", 1)
            pts = defaultdict(dict)
            for (name, c, b), r in rows.items():
                if name == key and c == cell:
                    pts[b] = r
            if not pts:
                continue
            xs = sorted(pts)
            col = COLORS[k % len(COLORS)]
            ls = "--" if k == len(COLORS) - 1 else "-"
            hit = [
                (pts[b]["leaves"], pts[b]["ce_match"])
                for b in xs
                if pts[b]["ce_match"] is not None
            ]
            miss = [pts[b]["leaves"] for b in xs if pts[b]["ce_match"] is None]
            if hit:
                ax.plot(*zip(*[(x, min(y, TOP)) for x, y in hit]), color=col, ls=ls, lw=2, marker="o", ms=4.5,
                        label=label if j == 0 else None)  # fmt: skip
            if miss:
                ax.plot(miss, [TOP] * len(miss), ls="none", marker="o", ms=5, mfc="none", mec=col, mew=1.5,
                        label=None if hit or j else label)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 128, 512, 2048])
        ax.set_xticklabels(["8", "32", "128", "512", "2k"])
        ax.set_title(cell.replace("/", " "), color="#0b0b0b")
        ax.set_xlabel("leaf evaluations")
        ax.grid(axis="y", color="#e5e7eb")
        ax.set_ylim(-0.07, TOP * 1.1)
    axes[0].set_ylabel("CE at matched accuracy − raw (nats)")
    axes[0].text(
        9, TOP * 0.82, "open: accuracy unreachable", fontsize=8, color="#52514e"
    )
    fig.legend(
        loc="lower center",
        ncol=len(a.series),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
