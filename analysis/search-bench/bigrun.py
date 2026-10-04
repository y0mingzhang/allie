"""Search gain of Allie 2.0 (MoE 0.69B active / 5.6B total) per budget and Elo band, next to the sweep ladder.

usage: bigrun.py <tag> [step]   (<tag>/<budget> from run.py and fit.py's devcal-*.npz; prefix every point finished)
Gain = the oracle's own raw (legal) CE minus the searched CE on the same positions, 95% game bootstrap.
Writes bigrun-<tag>.md, D/bigrun-<tag>.json and ladder-<tag>.png (MoE 128-sim refit gain vs useful training FLOPs).
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import report  # noqa: E402

D = report.DATA / "search"
BUDGETS = ["5", "8", "25", "128", "460"]
# useful training FLOPs of the sweep's MoE optima (sweep-readout-c8s200f0v4-w4.json "c")
LADDER = {
    "sw1e17-moe-s42": 6.296e17,
    "sw3e17-moe-s42": 1.850e18,
    "sw-moe-s42": 6.164e18,
}


def sha(path):
    """sha256 of a checkpoint, cached next to it (keyed by inode, size and mtime)."""
    st = path.stat()
    cache = path.with_name(f"{path.name}.sha256")
    key = f"{st.st_ino} {st.st_size} {st.st_mtime_ns}"
    if cache.exists() and cache.read_text().split("\n")[0] == key:
        return cache.read_text().split("\n")[1]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    cache.write_text(f"{key}\n{h.hexdigest()}\n")
    return h.hexdigest()


def main():
    tag = sys.argv[1]
    rng = np.random.default_rng(7)
    with np.load(report.DATA / "games.npz") as z:
        s = z["sel"][z["keep"]]
    runs, ix = report.load(tag)
    game, cell = s[ix, 0], s[ix, 2]
    raw = runs["frozen", "legal"]
    info = json.loads(report.RAW[tag].with_suffix(".json").read_text())
    # bench and dev folders and the raw reference must all be the same checkpoint
    mb, md = (json.loads((D / t / "manifest.json").read_text()) for t in (tag, f"{tag}-dev"))
    assert mb["model"] == md["model"] and (mb["set"], md["set"]) == ("bench", "dev"), (mb, md)
    assert sha(Path(mb["model"])) == info["model_sha256"], (mb["model"], report.RAW[tag])
    f = lambda t, fmt="{:+.4f}": (
        f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"
    )
    out = dict(
        tag=tag,
        n=len(ix),
        step=info["step"],
        useful_training_flops=info["useful_training_flops"],
        gains={},
    )
    lines = [
        f"n = {len(ix):,} positions ({len(ix) // 4:,} per band); big run step {info['step']:,}. "
        "Gain = raw CE minus searched CE (positive = search helps), 95% game-bootstrap intervals.",
        "",
        "| Budget | calibration | new evals/move | CE gain | acc gain (pp) | "
        + " | ".join(f"CE gain {b}" for b in report.BANDS.values())
        + " |",
        "|---|---|---:|---:|---:|" + "---:|" * 4,
    ]
    for b in BUDGETS:
        for group, name in (("devcal", "dev refit"), ("frozen", "frozen")):
            if (group, b) not in runs:
                continue
            x = runs[group, b]
            d = raw["ce"] - x["ce"]
            g = out["gains"][f"{b}/{group}"] = dict(
                nodes=float(runs["frozen", b]["nodes"].mean()),
                ce=report.ci(game, d, rng),
                acc=report.ci(game, 100 * (x["top1"].astype(float) - raw["top1"]), rng),
                bands={
                    report.BANDS[c]: report.ci(game[cell == c], d[cell == c], rng)
                    for c in report.BANDS
                },
            )
            lines.append(
                f"| {b} | {name} | {g['nodes']:.1f} | {f(g['ce'])} | {f(g['acc'], '{:+.2f}')} | "
                + " | ".join(f(v) for v in g["bands"].values())
                + " |"
            )
    lad = json.loads((D / "ladder.json").read_text())
    pts = [
        (LADDER[t], lad[t]["refit"]["gain"], lad[t]["refit"]["gain_2400"], t)
        for t in LADDER
    ]
    if "128/devcal" in out["gains"]:
        g = out["gains"]["128/devcal"]
        e = cell == 7
        pts.append(
            (
                info["useful_training_flops"],
                list(g["ce"]),
                list(
                    report.ci(
                        game[e], (raw["ce"] - runs["devcal", "128"]["ce"])[e], rng
                    )
                ),
                tag,
            )
        )
        lines += [
            "",
            "128-simulation gain (own dev refit) against training compute, MoE family:",
            "",
            "| Run | useful training FLOPs | raw CE | CE gain | CE gain >=2400 |",
            "|---|---:|---:|---:|---:|",
        ]
        for c, gain, g24, t in pts:
            rawce = lad[t]["raw"] if t in lad else float(raw["ce"].mean())
            lines.append(f"| {t} | {c:.2e} | {rawce:.4f} | {f(gain)} | {f(g24)} |")
        plot(pts, tag)
    out["ladder"] = [
        dict(run=t, flops=c, gain=gain, gain_2400=g24) for c, gain, g24, t in pts
    ]
    (D / f"bigrun-{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    (report.HERE / f"bigrun-{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(pts, tag):
    muted, grid, surface, ink, blue, violet = (
        "#52514e",
        "#e4e3df",
        "#fcfcfb",
        "#0b0b0b",
        "#2a78d6",
        "#4a3aa7",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=surface)
    for ax, k, title in (
        (axes[0], 1, "CE gain from 128-sim search, all bands (nats)"),
        (axes[1], 2, "CE gain from 128-sim search, >=2400 band (nats)"),
    ):
        ax.set_facecolor(surface)
        x = [p[0] for p in pts]
        y = np.array([p[k] for p in pts])
        err = [y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]]
        ax.errorbar(x[:3], y[:3, 0], yerr=[e[:3] for e in err], color=blue, lw=2, marker="o", ms=7, capsize=3,
                    label="sweep MoE optima (1e17 / 3e17 / 1e18 budgets)")  # fmt: skip
        if len(pts) > 3:
            ax.errorbar(x[3:], y[3:, 0], yerr=[e[3:] for e in err], color=violet, marker="D", ms=8, capsize=3,
                        ls="none", label="big run, MoE 0.69B active / 5.6B total")  # fmt: skip
            ax.plot(x[2:], y[2:, 0], color=muted, lw=1, ls=":")
            ax.annotate(f"{y[3, 0]:+.4f}", (x[3], y[3, 0]), xytext=(-10, 0), textcoords="offset points", ha="right",
                        fontsize=8.5, color=ink)  # fmt: skip
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("useful training FLOPs", color=muted)
        ax.set_title(title, color=ink, fontsize=10.5, loc="left")
        ax.grid(True, color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=muted)
    fig.tight_layout()
    fig.savefig(report.HERE / f"ladder-{tag}.png", dpi=160, facecolor=surface)


if __name__ == "__main__":
    main()
