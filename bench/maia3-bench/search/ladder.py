"""Search gain vs training compute: raw vs 128-sim search (own dev refit, and frozen) per sweep optimum.

Same 20K positions for every model; gain = raw CE - searched CE, paired, 95% game bootstrap.
Raw is the search oracle's own legal policy; parity is against the training-forward reference.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

import report  # noqa: E402

D = report.DATA / "search"
LADDER = {  # family -> [(training FLOPs, tag, run)]
    "MoE": [
        (1e17, "sw1e17-moe-s42", "sw-1e17-s16-10x448"),
        (3e17, "sw3e17-moe-s42", "sw-3e17-s16-14x640"),
        (1e18, "sw-moe-s42", "sw-1e18-s16-18x896"),
    ],
    "dense": [
        (1e17, "sw1e17-dense-s42", "sw-1e17-dense-12x512"),
        (3e17, "sw3e17-dense-s42", "sw-3e17-dense-14x640"),
        (1e18, "sw-dense-s42", "sw-1e18-dense-18x896"),
    ],
}


def arrays(folder):
    z = [np.load(f) for f in sorted(folder.glob("[0-9]*[0-9].npz"))]
    return {
        k: np.concatenate([x[k] for x in z]) for k in ("index", "ce", "top1", "nodes")
    }


def reference(tag):
    for folder in (D / "scores", report.DATA / "scores"):
        if (folder / f"{tag}.npz").exists():
            return np.load(folder / f"{tag}.npz")["ce_legal"]


def main():
    rng = np.random.default_rng(7)
    with np.load(report.DATA / "games.npz") as z:
        s = z["sel"][z["keep"]]
    rows, out = [], {}
    for family, rungs in LADDER.items():
        for flops, tag, run in rungs:
            if not (D / tag / f"devcal-128.npz").exists():
                continue
            raw, frozen = arrays(D / tag / "legal"), arrays(D / tag / "128")
            cal = dict(np.load(D / tag / "devcal-128.npz"))
            n = min(len(raw["ce"]), len(frozen["ce"]), len(cal["ce"]))
            ix = raw["index"][:n]
            assert np.array_equal(frozen["index"][:n], ix) and np.array_equal(
                cal["index"][:n], ix
            )
            game, e = s[ix, 0], s[ix, 2] == 7
            ref = reference(tag)
            r = dict(
                family=family,
                flops=flops,
                run=run,
                n=int(n),
                raw=float(raw["ce"][:n].mean()),
                raw_acc=float(100 * raw["top1"][:n].mean()),
                parity=float((raw["ce"][:n] - ref[ix]).mean())
                if ref is not None
                else None,
                nodes=float(frozen["nodes"][:n].mean()),
            )
            for name, x in (("refit", cal), ("frozen", frozen)):
                r[name] = dict(
                    ce=float(x["ce"][:n].mean()),
                    acc=float(100 * x["top1"][:n].mean()),
                    gain=report.ci(game, raw["ce"][:n] - x["ce"][:n], rng),
                    gain_2400=report.ci(
                        game[e], raw["ce"][:n][e] - x["ce"][:n][e], rng
                    ),
                    acc_gain=report.ci(
                        game, 100 * (x["top1"][:n].astype(float) - raw["top1"][:n]), rng
                    ),
                )
            out[tag] = r
            f = lambda t, fmt="{:+.4f}": (
                f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"
            )
            rows.append(
                f"| {family} | {f"{flops:.0e}".replace("e+", "e")} | {run} | {r['raw']:.4f} / {r['raw_acc']:.2f} | "
                f"{r['refit']['ce']:.4f} / {r['refit']['acc']:.2f} | {f(r['refit']['gain'])} | "
                f"{f(r['refit']['gain_2400'])} | {f(r['refit']['acc_gain'], '{:+.2f}')} | "
                f"{f(r['frozen']['gain'])} | "
                + ("-" if r["parity"] is None else f"{r['parity']:+.4f}")
                + " |"
            )
    head = [
        "| Family | Train FLOPs | Run | Raw CE / acc | 128 sims refit CE / acc | CE gain (refit) | "
        "CE gain >=2400 (refit) | Acc gain pp (refit) | CE gain (frozen) | Raw parity |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    text = "\n".join(head + rows) + "\n"
    (report.HERE / "ladder.md").write_text(text)
    (D / "ladder.json").write_text(json.dumps(out, indent=1) + "\n")
    print(text)
    plot(out)


def plot(out):
    muted, grid, surface, ink = "#52514e", "#e4e3df", "#fcfcfb", "#0b0b0b"
    hue = {"MoE": "#2a78d6", "dense": "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=surface)
    for ax, key, title in (
        (axes[0], "gain", "CE gain from 128-sim search, all bands (nats)"),
        (axes[1], "gain_2400", "CE gain from 128-sim search, >=2400 band (nats)"),
    ):
        ax.set_facecolor(surface)
        for family, color in hue.items():
            pts = sorted(
                (r["flops"], r["refit"][key])
                for r in out.values()
                if r["family"] == family
            )
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = np.array([p[1] for p in pts])
            ax.errorbar(
                x,
                y[:, 0],
                yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]],
                color=color,
                lw=2,
                marker="o",
                ms=7,
                capsize=3,
                label=f"{family}, own dev refit",
            )
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([1e17, 3e17, 1e18]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["1e17", "3e17", "1e18"])
        ax.set_xlabel("training FLOPs (sweep optimum per budget)", color=muted)
        ax.set_title(title, color=ink, fontsize=10.5, loc="left")
        ax.grid(True, color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=muted)
    fig.tight_layout()
    fig.savefig(report.HERE / "ladder.png", dpi=160, facecolor=surface)


if __name__ == "__main__":
    main()
