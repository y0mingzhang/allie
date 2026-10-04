"""Head-to-head tables for Allie 2.0 (MoE 0.69B active / 5.6B total) against Maia-3 5M / 23M / 79M and
the best 1e18 sweep MoE, on the benchmark's 80,000 blitz positions (eval.maia3.aggregate's views and game
bootstrap) and on the by-format companion sample (eval.maia3.formats: 5,000 per bullet / rapid / classical
cell). Slices: Elo band, 200-Elo bin of the mover, blitz time control, game format, mover's clock, ply.

usage: eval.maia3.report STEP          (bigrun-v2/step-STEP/scores.npz, formats/scores/bigrun-STEP.npz)
       eval.maia3.report NAME SCORES   (SCORES: NAME's benchmark scores.npz, beside its .json; formats/scores/NAME.npz)
Writes tables-bigrun.md and aggregate-bigrun.json, or tables-NAME.md and aggregate-NAME.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from allie import paths
from allie.data.vocab import INCREMENTS, SECONDS
from allie.eval.maia3.aggregate import CELLS, DATA, boot, load, stats

HERE = paths.ROOT / "results/recipe10x/maia3-bench"  # flops.json in, tables out
FMT = DATA / "formats"
ORDER = ["maia3-5m", "maia3-23m", "maia3-79m", "sw-moe-s42", "bigrun"]
LABEL = {
    "maia3-5m": "Maia-3 5M",
    "maia3-23m": "Maia-3 23M",
    "maia3-79m": "Maia-3 79M",
    "sw-moe-s42": "best 1e18 sweep MoE",
    "bigrun": "**big run**",
}
GFLOPS = {
    "sw-moe-s42": 0.362,
    "bigrun": 1.389,
}  # 2 x active matmul params; the Maia-3 models' come from flops.json
PARAMS = {
    "maia3-5m": "5M",
    "maia3-23m": "23M",
    "maia3-79m": "79M",
    "sw-moe-s42": "MoE 181M active / 1.42B total",
    "bigrun": "MoE 0.69B active / 5.6B total",
}
BANDS = list(CELLS.values())
ELO = [(0, 800)] + [(a, a + 200) for a in range(800, 2600, 200)] + [(2600, 10**4)]
CLOCK = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 10**6)]
PLY = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 10**6)]
TC = ["180+0", "180+2", "300+0", "300+3"]
FORMATS = ["bullet", "blitz", "rapid", "classical"]
REFS = ("maia3-79m", "sw-moe-s42")


def scores(path, keep):
    with np.load(path) as z:
        ce, top1 = (
            (z["ce_legal"], z["top1"]) if "ce_legal" in z else (z["ce"], z["top1"])
        )
    if keep is not None and len(ce) != len(keep):
        ce, top1 = ce[keep], top1[keep]
    return dict(ce=np.asarray(ce, float), top1=np.asarray(top1, float))


def ci(d, scale=1, fmt="{:+.4f}"):
    f = lambda v: fmt.format(v * scale)
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"


def half(games, x, rng):
    lo, hi = np.percentile(boot(games, x, rng), [2.5, 97.5])
    return (hi - lo) / 2


def macro_boot(games, cell, x, rng, reps=2000):
    """Bootstrap draws of the macro over cells of mean(x): whole games resampled once per draw, shared by
    every cell, and each cell's mean recomputed from that draw before the cells are averaged."""
    _, g = np.unique(games, return_inverse=True)
    cells = np.unique(cell)
    pick = rng.integers(0, g.max() + 1, (reps, g.max() + 1))
    out = np.zeros(reps)
    for c in cells:
        k = cell == c
        sx = np.bincount(g[k], x[k], minlength=g.max() + 1)
        sn = np.bincount(g[k], minlength=g.max() + 1).astype(float)
        out += sx[pick].sum(1) / sn[pick].sum(1)
    return out / len(cells)


def label(lo, hi, unit=""):
    return (
        f"{lo}+{unit}"
        if hi >= 10**4
        else (f"<{hi}{unit}" if lo == 0 and unit == "" else f"{lo}-{hi}{unit}")
    )


def macro(s, mask, x):
    return float(np.mean([x[mask & (s[:, 2] == c)].mean() for c in CELLS]))


def overall(s, masks, sc, rng, out, lines):
    for vname, title in (
        ("all", "Every scored move in the cell (our protocol)"),
        ("maia_protocol", "Maia-3's protocol (from ply 10, until a player first has under 30 s)"),
    ):
        vm = masks[vname]
        r = out["views"][vname] = {}
        lines += [
            f"### Overall: {title}",
            "",
            f"{int(vm.sum()):,} of {len(s):,} positions.",
            "",
        ]
        for metric, fmt, scale, name in (
            ("ce", "{:.4f}", 1, "Cross-entropy over legal moves (nats)"),
            ("top1", "{:.1f}", 100, "Top-1 move-matching accuracy (%)"),
        ):
            lines += [
                f"**{name}**",
                "",
                "| Model | Params | GFLOPs/move | "
                + " | ".join(b.split("/")[1] for b in BANDS)
                + " | macro |",
                "|---|---|---:|" + "---:|" * (len(BANDS) + 1),
            ]
            for m in ORDER:
                cells = []
                for c in CELLS:
                    k = vm & (s[:, 2] == c)
                    x = sc[m][metric][k]
                    h = half(s[k, 0], x, rng)
                    cells.append(
                        fmt.format(x.mean() * scale)
                        + " ±"
                        + fmt.format(h * scale).lstrip("0")
                    )
                mac = macro(s, vm, sc[m][metric])
                r.setdefault(m, {})[metric] = mac
                lines.append(
                    f"| {LABEL[m]} | {PARAMS[m]} | {GFLOPS[m]:.2f} | "
                    + " | ".join(cells)
                    + f" | {fmt.format(mac * scale)} |"
                )
            lines.append("")
        lines += [
            "**Big run minus each model, same positions, pooled over the four bands "
            "(negative CE / positive accuracy favours the big run)**",
            "",
            "| vs | delta CE (nats) | delta accuracy (pp) |",
            "|---|---:|---:|",
        ]
        for m in ORDER[:-1]:
            dce = stats(s[vm, 0], sc["bigrun"]["ce"][vm] - sc[m]["ce"][vm], rng)
            dacc = stats(s[vm, 0], sc["bigrun"]["top1"][vm] - sc[m]["top1"][vm], rng)
            r.setdefault("paired", {})[m] = dict(d_ce=dce, d_acc=dacc)
            lines.append(f"| {LABEL[m]} | {ci(dce)} | {ci(dacc, 100, '{:+.2f}')} |")
        lines.append("")


def sliced(
    name,
    title,
    groups,
    sc,
    rng,
    out,
    lines,
    note="Pooled over the four blitz bands.",
):
    """groups: [(label, mask, games)]; per group each model's acc / CE and Allie 2.0's paired deltas."""
    rows = out["slices"][name] = {}
    lines += [
        f"### {title}",
        "",
        f"Accuracy % / CE (nats). {note} Deltas: big run minus the model, 95% game-bootstrap interval.",
        "",
        "| "
        + name
        + " | n | "
        + " | ".join(LABEL[m].strip("*") for m in ORDER)
        + " | dCE vs 79M | dAcc vs 79M (pp) | dCE vs sweep MoE |",
        "|---|---:|" + "---:|" * (len(ORDER) + 3),
    ]
    for g, k, games in groups:
        if not k.any():
            continue
        d = rows[g] = dict(
            n=int(k.sum()),
            models={
                m: dict(
                    acc=float(sc[m]["top1"][k].mean()), ce=float(sc[m]["ce"][k].mean())
                )
                for m in ORDER
            },
        )
        for ref in REFS:
            d[f"d_ce_{ref}"] = stats(
                games[k], sc["bigrun"]["ce"][k] - sc[ref]["ce"][k], rng
            )
        d["d_acc_maia3-79m"] = stats(
            games[k], sc["bigrun"]["top1"][k] - sc["maia3-79m"]["top1"][k], rng
        )
        lines.append(
            f"| {g} | {d['n']:,} | "
            + " | ".join(
                f"{v['acc'] * 100:.1f} / {v['ce']:.4f}" for v in d["models"].values()
            )
            + f" | {ci(d['d_ce_maia3-79m'])} | {ci(d['d_acc_maia3-79m'], 100, '{:+.2f}')} | {ci(d['d_ce_sw-moe-s42'])} |"
        )
    lines.append("")


def main():
    if len(sys.argv) > 2:
        name, bench, out_name = sys.argv[1], Path(sys.argv[2]), sys.argv[1]
    else:
        step = int(sys.argv[1])
        name, out_name = f"bigrun-{step}", "bigrun"
        bench = DATA / "bigrun-v2" / f"step-{step:08d}" / "scores.npz"
    rng = np.random.default_rng(7)
    s, masks, cached = load()
    keep = np.load(DATA / "games.npz")["keep"]
    sc = {
        m: dict(ce=cached[m]["ce"].astype(float), top1=cached[m]["top1"].astype(float))
        for m in ORDER[:-1]
    }
    sc["bigrun"] = scores(bench, keep)
    fl = json.loads((HERE / "flops.json").read_text())
    GFLOPS.update({m: fl[m]["flops_per_move"] / 1e9 for m in ORDER[:3]})
    info = json.loads(bench.with_suffix(".json").read_text())
    step = info["step"]
    out = dict(step=step, views={}, slices={}, model_sha256=info["model_sha256"])
    lines = [
        f"{name}: step {step:,} ({info['tokens'] / 1e9:.1f}B tokens), checkpoint sha256 {info['model_sha256'][:12]}.",
        "",
    ]
    overall(s, masks, sc, rng, out, lines)

    with np.load(DATA / "games.npz") as z:
        meta = z["meta"]
    g, ply = s[:, 0], s[:, 1]
    elo = np.where(ply % 2 == 0, meta[g, 2], meta[g, 3])
    sliced(
        "Elo",
        "By the mover's rating, 200-point bins",
        [(label(a, b), (elo >= a) & (elo < b), g) for a, b in ELO],
        sc,
        rng,
        out,
        lines,
    )
    tc = np.array([f"{SECONDS[a - 192]}+{INCREMENTS[b - 10]}" for a, b in meta[g, :2]])
    other = ~np.isin(tc, TC)
    sliced(
        "time control",
        "By blitz time control (base seconds + increment)",
        [(t, tc == t, g) for t in TC] + [("other", other, g)],
        sc,
        rng,
        out,
        lines,
    )
    sliced(
        "clock",
        "By time left on the mover's clock",
        [
            (
                label(a, b, " s") if b < 10**6 else f"{a}+ s",
                (s[:, 3] >= a) & (s[:, 3] < b),
                g,
            )
            for a, b in CLOCK
        ],
        sc,
        rng,
        out,
        lines,
    )
    sliced(
        "ply",
        "By ply (0-based half-move index)",
        [
            (f"{a}+" if b >= 10**6 else f"{a}-{b - 1}", (ply >= a) & (ply < b), g)
            for a, b in PLY
        ],
        sc,
        rng,
        out,
        lines,
    )

    f = FMT / "scores" / f"{name}.npz"
    if f.exists() and all((FMT / "scores" / f"{m}.npz").exists() for m in ORDER[:-1]):
        with np.load(FMT / "games.npz") as z:
            fs = z["sel"][z["keep"]]
        fsc = {m: scores(FMT / "scores" / f"{m}.npz", None) for m in ORDER[:-1]}
        fsc["bigrun"] = scores(f, None)
        # one sample: the 12 other cells (5,000 each) and the blitz benchmark (20,000 each); games are disjoint
        # across the two files, so offset the companion's game ids
        cat = {
            m: {k: np.r_[fsc[m][k], sc[m][k]] for k in ("ce", "top1")} for m in ORDER
        }
        games = np.r_[fs[:, 0] + s[:, 0].max() + 1, s[:, 0]]
        cell = np.r_[fs[:, 2], s[:, 2]]
        rows = out["slices"]["format"] = {}
        lines += [
            "### By game format (macro over the four Elo bands)",
            "",
            "Bullet / rapid / classical: build_formats.py's 5,000 positions per cell; blitz: the 80,000 above. "
            "Accuracy % / CE (nats); deltas: big run minus the model, 95% game-bootstrap interval of the macro.",
            "",
            "| format | n | "
            + " | ".join(LABEL[m].strip("*") for m in ORDER)
            + " | dCE vs 79M | dAcc vs 79M (pp) | dCE vs sweep MoE |",
            "|---|---:|" + "---:|" * (len(ORDER) + 3),
        ]
        for i, name in enumerate(FORMATS):
            k = (cell >= 4 * i) & (cell < 4 * i + 4)
            mac = lambda x: float(np.mean([x[k & (cell == c)].mean() for c in np.unique(cell[k])]))
            d = rows[name] = dict(
                n=int(k.sum()),
                models={m: dict(acc=mac(cat[m]["top1"]), ce=mac(cat[m]["ce"])) for m in ORDER},
            )
            for ref, key in (("maia3-79m", "ce"), ("sw-moe-s42", "ce"), ("maia3-79m", "top1")):
                x = cat["bigrun"][key] - cat[ref][key]
                lo, hi = np.percentile(macro_boot(games[k], cell[k], x[k], rng), [2.5, 97.5])
                d[f"d_{key}_{ref}"] = dict(mean=mac(x), lo=float(lo), hi=float(hi))
            lines.append(
                f"| {name} | {d['n']:,} | "
                + " | ".join(
                    f"{v['acc'] * 100:.1f} / {v['ce']:.4f}"
                    for v in d["models"].values()
                )
                + f" | {ci(d['d_ce_maia3-79m'])} | {ci(d['d_top1_maia3-79m'], 100, '{:+.2f}')} |"
                f" {ci(d['d_ce_sw-moe-s42'])} |"
            )
        lines += [
            "",
            "Per cell, CE (nats):",
            "",
            "| cell | n | " + " | ".join(LABEL[m].strip("*") for m in ORDER) + " |",
            "|---|---:|" + "---:|" * len(ORDER),
        ]
        names = json.loads(
            (DATA.parent / "strat-eval-v1" / "manifest.json").read_text()
        )["cells"]
        for c in range(16):
            k = cell == c
            lines.append(
                f"| {names[c]} | {int(k.sum()):,} | "
                + " | ".join(f"{cat[m]['ce'][k].mean():.4f}" for m in ORDER)
                + " |"
            )
        lines.append("")

    (HERE / f"aggregate-{out_name}.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / f"tables-{out_name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
