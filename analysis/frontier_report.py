"""distill-v2 tables and Pareto plot: Maia-bench CE / top-1 (all 80K positions, macro over the four blitz bands),
paired differences with game-bootstrap 95% intervals against each run's control, Maia-3 23M / 79M and Allie 2.0,
golden (strat-eval-v1) macro / expert / per format, and cost in GFLOPs per move.

Runs: every scores/NAME.npz under $G/distill-v2 (eval.maia3.score_moe). Controls: kd-sW-aA-tS... -> kd-sW-a0-tS; ann-* ->
Allie 2.0's final checkpoint. Writes tables.md, report.json, pareto.png next to this file."""

import json
import re
from pathlib import Path

import numpy as np

from allie import paths
from allie.eval.maia3.aggregate import DATA, stats, views

HERE = Path(__file__).resolve().parent
G = paths.DATA
ROOT = paths.ROOT / "results"
SCORES = G / "distill-v2/scores"
BIG = DATA / "bigrun-v2"
FINAL = 143051
FORMATS = ("bullet", "blitz", "rapid", "classical")
MAIA = {"maia3-5m": 0.602, "maia3-23m": 2.402, "maia3-79m": 9.232}
STUDENT = {"896": "sw-1e18-s16-18x896-c8s200f0v4-s42", "1024": "sw-1e18-s16-20x1024-c8s200f0v4-s42",
           "1152": "sw-1e18-s16-22x1152-c8s200f0v4-s42"}  # fmt: skip


def gflops(run):
    from allie.model.arch import extra_flops

    c = json.loads((ROOT / "pretrain" / run / "config.json").read_text())["config"]
    d, L, heads = c["width"], c["layers"], c["width"] // c["head_dim"]
    gates = L + 2 * min(5, L // 2)
    f = (
        24 * L * d * d
        + 2 * d * 2432
        + 2 * (gates * heads * 16 + 64)
        + extra_flops(c["arch"], d, L)
    )
    return f / 1e9


def load(path):
    """Legal CE and top-1 on the 80K benchmark positions (scorers store either those or every blitz move)."""
    with np.load(path) as z:
        ce, top1 = (z["ce_legal"] if "ce_legal" in z else z["ce"]), z["top1"]
    with np.load(DATA / "games.npz") as z:
        keep = z["keep"]
    if len(ce) != len(keep):
        ce, top1 = ce[keep], top1[keep]
    return np.asarray(ce, float), np.asarray(top1, float)


def golden(name, info=None):
    p = ROOT / "lm-eval" / name / "strat-v1.json"
    if p.exists():
        cells = json.loads(p.read_text())["cells"]
        cells = {k: v["ce"] if isinstance(v, dict) else v for k, v in cells.items()}
    elif info:
        cells = info["canonical_cells"]
    else:
        return None
    out = dict(macro=np.mean(list(cells.values())),
               expert=np.mean([v for k, v in cells.items() if k.endswith(">=2400")]))  # fmt: skip
    for f in FORMATS:
        out[f] = np.mean([v for k, v in cells.items() if k.startswith(f + "/")])
    return out


def main():
    s, _ = views()
    games = s[:, 0]
    rows = {}
    for m, g in MAIA.items():
        rows[m] = dict(scores=load(DATA / "scores" / f"{m}.npz"), gflops=g, golden=None)
    for step in (132096, FINAL):
        p = BIG / f"step-{step:08d}" / "scores.npz"
        if p.exists():
            info = json.loads(p.with_suffix(".json").read_text())
            rows[f"bigrun-{step}"] = dict(
                scores=load(p), gflops=1.389, golden=golden("", info)
            )
    big = gflops("bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42")
    per_expert = 2 * 23 * 3 * 1536 * 192 / 1e9  # one routed expert in the 23 MoE layers, width 1536, expert width 192
    for p in sorted(SCORES.glob("*.npz")):
        info = json.loads(p.with_suffix(".json").read_text())
        n = p.stem
        m, k = re.match(r"kd-s(\d+)-", n), re.search(r"-k(\d+)(-|$)", n)
        if m:
            cost = gflops(STUDENT[m[1]])
        elif (ROOT / "pretrain" / n / "config.json").exists():
            cost = gflops(n)  # the run's own arch (moe_keep included)
        else:  # no config.json: Allie 2.0 with the K experts in its name
            cost = big - (16 - int(k[1])) * per_expert if k else big
        rows[n] = dict(scores=load(p), gflops=cost, golden=golden(n, info))
    for p in sorted((G / "distill-v2/scores-topk").glob("*-k*.npz")):
        k = int(p.stem.split("-k")[1].split("-")[0])
        if k < 16:
            info = json.loads(p.with_suffix(".json").read_text())
            rows[p.stem] = dict(scores=load(p), gflops=big - (16 - k) * per_expert, golden=golden("", info))
    for w, run in STUDENT.items():
        p = DATA / "scores/sw-moe-s42.npz" if w == "896" else G / f"distill-v2/scores-base/{run}.npz"
        if p.exists():
            rows[f"base-s{w}"] = dict(scores=load(p), gflops=gflops(run), golden=golden(run))
    rng = lambda: np.random.default_rng(7)
    out = {}
    for n, r in rows.items():
        ce, top1 = r["scores"]
        assert len(ce) == len(s), (n, len(ce))
        res = dict(
            gflops=r["gflops"], ce=ce.mean(), acc=100 * top1.mean(), golden=r["golden"]
        )
        refs = [x for x in ("maia3-23m", "maia3-79m") if x != n]
        if m := re.match(r"kd-s(\d+)-", n):
            refs.append(f"base-s{m[1]}")
        control = re.sub(r"-T[0-9.]+|-g[a-z0-9]+", "", re.sub(r"-a\d+-", "-a0-", n))
        if n.startswith("kd-") and control in rows and control != n:
            refs.append(control)
        if n.startswith(("ann-", "final-k")) and f"bigrun-{FINAL}" in rows:
            refs.append(f"bigrun-{FINAL}")
        if n.startswith(("ann2-k", "ann-all-on2-")) and "ann-all-p05-t1907" in rows:
            refs.append("ann-all-p05-t1907")  # the annealed model they start from
        for ref in refs:
            c2, t2 = rows[ref]["scores"]
            dc, da = (
                stats(games, ce - c2, rng()),
                stats(games, 100 * (top1 - t2), rng()),
            )
            res[f"vs {ref}"] = dict(
                dce=[dc["mean"], dc["lo"], dc["hi"]],
                dacc=[da["mean"], da["lo"], da["hi"]],
            )
            g1, g2 = r["golden"], rows[ref]["golden"]
            if g1 and g2:
                res[f"vs {ref}"]["golden"] = {k: g1[k] - g2[k] for k in g1}
        out[n] = res
    (HERE / "report.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    iv = lambda x, f: f"{x[0]:+{f}} [{x[1]:+{f}}, {x[2]:+{f}}]"
    lines = ["| model | GFLOPs | CE | acc % | golden macro | expert | " + " | ".join(FORMATS) + " |",
             "|---|---:|---:|---:|---:|---:|" + "---:|" * len(FORMATS)]  # fmt: skip
    for n, r in out.items():
        g = r["golden"]
        gl = [f"{g[k]:.4f}" for k in ("macro", "expert", *FORMATS)] if g else ["-"] * 6
        lines.append(
            f"| {n} | {r['gflops']:.2f} | {r['ce']:.4f} | {r['acc']:.2f} | "
            + " | ".join(gl)
            + " |"
        )
    lines += ["", "| model | vs | dCE | dAcc pp | d golden macro | d expert | " + " | ".join(FORMATS) + " |",
              "|---|---|---:|---:|---:|---:|" + "---:|" * len(FORMATS)]  # fmt: skip
    for n, r in out.items():
        for k, v in r.items():
            if k.startswith("vs "):
                g = v.get("golden")
                gl = (
                    [f"{g[x]:+.4f}" for x in ("macro", "expert", *FORMATS)]
                    if g
                    else ["-"] * 6
                )
                lines.append(
                    f"| {n} | {k[3:]} | {iv(v['dce'], '.4f')} | {iv(v['dacc'], '.2f')} | "
                    + " | ".join(gl)
                    + " |"
                )
    (HERE / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(out)


def plot(out):
    """Cost vs Maia-bench CE and vs top-1, one panel each: Maia-3, Allie 2.0 (raw / annealed), students."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = {"Maia-3": "#2a78d6", "big run": "#eb6834", "students": "#1baf7a"}
    hidden = ("-renorm", "ann2-k", "ann-all-on2-", "ann-all-p2-", "bigrun-132096", "-k1-")
    shown = lambda n: not any(h in n for h in hidden)
    kind = lambda n: "Maia-3" if n.startswith("maia") else "big run" if n.startswith(("bigrun", "ann", "final-k")) else "students"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#fcfcfb")
    for ax, key, label in zip(axes, ("ce", "acc"), ("Maia-bench CE, macro (nats, lower is better)",
                                                    "top-1 accuracy % (higher is better)")):  # fmt: skip
        ax.set_facecolor("#fcfcfb")
        for k, color in kinds.items():
            pts = [(r["gflops"], r[key]) for n, r in out.items() if kind(n) == k and shown(n)]
            if pts:
                ax.scatter(*zip(*pts), s=40, color=color, edgecolor="#fcfcfb", linewidth=1.5, zorder=3, label=k)
        trunc = sorted((r["gflops"], r[key]) for n, r in out.items() if (n.startswith("final-k") and n.endswith("-trunc") and shown(n)) or n == f"bigrun-{FINAL}")
        if trunc:
            ax.plot(*zip(*trunc), color=kinds["big run"], linewidth=1, alpha=0.6, zorder=2)
        best = min((r["ce"], n) for n, r in out.items() if kind(n) == "students")[1]
        for n, r in out.items():
            name = (
                n.replace("maia3-", "Maia-3 ").upper().replace("MAIA-3", "Maia-3") if n.startswith("maia")
                else "big run, 16 experts" if n == f"bigrun-{FINAL}"
                else f"K={n.split('-k')[1].split('-')[0]}" if n.startswith("final-k") and n.endswith("-trunc") and "-k12-" not in n and shown(n)
                else f"K={n.split('-k')[1].split('-')[0]}, KD" if n.startswith("ann-all-p05-t954-k")
                else "annealed" if n == "ann-all-p05-t1907"
                else "best student" if n == best
                else None
            )  # fmt: skip
            if name:
                ax.annotate(name, (r["gflops"], r[key]), fontsize=7, color="#52514e", xytext=(5, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("GFLOPs per move (log)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "pareto.png", dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
