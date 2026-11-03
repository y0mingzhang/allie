"""Every static figure of the README and DETAILS, drawn from data.json beside this script.

usage: python docs/explainers/figures/make_figures.py [--collect] [name ...]

--collect rebuilds data.json from the results under ALLIE_DATA and needs the allie package on the path. Drawing needs
only data.json, matplotlib and docs/explainers/fonts. Text that clips, overlaps or falls below 13 pt warns.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S  # noqa: E402

BUDGETS = "1e17", "3e17", "1e18"
FORMATS = "bullet", "blitz", "rapid", "classical"
BANDS = "<1400", "1400-2000", "2000-2400", ">=2400"
RATINGS = range(800, 2601, 200)
MAIA = {"maia3-5m": "Maia-3 5M", "maia3-23m": "Maia-3 23M", "maia3-79m": "Maia-3 79M"}
ANN, V21 = "ann-all-p05-t1907", "allie-2.1"  # the released Allie 2.0's and Allie 2.1's run names
BIG = "bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42"  # Allie 2.0's big run


def jload(p):
    return json.loads(Path(p).read_text())


def triple(d):
    return [d[k] for k in ("mean", "lo", "hi")]


def collect():
    from scipy.optimize import minimize_scalar

    from allie import paths
    from allie.eval.maia3 import aggregate
    from allie.model import arch

    DATA = paths.DATA
    R = next(p for p in (paths.ROOT / "results", DATA / "results") if p.exists())
    X, ORIGINAL, SEARCH = R / "recipe10x", DATA / "original-allie", DATA / "maia3-bench/search"
    CALIB, BASE = DATA / "release-2.1", X / "elo-strength/calib/klcal/baselines/sept"

    def strat(name, split="strat-v1"):
        return jload(R / f"lm-eval/{name}/{split}.json")

    def trajectory(tag):
        rows = map(json.loads, (X / f"maia3-bench/bigrun-trajectory-{tag}.jsonl").read_text().splitlines())
        return sorted(rows, key=lambda r: r["step"])

    def run(cfg, tokens, last, name):
        """Active parameters as the sweep readout counts N, tokens, FLOPs and main-test loss."""
        L, d, a = cfg["layers"], cfg["width"], cfg["arch"]
        e, k, routed, shared = arch.moe_dims(d, a)[:4]
        dense = arch.resolve(a).get("moe_dense_first", 1)
        n = (
            4 * L * d * d
            + dense * 3 * d * arch.swiglu_hidden(d)
            + (L - dense) * (3 * d * (shared + k * routed) + d * e)
        )
        return dict(n=n, d=tokens, c=last["useful_training_flops"], loss=strat(name)["macro"])

    r20, r21 = trajectory("v2"), trajectory("v4")
    t20, t21 = ([[r["tokens"] / 1e9, r["golden_macro"]] for r in rs if r.get("golden_macro")] for rs in (r20, r21))
    big_cfg = jload(R / f"pretrain/{BIG}/resume-config.json")["config"]
    runs = dict(
        big=run(big_cfg, big_cfg["scheduled_steps"] * 524288, r20[-1], BIG),
        v21=run(jload(paths.ROOT / "configs/allie-2.1.json")["trainer"], r21[-1]["tokens"], r21[-1], V21),
        v20=dict(loss=strat(ANN)["macro"], t=t20[-1][0] + jload(R / f"pretrain/{ANN}/done.json")["tokens_processed"] / 1e9),
    )  # fmt: skip
    runs["v21"]["total"] = strat(V21)["parameters"]

    sw = jload(X / "sweep-readout-c8s200f0v4-w4.json")
    m = next(x for x in sw["metrics"] if x["metric"] == "macro")
    theta = m["law"]["s16"]["theta"]
    E, A, B = math.exp(theta[0]), math.exp(theta[1]), math.exp(theta[3])
    ks = sorted((r["n"], r["c"] / (r["n"] * r["d"])) for r in sw["runs"])

    def law_at(n, d):
        return E + A * (n / 1e7) ** -theta[2] + B * (d / 1e8) ** -theta[4]

    def tokens_at(c, n):
        return c / (float(np.interp(math.log(n), np.log([x for x, _ in ks]), [k for _, k in ks])) * n)

    def optimum(c):
        f = lambda x: law_at(math.exp(x), tokens_at(c, math.exp(x)))  # noqa: E731
        n = math.exp(minimize_scalar(f, bounds=(math.log(1e5), math.log(c / 1e5)), method="bounded").x)
        return [c, law_at(n, tokens_at(c, n)), n, tokens_at(c, n)]

    sweep = {}
    for fam, b in ((f, b) for f in ("dense", "s16") for b in BUDGETS):
        cell = m["cells"][fam][b]
        rs = [r for r in sw["runs"] if r["fam"] == fam and r["budget"] == b and r["shape"] in cell["fit"]]
        n = np.array([r["n"] for r in rs], float)
        sweep.setdefault(fam, {})[b] = dict(
            c=cell["c"], fit=list(np.polyfit(np.log(n), [r["y"]["macro"] for r in rs], 2)), span=[n.min(), n.max()],
            best=[cell["n_opt"], cell["l"]], n_ci=cell["n_ci"], l_ci=cell["l_ci"],
            d=[tokens_at(cell["c"], x) for x in (cell["n_ci"][0], cell["n_opt"], cell["n_ci"][1])],
            points=[[s, np.mean([r["y"]["macro"] for r in rs if r["n"] == s])] for s in sorted({r["n"] for r in rs})],
        )  # fmt: skip
    big = runs["big"]
    ci = jload(HERE / "law_ci.json")
    law = dict(
        floor=E, floor_ci=m["law"]["s16"]["ident"]["range90"][0], forecast=law_at(big["n"], big["d"]),
        forecast_ci=ci["f20"], opt_n_ci=ci["Nopt_c20"], opt_d_ci=ci["Dopt_c20"],
        at_big=optimum(big["c"]), last=sweep["s16"]["1e18"]["c"], kappa_big=big["c"] / (big["n"] * big["d"]),
        path=[optimum(float(c)) for c in np.geomspace(4.5e17, runs["v21"]["c"], 60)],
    )  # fmt: skip
    cm = {b: [m["cm"]["s16"][b]["separate"]["cm"], *m["cm"]["s16"][b]["separate"]["noise"]] for b in BUDGETS}
    speed = {}
    for r in sw["runs"]:
        speed.setdefault((r["budget"], r["shape"], r["gpu"]), {}).setdefault(r["fam"], []).append(
            r["tps"] * r["c"] / r["d"]
        )
    flops_rate = dict(
        _src="Mixture of experts' FLOPs per second over dense's, same budget, shape and GPU, seeds averaged: steady tokens per second times counted FLOPs per token, from the sweep readout. Its three 1.8e18 pairs are resumed runs, whose earlier GPUs the readout doesn't verify.",
        pairs={"/".join(k): np.mean(v["s16"]) / np.mean(v["dense"]) for k, v in speed.items() if len(v) == 2},
    )  # fmt: skip

    with np.load(R / f"lm-eval/{V21}/strat-v2-rows.npz") as a, np.load(R / f"lm-eval/{ANN}/strat-v2-rows.npz") as b:
        assert np.array_equal(a["counts"], b["counts"])
        count, gain = a["counts"], b["nll_sums"] - a["nll_sums"]
    rows = np.random.default_rng(7).integers(0, len(gain), (2000, len(gain)))
    drop = lambda split: {k: v - strat(V21, split)["cells"][k] for k, v in strat(ANN, split)["cells"].items()}  # noqa: E731
    sept_ci = np.percentile(gain[rows].sum(1) / count[rows].sum(1), [2.5, 97.5], 0).T.tolist()
    cells = dict(july=drop("strat-v1"), sept=drop("strat-v2"), sept_ci=sept_ci)

    with np.load(DATA / "maia3-bench/games.npz") as z:
        keep = z["keep"]
        s = z["sel"][keep]
    reports = {k: jload(SEARCH / f"report-{k}.json") for k in ("allie-2.0", V21)}
    legal = {k: [np.load(f) for f in sorted((SEARCH / k / "legal").glob("[0-9]*.npz"))] for k in reports}

    def boot(ix, d):
        return [d.mean(), *np.percentile(aggregate.boot(s[ix, 0], d, np.random.default_rng(7)), [2.5, 97.5])]

    def search(tag):
        rep, zs = reports[tag], legal[tag]
        ix, raw = (np.concatenate([x[k] for x in zs])[: rep["n"]] for k in ("index", "ce"))
        q = rep["points"]
        return dict(
            raw=[q["frozen/legal"]["gflops"], q["frozen/legal"]["ce"]], gap_79m=q["frozen/legal"]["d_ce"]["maia3-79m"],
            curve=[[q[f"devcal/{b}"]["gflops"], q[f"devcal/{b}"]["ce"]] for b in (2, 5, 10, 25, 50, 100)],
            gain=boot(ix, raw - np.load(SEARCH / tag / "devcal-100.npz")["ce"][: rep["n"]]),
        )  # fmt: skip

    def scores(path):
        with np.load(path) as z:
            ce, top1 = (z["ce_legal"], z["top1"]) if "ce_legal" in z else (z["ce"], z["top1"])
        return (ce, top1) if len(ce) == len(keep) else (ce[keep], top1[keep])

    named = {name: DATA / f"maia3-bench/scores/{k}.npz" for k, name in MAIA.items()}
    named["Original Allie"] = ORIGINAL / "bench/allie1-medium.npz"
    named["Allie 2.0"] = DATA / f"distill-v2/scores/{ANN}.npz"
    named["Allie 2.1"] = DATA / "release-2.1/maia3/blitz.npz"
    named = {k: scores(v) for k, v in named.items()}
    res = aggregate.compare(named)["maia_protocol"]
    models = {k: dict(ce=triple(res[k]["d_ce"]), top=triple(res[k]["d_top1"])) for k in named if k != "Maia-3 79M"}
    agg = jload(X / f"maia3-bench/aggregate-{V21}.json")
    q = agg["views"]["maia_protocol"]["paired"]["maia3-79m"]  # the draws DETAILS quotes for Allie 2.1
    models["Allie 2.1"] = dict(ce=triple(q["d_ce"]), top=[100 * v for v in triple(q["d_acc"])])
    protocol = dict(positions=res["positions"], models=models)

    p, n = reports["allie-2.0"]["points"], reports["allie-2.0"]["n"]
    ix = np.concatenate([x["index"] for x in legal["allie-2.0"]])[:n]
    orig = named["Original Allie"][0][ix]
    flops = jload(ORIGINAL / "flops-medium.json")["flops_per_move_incremental"] / 1e9
    pareto = dict(
        n=n, v20=search("allie-2.0"), v21=search(V21),
        maia={v: [p[f"maia/{k}"]["gflops"], p[f"maia/{k}"]["ce"], p[f"maia/{k}"]["d_ce"]["maia3-79m"]] for k, v in MAIA.items()},
        original=[flops, orig.mean(), boot(ix, np.asarray(orig - named["Maia-3 79M"][0][ix], float))],
    )  # fmt: skip

    sl = agg["slices"]
    versatility = dict(
        format={f: triple(sl["format"][f]["d_ce_maia3-79m"]) for f in FORMATS},
        clock={k: triple(v["d_ce_maia3-79m"]) for k, v in sl["clock"].items()},
    )

    new, old = (jload(X / f"bigrun-progress/acc-by-game-rating-{k}.json") for k in (V21, ANN))
    old = {r["bin"]: r for r in old}
    with np.load(DATA / "maia3-bench/rating/games.npz") as z:
        rsel, meta = z["sel"][z["keep"]], z["meta"]
    man = jload(DATA / "strat-eval-v1/manifest.json")
    weight = (np.array(man["population_moves"]) / np.array(man["scored_moves"]))[rsel[:, 2]]
    game_rating = meta[rsel[:, 0], 2:4].mean(1)
    with np.load(ORIGINAL / "rating/allie1-medium.npz") as z:
        oce = np.asarray(z["ce_legal"], float)
    rating = []
    for r in filter(lambda r: r["n"], new):
        lo, hi = map(float, r["bin"].split("-"))
        k, q = (game_rating >= lo) & (game_rating < hi), r["models"]
        rating.append(dict(
            x=(lo + hi) / 2, n=r["n"], games=r["games"], v21=q[V21]["d_ce_79m"], v20=old[r["bin"]]["models"][ANN]["d_ce_79m"],
            m5=q["maia3-5m"]["d_ce_79m"], m23=q["maia3-23m"]["d_ce_79m"],
            original=np.average(oce[k], weights=weight[k]) - q["maia3-79m"]["ce"][0],
        ))  # fmt: skip

    plot = jload(CALIB / "calib-fit/win21-sept.plot.json")["cells"]
    tests = next(iter(jload(CALIB / "calib-fit/win21-sept.json")["scores"].values()))
    video = jload(CALIB / "video-refresh/cells.json")["sept"]["cells"]
    maia = jload(BASE / "maia3-79m.json")["cells"]
    with np.load(BASE / "prep-all.npz") as z:
        off, mm, H, fmt, bn, game = z["loff"], z["mm"], z["H"], z["f"], z["bin"], z["game"]
    pm = np.load(BASE / "maia3-79m.npz")["p"].astype(float)
    sampled = np.stack([np.add.reduceat(pm * mm[:, j], off[:-1]) for j in range(2)], 1)
    top = mm[[off[i] + pm[off[i] : off[i + 1]].argmax() for i in range(len(H))]]

    def paired(x_, sel):
        """calib_cells.score's test of a model against humans, clustered by game: z (positive plays stronger) and SE
        in points, for accuracy, then blunder rate."""
        x, h0 = x_[sel] - H[sel], H[sel].mean(0)
        inv = np.unique(game[sel], return_inverse=True)[1]
        c = np.zeros((inv.max() + 1, 2))
        np.add.at(c, inv, x - x.mean(0))
        se, gm = np.sqrt((c**2).sum(0)) / len(x), x.mean(0)
        return [gm[0] / se[0], se[0], -math.log(max(h0[1] + gm[1], 1e-9) / h0[1]) / (se[1] / h0[1]), 100 * se[1]]

    def unelo(v, ga, gb, h0):
        """The same from calib_cells.score's Elo gaps and their SEs; gb and h0 are fractions."""
        lb = math.log(max(h0 + gb, 1e-9) / h0)
        return [v[0] / v[1], abs(ga * v[1] / v[0]), v[2] / v[3], abs(100 * lb * v[3] / v[2] * h0)]

    calib = {k: {} for k in ("calibrated", "raw", "sampled", "argmax")}
    for i, f in enumerate(FORMATS):
        for b in RATINGS:
            key, sel = f"{f}/{b}", (fmt == i) & (bn == b)
            h = plot[key]["human"]
            assert abs(maia[key]["human"]["accuracy"] - h["accuracy"]) < 1e-6
            for name, c, z_ in (
                ("calibrated", plot[key]["calibrated"], tests["cells"][key]), ("raw", plot[key]["raw"], video[key]["raw"]),
                ("sampled", maia[key]["sampled"], paired(sampled, sel)), ("argmax", maia[key]["argmax"], paired(top, sel)),
            ):  # fmt: skip
                ga, gb = c["accuracy"] - h["accuracy"], c["blunder"] - h["blunder"]
                t = z_ if name in ("sampled", "argmax") else unelo(z_, ga, gb, h["blunder"])
                calib[name][key] = dict(acc=ga, blunder=-100 * gb, z_acc=t[0], acc_se=t[1], z_blunder=t[2], blunder_se=t[3])  # fmt: skip
    outside = {
        k: int(sum(abs(c[z]) > 1.96 for c in v.values() for z in ("z_acc", "z_blunder"))) for k, v in calib.items()
    }
    assert outside["calibrated"] == tests["tests_outside"]

    data = dict(
        runs=runs, sweep=sweep, cm=cm, flops_rate=flops_rate, law=law, training=dict(v20=t20, v21=t21), cells=cells, pareto=pareto,
        protocol=protocol, versatility=versatility, rating=rating,
        calibrated=dict(positions=sum(c["n"] for c in plot.values()), outside=outside, models=calib),
    )  # fmt: skip
    six = lambda s: float(f"{float(s):.6g}")  # noqa: E731
    data = json.loads(json.dumps(data, default=float), parse_float=six)
    (HERE / "data.json").write_text(json.dumps(data, indent=1) + "\n")


W, DPI, MARGIN = 7.2, 200, 0.24
TITLE, SUB, LABEL, TICK = 22, 14.5, 14, 13.5
A21, A20, MAIA3, ORIG = S.ALLIE21, S.ALLIE20, S.MAIA3, S.ORIGINAL
MOE = S.SEQ[3]  # brass one step lighter: BRASS against DENSE misses the validator's normal-vision floor (12.7 of 15)
RAW = S.SEQ[4]  # the raw model is still Allie 2.1: its brass family, with shape telling how it plays
SUPER = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def at(fig, x, y, s, **kw):
    """Figure text at a position in inches."""
    return fig.text(x / W, y / fig.get_figheight(), s, **kw)


def head(title, sub, h):
    """A figure of height h inches with its title and subtitle; returns it and the height left below them."""
    fig = plt.figure(figsize=(W, h))
    slab = dict(family=S.SLAB, weight="semibold", size=TITLE, linespacing=1.1, fontfeatures=["lnum"])
    at(fig, MARGIN, h - 0.2, title, va="top", color=S.INK, **slab)
    y = h - 0.2 - 0.37 * (title.count("\n") + 1) - 0.08
    at(fig, MARGIN, y, sub, size=SUB, color=S.INK2, va="top", linespacing=1.28)
    return fig, y - 0.262 * (sub.count("\n") + 1)


def place(fig, x, y, w, hh):
    """Axes at a position in inches."""
    fh = fig.get_figheight()
    ax = fig.add_axes((x / W, y / fh, w / W, hh / fh))
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, labelsize=TICK, pad=6)
    return ax


def canvas(title, sub, h, ylabel, xlabel, left=0.68, right=0.3, bottom=0.72):
    """One set of axes under the header, the y caption set horizontally above them."""
    fig, y = head(title, sub, h)
    at(fig, MARGIN, y - 0.14, ylabel, size=TICK, color=S.INK2, va="top")
    ax = place(fig, left, bottom, W - left - right, y - 0.46 - bottom)
    ax.set_xlabel(xlabel, size=TICK, labelpad=6)
    ax.xaxis.grid(False)
    return fig, ax


def say(ax, x, y, s, dx=0, dy=0, ha="left", color=S.INK, size=LABEL, weight="medium"):
    """Text at a data point, offset in points."""
    off = dict(xytext=(dx, dy), textcoords="offset points")
    ax.annotate(s, (x, y), ha=ha, va="center", color=color, size=size, weight=weight, linespacing=1.2, **off)


def dot(ax, x, y, color, size=9.5, hollow=False, z=5, ring=S.GROUND):
    face, edge = (S.GROUND, color) if hollow else (color, ring)
    ax.plot(x, y, "o", ms=size, mfc=face, mec=edge, mew=2.2 if hollow else 1.4, zorder=z)


def xticks(ax, xs, labels=None):
    ax.set_xticks(xs, labels or [f"{x:g}" for x in xs])
    ax.minorticks_off()


def yticks(ax, ys):
    ax.set_yticks(ys, [f"{v:.2f}" for v in ys])


def power(v):
    e = math.floor(math.log10(v))
    return f"{v / 10**e:.1f}×10{str(e).translate(SUPER)}"


def signed(v, nd=3):
    return f"{v:+.{nd}f}".replace("-", "−")


def save(fig, name):
    """Warn about text outside the canvas, below 13 pt or over other text, then write the PNG."""
    fig.canvas.draw()
    r, box = fig.canvas.get_renderer(), fig.bbox
    items = [(t, Text.get_window_extent(t, r)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    for i, (t, b) in enumerate(items):
        if b.x0 < 0 or b.y0 < 0 or b.x1 > box.x1 or b.y1 > box.y1:
            print(f"  {name}: clipped {t.get_text()!r}")
        if t.get_fontsize() < 13:
            print(f"  {name}: small {t.get_text()!r}")
        for u, c in items[:i]:
            if b.overlaps(c):
                print(f"  {name}: overlap {t.get_text()!r} / {u.get_text()!r}")
    fig.savefig(HERE / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {name}.png")


def model(d):
    """The whole model, then one routed block with 16 of its 256 experts lit."""
    act, total = d["runs"]["v21"]["n"] / 1e9, d["runs"]["v21"]["total"] / 1e9
    h = 6.72
    fig, _ = head(f"Each move uses only {act:.1f}B of {total:.1f}B parameters",
                  "21 of Allie 2.1's 24 blocks send each move to 16 of 256\nsmall experts, plus one shared expert.", h)  # fmt: skip
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(0, h))
    ax.axis("off")
    arrow = dict(arrowstyle="-|>", color=S.INK2, lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=14)

    def rect(x, y, w, hh, fc=S.PANEL, z=2):
        ax.add_patch(Rectangle((x, y), w, hh, fc=fc, ec="none", zorder=z, antialiased=False))

    def wire(xs, ys, head=True):
        ax.plot(xs, ys, color=S.INK2, lw=1.8, zorder=3, solid_capstyle="butt", solid_joinstyle="miter")
        if head:
            ax.annotate("", (xs[-1], ys[-1]), (xs[-2], ys[-2]), arrowprops=arrow)

    def label(x, y, s, ha="center", color=S.INK, size=LABEL, weight="medium"):
        ax.text(x, y, s, ha=ha, va="center", color=color, size=size, weight=weight, linespacing=1.15, zorder=6)

    top, box, row = 5.14, 0.34, 0.4
    ys = [top - box / 2 - i * row for i in range(3)]
    mid, bottom = ys[1], top - box - 2 * row
    bx, x0, bw, gap, zoom = MARGIN + 1.5, 1.92, 0.1, 0.032, 12
    x1 = x0 + 24 * (bw + gap) - gap
    xo = x1 + 0.38
    for y, s in zip(ys, ("Game + ratings", "Clock", "Board")):
        rect(MARGIN, y - box / 2, 1.42, box)
        label(MARGIN + 0.71, y, s)
        wire([MARGIN + 1.42, bx], [y, y], head=False)
    wire([bx, bx], [ys[0], ys[2]], head=False)
    wire([bx, x0 - 0.02], [mid, mid])
    for i in range(24):
        rect(x0 + i * (bw + gap), bottom, bw, top - bottom, fc=S.DENSE if i < 3 else S.INK if i == zoom else A21)
    label(x0, top + 0.16, "3 dense", ha="left", size=TICK, color=S.INK2)
    label(x1, top + 0.16, "21 routed blocks", ha="right", size=TICK, color=S.INK2)
    wire([x1 + 0.03, xo - 0.17], [mid, mid], head=False)
    wire([xo - 0.17, xo - 0.17], [ys[0], ys[2]], head=False)
    for i, (y, s) in enumerate(zip(ys, ("Next move", "Think time", "Game result"))):
        wire([xo - 0.17, xo - 0.01], [y, y])
        rect(xo, y - box / 2, W - MARGIN - xo, box)
        label((xo + W - MARGIN) / 2, y, s, weight="medium" if i else "semibold", color=S.INK2 if i else S.INK)

    b0, b1 = 0.86, bottom - 0.28
    zx = x0 + zoom * (bw + gap)
    for xa, xb in ((zx, MARGIN), (zx + bw, W - MARGIN)):
        ax.plot([xa, xb], [bottom, b1], color=S.RULE, lw=1.2, zorder=1)
    ax.add_patch(Rectangle((MARGIN, b0), W - 2 * MARGIN, b1 - b0, fc="none", ec=S.RULE, lw=1.2, zorder=1))
    label(MARGIN + 0.14, b1 - 0.2, "One routed block", ha="left", weight="semibold")

    cell, cgap, gx = 0.082, 0.024, 2.86
    side = 16 * cell + 15 * cgap
    sy, gtop = b1 - 0.5, b1 - 0.74
    midy = gtop - side / 2
    lit = set(np.random.default_rng(16).choice(256, 16, replace=False).tolist())
    for i in range(256):
        x, y = gx + (i % 16) * (cell + cgap), gtop - side + (i // 16) * (cell + cgap)
        rect(x, y, cell, cell, fc=S.HILITE if i in lit else S.PANEL, z=3)
    label(gx + side / 2, gtop - side - 0.17, "16 of 256 experts", weight="semibold")
    rect(gx, sy - 0.16, side, 0.32, fc=S.HILITE)
    label(gx + side / 2, sy, "Shared expert", color=S.GROUND, weight="semibold")
    rect(MARGIN + 0.14, midy - 0.21, 1.08, 0.42)
    label(MARGIN + 0.68, midy, "Attention")
    xr, xp = 1.92, 5.34
    wire([MARGIN + 1.22, xr], [midy, midy], head=False)
    ax.plot(xr, midy, "o", ms=11, color=S.INK2, zorder=5)
    label(xr, midy - 0.25, "router", size=TICK, color=S.INK2)
    wire([xr, gx - 0.05], [midy, midy])
    wire([xr, xr, gx - 0.05], [midy, sy, sy])
    wire([gx + side + 0.04, xp - 0.13], [midy, midy])
    wire([gx + side + 0.04, xp, xp], [sy, sy, midy + 0.13])
    ax.plot(xp, midy, "o", ms=19, mfc=S.GROUND, mec=S.INK2, mew=1.6, zorder=5)
    label(xp, midy, "+", color=S.INK2, weight="semibold")
    wire([xp + 0.13, xp + 0.58], [midy, midy])
    label(xp + 0.66, midy, "Next\nblock", ha="left", size=TICK, color=S.INK2)

    used = act / total * (W - 2 * MARGIN)
    rect(MARGIN, 0.3, W - 2 * MARGIN, 0.26)
    rect(MARGIN, 0.3, used, 0.26, fc=S.HILITE, z=3)
    label(MARGIN + used + 0.1, 0.43, f"{act:.1f}B used for this move, of {total:.1f}B", ha="left", size=TICK)
    save(fig, "model")


def sweep(d):
    """Isoflop curves on one axis: minima move down and right with compute, MoE under dense at each budget."""
    fig, ax = canvas(
        "Mixture of experts beats dense at every budget",
        "Loss against model size at three budgets of counted FLOPs, and the\n"
        "compute a dense model needs to match (90% intervals). Rings: best size.",
        6.2, "Loss (nats), lower is better. Axis starts at 1.28", "Active parameters",
    )  # fmt: skip
    for fam, color, lw in (("dense", S.DENSE, 2.4), ("s16", MOE, 2.8)):
        for c in d["sweep"][fam].values():
            t = np.linspace(math.log(c["span"][0]) - 0.15, math.log(c["span"][1]) + 0.15, 60)
            ax.plot(np.exp(t), np.polyval(c["fit"], t), color=color, lw=lw, zorder=3, solid_capstyle="round")
            ax.plot(*zip(*c["points"]), "o", ms=6.5, color=color, zorder=4)
            ax.plot(*c["best"], "o", ms=15, mfc="none", mec=color, mew=2, zorder=4)
    ax.set_xscale("log")
    ax.set(xlim=(1.1e7, 6.5e8), ylim=(1.283, 1.465))
    xticks(ax, [2e7, 5e7, 1e8, 2e8, 5e8], ["20M", "50M", "100M", "200M", "500M"])
    yticks(ax, [1.30, 1.35, 1.40, 1.45])
    for b, x, y in (("1e17", 1.2e7, 1.449), ("3e17", 1.2e7, 1.3245), ("1e18", 1.1e8, 1.2975)):
        v, lo, hi = d["cm"][b]
        say(ax, x, y, f"{power(d['sweep']['s16'][b]['c'])} FLOPs\ndense needs {v:.1f}× [{lo:.1f}, {hi:.1f}]")
    say(ax, 1.3e8, 1.451, "dense", dx=6)
    say(ax, 8.6e7, 1.41, "mixture of\nexperts", dx=6)
    save(fig, "sweep")


def law_line(ax, law, xi, yi, end):
    """The law's optimum up to compute end: solid over the sweep, dashed beyond."""
    path = np.array([p for p in law["path"] if p[0] <= end])
    inside = path[:, 0] <= law["last"]
    ax.plot(path[inside, xi], path[inside, yi], color=S.INK2, lw=2.6, zorder=3)
    ax.plot(path[~inside, xi], path[~inside, yi], color=S.INK2, lw=2.6, ls=(0, (4, 3)), zorder=3)


def frontier(d):
    """The law's extrapolation against where the two real runs landed."""
    fig, ax = canvas(
        "Both big runs landed below the scaling law",
        "Best loss against training compute. The law was fit on the sweep,\n"
        "before either run. Whiskers and band: 90% intervals.",
        6.3, "Loss (nats), lower is better. Axis starts at 1.20", "Training compute (counted FLOPs, not time)",
    )  # fmt: skip
    law, big, v21 = d["law"], d["runs"]["big"], d["runs"]["v21"]
    fl, (lo, hi), fc = law["floor"], law["floor_ci"], law["forecast"]
    law_line(ax, law, 0, 1, v21["c"])
    ax.axhspan(lo, hi, color=S.PANEL, lw=0, zorder=1)
    for c in d["sweep"]["s16"].values():
        ax.plot([c["c"], c["c"]], c["l_ci"], color=S.INK2, lw=1.4, zorder=4)
        dot(ax, c["c"], c["best"][1], S.INK2, size=8)
    ax.plot([big["c"], big["c"]], [big["loss"], law["forecast_ci"][1]], color=S.INK, lw=1.6, zorder=4)
    dot(ax, big["c"], fc, S.INK, hollow=True, size=11)
    dot(ax, big["c"], big["loss"], A20, size=12)
    dot(ax, v21["c"], v21["loss"], A21, size=12)
    ax.set_xscale("log")
    ax.set(xlim=(4e17, 3e21), ylim=(1.2, 1.395))
    xticks(ax, [1e18, 1e19, 1e20, 1e21], [f"10{str(e).translate(SUPER)}" for e in (18, 19, 20, 21)])
    yticks(ax, [1.20, 1.25, 1.30, 1.35])
    say(ax, 2.2e18, 1.357, "Sweep's best models, and the\nlaw fit to them (dashed beyond)", dx=10)
    say(ax, 4.6e17, fl + 0.0015, f"Law's limit at unlimited compute: {fl:.2f} [{lo:.2f}, {hi:.2f}]", size=TICK)
    say(
        ax,
        big["c"],
        fc,
        f"Forecast {fc:.4f}\n[{law['forecast_ci'][0]:.3f}, {law['forecast_ci'][1]:.3f}]",
        dy=25,
        ha="center",
    )
    s = f"Allie 2.0's big run {big['loss']:.4f}\n{fc - big['loss']:.3f} below its forecast"
    say(ax, big["c"], big["loss"], s, dx=-13, ha="right", weight="semibold")
    s = f"Allie 2.1 {v21['loss']:.4f}. It changed the recipe\ntoo, so this gap isn't only the law's miss"
    say(ax, v21["c"], v21["loss"], s, dx=-13, dy=-5, ha="right", weight="semibold")
    save(fig, "frontier")


def optimal(d):
    """Size against tokens: the law's choice and the big run sit on one line of equal compute."""
    law, big, v21 = d["law"], d["runs"]["big"], d["runs"]["v21"]
    fig, ax = canvas(
        "The law wanted a bigger model on fewer tokens",
        f"Dashed: the law's best size past the sweep, up to {big['c'] / law['last']:.0f}× its largest\n"
        "budget, so a direction, not a result. Lines: 90% intervals.",
        6.2, "Active parameters", "Training tokens",
    )  # fmt: skip
    law_line(ax, law, 3, 2, big["c"])
    for c in d["sweep"]["s16"].values():
        ax.plot([c["d"][0], c["d"][2]], c["n_ci"], color=S.INK2, lw=1.4, zorder=4)
        dot(ax, c["d"][1], c["best"][0], S.INK2, size=8)
    xs = np.array([8e9, 3e11])
    ax.plot(xs, big["c"] / (law["kappa_big"] * xs), color=S.MUTED, lw=1.4, zorder=2)
    n_opt, d_opt = law["at_big"][2:]
    ax.plot(law["opt_d_ci"][::-1], law["opt_n_ci"], color=S.INK, lw=2.2, zorder=4)
    dot(ax, d_opt, n_opt, S.INK, hollow=True, size=11)
    dot(ax, big["d"], big["n"], A20, size=12)
    dot(ax, v21["d"], v21["n"], A21, size=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1.5e9, 3e11), ylim=(2e7, 5e9))
    xticks(ax, [2e9, 5e9, 1e10, 2e10, 5e10, 1e11, 2e11], ["2B", "5B", "10B", "20B", "50B", "100B", "200B"])
    ax.set_yticks([3e7, 1e8, 3e8, 1e9, 3e9], ["30M", "100M", "300M", "1B", "3B"])
    ax.minorticks_off()
    s = f"Law's choice\n{n_opt / 1e9:.1f}B on {d_opt / 1e9:.0f}B tokens"
    leader = dict(arrowstyle="-", color=S.MUTED, lw=1, shrinkB=7)
    ax.annotate(
        s, (d_opt, n_opt), (-48, 0), "data", "offset points", ha="right", va="center", size=LABEL, arrowprops=leader
    )
    s = f"Allie 2.0's big run\n{big['n'] / 1e9:.2f}B on {big['d'] / 1e9:.0f}B"
    say(ax, big["d"], big["n"], s, dx=-7, dy=-24, ha="right", weight="semibold")
    s = f"Allie 2.1\n{v21['n'] / 1e9:.1f}B on {v21['d'] / 1e9:.0f}B\n{v21['c'] / big['c']:.1f}× the compute"
    say(ax, v21["d"], v21["n"], s, dy=37, ha="center", weight="semibold")
    say(ax, 5.8e9, 3.1e7, "Sweep's best sizes")
    say(ax, 2.8e11, 1.4e8, f"Equal compute,\n{power(big['c'])} FLOPs", ha="right", color=S.INK2)
    save(fig, "optimal")


def training(d):
    """Both runs' loss by tokens, and both releases."""
    v20, v21 = d["runs"]["v20"], d["runs"]["v21"]
    fig, ax = canvas(
        f"Allie 2.1 finished {v20['loss'] - v21['loss']:.3f} below Allie 2.0",
        "Each run's learning rate decays to near zero at its own end, so\n"
        "the crossing near 70B tokens says little.",
        5.8, "Loss (nats), lower is better. Axis starts at 1.18", "Training tokens",
    )  # fmt: skip
    t20, t21 = np.array(d["training"]["v20"]), np.array(d["training"]["v21"])
    ax.plot(*t20.T, color=A20, lw=2.6, zorder=3)
    ax.plot([t20[-1, 0], v20["t"]], [t20[-1, 1], v20["loss"]], color=A20, lw=2.6, zorder=3)
    ax.plot(*t21.T, color=A21, lw=2.8, zorder=4)
    dot(ax, v20["t"], v20["loss"], A20, size=12)
    dot(ax, t21[-1, 0], v21["loss"], A21, size=12)
    ax.set(xlim=(0, 122), ylim=(1.18, 1.425))
    xticks(ax, range(0, 121, 20), ["0"] + [f"{v}B" for v in range(20, 121, 20)])
    yticks(ax, [1.20, 1.25, 1.30, 1.35, 1.40])
    say(ax, *t21[0], "Allie 2.1", dx=9, dy=9)
    say(ax, *t20[0], "Allie 2.0's big run, scored from 22B on", dx=3, dy=9)
    s = f"Allie 2.0 {v20['loss']:.4f}\nafter a second anneal"
    say(ax, v20["t"], v20["loss"], s, dx=-2, dy=-27, ha="right", weight="semibold")
    say(ax, t21[-1, 0], v21["loss"], f"Allie 2.1 {v21['loss']:.4f}", dx=6, dy=-17, ha="right", weight="semibold")
    save(fig, "training")


def cells(d):
    """The main test's 16 cells: every one improved, more with rating and in slower time controls."""
    c = d["cells"]
    fig, y = head(
        "Allie 2.1 improved every cell of the main test",
        "Allie 2.0's loss minus Allie 2.1's, in nats. Large: July test games.\n"
        "Small: the same cells on September games.",
        5.4,
    )  # fmt: skip
    ax = place(fig, 1.12, 0.22, W - 1.12 - MARGIN, y - 0.64)
    ramp = LinearSegmentedColormap.from_list("seq", S.SEQ[2:])
    lo, hi = min(c["july"].values()), max(c["july"].values())
    ink = dict(ha="center", va="center", color=S.GROUND)
    for i, f in enumerate(FORMATS):
        for j, b in enumerate(BANDS):
            v, y0 = c["july"][f"{f}/{b}"], 3 - i
            ax.add_patch(Rectangle((j + 0.02, y0 + 0.03), 0.96, 0.94, fc=ramp((v - lo) / (hi - lo)), ec="none"))
            ax.text(j + 0.5, y0 + 0.6, f"{v:.3f}", size=17, weight="semibold", **ink)
            ax.text(j + 0.5, y0 + 0.25, f"Sept {c['sept'][f'{f}/{b}']:.3f}", size=13, **ink)
    ax.set(xlim=(0, 4), ylim=(0, 4))
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(4) + 0.5, ["Under 1400", "1400–2000", "2000–2400", "2400+"])
    ax.set_yticks(np.arange(4) + 0.5, [f.capitalize() for f in reversed(FORMATS)])
    ax.grid(False)
    ax.spines[:].set_visible(False)
    save(fig, "cells")


def pareto(d):
    """Loss against compute per move: Maia-3's sizes, the Allie releases, and what search adds."""
    p = d["pareto"]
    fig, ax = canvas(
        "Allie 2.1 beats Maia-3 79M on blitz,\nat about a quarter of the compute",
        f"{p['n']:,} blitz positions, including ones Maia-3's own test skips. Lines\n"
        "on dots: 95% interval of the gap to Maia-3 79M.",
        6.9, "Loss (nats), lower is better. Axis starts at 1.17", "Compute per move (counted GFLOPs, game cached, not time; log scale)",
    )  # fmt: skip

    def mark(x, y, gap, color, size):
        if any(gap):
            ax.plot([x, x], [y - gap[0] + gap[1], y - gap[0] + gap[2]], color=color, lw=2.2, zorder=4)
        dot(ax, x, y, color, size=size)

    ax.plot(*zip(*[v[:2] for v in p["maia"].values()]), color=MAIA3, lw=2.6, zorder=3)
    for name, (x, y, gap) in p["maia"].items():
        mark(x, y, gap, MAIA3, 10)
        say(ax, x, y, name, dx=11, dy=9)
    mark(*p["original"], ORIG, 10)
    say(ax, *p["original"][:2], "Original Allie", dx=-10, ha="right")
    for v, color, name, dy in ((p["v20"], A20, "Allie 2.0", 25), (p["v21"], A21, "Allie 2.1", -24)):
        ax.plot(*zip(v["raw"], *v["curve"]), color=color, lw=2.2, ls=(0, (4, 3)), zorder=3)
        for x, y in v["curve"]:
            dot(ax, x, y, color, hollow=True, size=6.5, z=4)
        mark(*v["raw"], v["gap_79m"], color, 12)
        say(ax, *v["raw"], name, dx=-11, ha="right", weight="semibold")
        g, lo, hi = v["gain"]
        s = f"With 100 search simulations:\n{g:.3f} lower [{lo:.3f}, {hi:.3f}]"
        say(ax, 400, v["curve"][-1][1], s, dy=dy, ha="right", size=TICK)
    ax.set_xscale("log")
    ax.set(xlim=(0.11, 420), ylim=(1.17, 1.32))
    xticks(ax, [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    yticks(ax, [1.18, 1.20, 1.22, 1.24, 1.26, 1.28, 1.30])
    save(fig, "pareto")


def forest(name, title, sub, panels, left):
    """Stacked panels of rows, each a dot and its interval, with both as text on the right. panels: (heading, rows,
    xlim, xticks, tick decimals, value decimals); rows: (label, [mean, lo, hi], colour, hollow, strong)."""
    row, right, foot = 0.335, 2.34, 0.42
    h = 1.68 + sum(0.34 + len(p[1]) * row + foot for p in panels)
    fig, y = head(title, sub, h)
    y -= 0.2
    for heading, rows, lim, xt, ntick, nval in panels:
        at(fig, MARGIN, y - 0.03, heading, size=LABEL, color=S.INK, va="top", weight="semibold")
        y -= 0.34 + len(rows) * row
        ax = place(fig, left, y, W - left - right, len(rows) * row)
        side = dict(transform=ax.get_yaxis_transform(), va="center")
        for i, (label, (v, lo, hi), color, hollow, strong) in enumerate(rows):
            yy, weight = len(rows) - 1 - i, "semibold" if strong else "medium"
            ax.plot([lo, hi], [yy, yy], color=color, lw=2.2, zorder=3, marker="|", ms=8, mew=2.2)
            dot(ax, v, yy, color, size=9, hollow=hollow, ring=color)
            ax.text(-0.012, yy, label, ha="right", size=LABEL, weight=weight, **side)
            s = f"{signed(v, nval)} [{signed(lo, nval)}, {signed(hi, nval)}]"
            ax.text(1.024, yy, s, ha="left", size=TICK, weight=weight, color=S.INK if strong else S.INK2, **side)
        ax.axvline(0, color=MAIA3, lw=2.2, zorder=2)
        ax.set(xlim=lim, ylim=(-0.55, len(rows) - 0.45))
        ax.set_xticks(xt, ["0" if x == 0 else signed(x, ntick) for x in xt])
        ax.set_yticks([])
        y -= foot
    save(fig, name)


def protocol(d):
    p = d["protocol"]
    order = (
        ("Allie 2.1", A21),
        ("Allie 2.0", A20),
        ("Maia-3 23M", MAIA3),
        ("Original Allie", ORIG),
        ("Maia-3 5M", MAIA3),
    )
    rows = lambda k: [(n, p["models"][n][k], c, False, n == "Allie 2.1") for n, c in order]  # noqa: E731
    forest(
        "protocol", "On Maia-3's own rule, Allie 2.1 still leads",
        f"Difference from Maia-3 79M on {p['positions']:,} blitz positions, scored by\n"
        "the Maia-3 paper's rule. Lines and brackets: 95% intervals.",
        (("Loss (nats). Left of zero is better than Maia-3 79M", rows("ce"), (-0.06, 0.09), [-0.04, 0, 0.04, 0.08], 2, 3),
         ("Top choice correct (points). Right of zero is better", rows("top"), (-2.7, 1.9), [-2, -1, 0, 1], 0, 2)),
        1.42,
    )  # fmt: skip


def versatility(d):
    v = d["versatility"]
    formats = [(f.capitalize(), v["format"][f], A21, f != "blitz", False) for f in FORMATS]
    clocks = ("Under 10 s", "10–30 s", "30–60 s", "1–2 min", "Over 2 min")
    clocks = [(n, x, A21, False, False) for n, x in zip(clocks, v["clock"].values())]
    lim, xt = (-0.31, 0.012), [-0.3, -0.2, -0.1, 0]
    forest(
        "versatility", "Allie 2.1 leads most in bullet and time trouble",
        "Allie 2.1's loss minus Maia-3 79M's, in nats. Left of zero, Allie 2.1\n"
        "predicts better. Lines and brackets: 95% intervals.",
        (("By time control (hollow: Maia-3 never trained on it)", formats, lim, xt, 1, 3),
         ("In blitz, by time left on the mover's clock", clocks, lim, xt, 1, 3)),
        1.12,
    )  # fmt: skip


def rating(d):
    """Every model against Maia-3 79M by game rating."""
    r = d["rating"]
    x = np.array([b["x"] for b in r])
    arr = lambda k: np.array([b[k] for b in r])  # noqa: E731
    fig, ax = canvas(
        "Allie 2.1 beats Maia-3 79M at every rating",
        "Every blitz position of the main test. Below zero beats Maia-3 79M.\n"
        "Bands: 95% intervals; the end bins hold few games.",
        5.9, "Loss minus Maia-3 79M's (nats)", "Game rating (Lichess blitz)", right=1.34,
    )  # fmt: skip
    ax.axhline(0, color=MAIA3, lw=2.4, zorder=3)
    for k, color, lw, ls, band in (("m5", MAIA3, 1.6, (0, (1, 1.6)), 0.1), ("m23", MAIA3, 1.6, "solid", 0.1),
                                   ("v20", A20, 2.2, "solid", 0.2), ("v21", A21, 3, "solid", 0.28)):  # fmt: skip
        v = arr(k)
        ax.fill_between(x, v[:, 1], v[:, 2], color=color, alpha=band, lw=0, zorder=2)
        ax.plot(x, v[:, 0], color=color, lw=lw, ls=ls, zorder=4)
    ax.plot(x, arr("original"), color=ORIG, lw=1.6, ls=(0, (4, 2.5)), zorder=4)
    ax.set(xlim=(620, 2880), ylim=(-0.145, 0.165))
    xticks(ax, range(800, 2801, 400))
    ys = (-0.10, -0.05, 0, 0.05, 0.10, 0.15)
    ax.set_yticks(ys, [signed(v, 2) if v else "0" for v in ys])
    leader = dict(arrowstyle="-", color=S.MUTED, lw=1, shrinkA=1, shrinkB=2)
    for name, yv, yl in (("Original Allie", arr("original")[-1], 0.152), ("Maia-3 5M", arr("m5")[-1, 0], 0.122),
                         ("Maia-3 23M", arr("m23")[-1, 0], 0.038), ("Maia-3 79M", 0.0, 0.0),
                         ("Allie 2.0", arr("v20")[-1, 0], -0.03), ("Allie 2.1", arr("v21")[-1, 0], -0.062)):  # fmt: skip
        weight = "semibold" if name == "Allie 2.1" else "medium"
        ax.annotate(name, (2850, yv), (2945, yl), annotation_clip=False, size=TICK, va="center", weight=weight, arrowprops=leader)  # fmt: skip
    save(fig, "rating")


def calibrated(d):
    """Four scorecards of the same 80 strength tests: one cell per time control, rating and metric."""
    c = d["calibrated"]
    h, x0, pw, gap, rh, span = 8.28, 1.0, 2.9, 0.16, 0.25, 4.0
    fig, y = head(
        "Calibrated play is closest to human strength",
        "Each cell is one test: a model against humans at that rating, on\n"
        "September 2026 games. Dot: outside the noise; about 4 of 80 by chance.",
        h,
    )  # fmt: skip
    cmap = LinearSegmentedColormap.from_list("gap", [S.FLAG, S.PANEL, S.SKY])
    panels = (("calibrated", "Allie 2.1, calibrated"), ("raw", "Allie 2.1, raw model"),
              ("sampled", "Maia-3 79M, sampled"), ("argmax", "Maia-3 79M, top move"))  # fmt: skip
    ph = 9 * rh
    y -= 0.14
    for k, (name, title) in enumerate(panels):
        left, top = x0 + (k % 2) * (pw + gap), y - 0.3 - (k // 2) * (ph + 0.58)
        ax = fig.add_axes((left / W, (top - ph) / h, pw / W, ph / h))
        ax.set(xlim=(0, 10), ylim=(0, 9))
        ax.axis("off")
        side = dict(ha="right", va="center", size=TICK)
        for g, (metric, z, group) in enumerate((("acc", "z_acc", "Accuracy"), ("blunder", "z_blunder", "Blunders"))):
            for i, f in enumerate(FORMATS):
                yy = 8 - i - g * 5
                for j, b in enumerate(RATINGS):
                    cell = c["models"][name][f"{f}/{b}"]
                    fill = cmap(0.5 + 0.5 * np.clip(cell[metric] / span, -1, 1))
                    ax.add_patch(Rectangle((j + 0.04, yy + 0.04), 0.92, 0.92, fc=fill, ec="none"))
                    if abs(cell[z]) > 1.96:
                        ax.plot(j + 0.5, yy + 0.5, "o", ms=4.6, color=S.INK, zorder=3)
                if k % 2 == 0:
                    ax.text(-0.25, yy + 0.5, f.capitalize(), color=S.INK2, **side)
            if k % 2 == 0:
                ax.text(-0.25, 9.55 - g * 5, group, weight="semibold", **side)
        ax.text(0, 9.2, title, ha="left", va="bottom", size=LABEL, weight="semibold")
        ax.text(10, 9.2, f"{c['outside'][name]} of 80", ha="right", va="bottom", size=LABEL)
        if k >= 2:
            for j, s in ((0.5, "800"), (5, "Rating"), (9.5, "2600")):
                ax.text(j, -0.2, s, ha="center", va="top", size=TICK, color=S.INK2)
    bar = fig.add_axes((x0 / W, 1.02 / h, (W - x0 - MARGIN) / W, 0.15 / h))
    bar.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=cmap)
    bar.axis("off")
    for xt, ha, s in ((x0, "left", "4+ points weaker"), ((x0 + W - MARGIN) / 2, "center", "same as humans"),
                      (W - MARGIN, "right", "4+ points stronger")):  # fmt: skip
        at(fig, xt, 0.96, s, ha=ha, va="top", size=TICK, color=S.INK2)
    s = "Stockfish scores accuracy and blunders. Maia-3 trains only on\nblitz and never sees the clock."
    at(fig, MARGIN, 0.62, s, size=TICK, color=S.INK2, va="top", linespacing=1.25)
    save(fig, "calibrated")


def calibrated_gaps(d):
    """Every strength test with its interval, for DETAILS."""
    c = d["calibrated"]["models"]
    series = (("argmax", MAIA3, "^", "Maia-3 79M, top move"), ("sampled", MAIA3, "v", "Maia-3 79M, sampled"),
              ("raw", RAW, "s", "Allie 2.1, raw model"), ("calibrated", A21, "o", "Allie 2.1, calibrated"))  # fmt: skip
    x0, pw, gap, ph = 0.7, 3.0, 0.26, 1.62
    fig, y = head(
        "Each model's distance from human strength",
        "Model minus humans, in points, on September 2026 games. Above\n"
        "zero plays stronger. Lines: 95% intervals, paired with the humans.",
        9.8,
    )  # fmt: skip
    h = fig.get_figheight()
    y -= 0.12
    for i, (_, color, m, label) in enumerate(reversed(series)):
        xl, yl = MARGIN + 0.16 + (i % 2) * 3.4, y - 0.13 - (i // 2) * 0.3
        fig.add_artist(plt.Line2D([xl / W], [yl / h], marker=m, ms=9, color=color, ls="none"))
        at(fig, xl + 0.16, yl, label, size=TICK, va="center")
    y -= 0.86
    ys = (-4, -2, 0, 2, 4, 6)
    for g, (metric, se, group) in enumerate((("acc", "acc_se", "Move accuracy"), ("blunder", "blunder_se", "Fewer blunders"))):  # fmt: skip
        at(fig, x0 + g * (pw + gap), y + 0.02, group, size=LABEL, weight="semibold", va="bottom")
        for r, f in enumerate(FORMATS):
            ax = place(fig, x0 + g * (pw + gap), y - 0.08 - (r + 1) * ph - r * 0.12, pw, ph)
            ax.axhline(0, color=S.INK2, lw=1.6, zorder=2)
            for k, (name, color, m, _) in enumerate(series):
                xs = np.array(RATINGS) + (k - 1.5) * 26
                v = np.array([c[name][f"{f}/{b}"][metric] for b in RATINGS])
                e = 1.96 * np.array([c[name][f"{f}/{b}"][se] for b in RATINGS])
                ax.vlines(xs, v - e, v + e, color=color, lw=1.4, zorder=3 + k)
                ax.plot(xs, v, m, color=color, ms=5.6 if name == "calibrated" else 4.6, zorder=4 + k, ls="none")
            ax.set(xlim=(690, 2710), ylim=(-4.6, 8.4))
            ax.set_xticks(range(800, 2401, 400), [str(b) if r == 3 else "" for b in range(800, 2401, 400)])
            ax.set_yticks(ys, [("0" if v == 0 else signed(v, 0)) if g == 0 else "" for v in ys])
            ax.xaxis.grid(False)
            ax.text(0.02, 0.98, f.capitalize(), transform=ax.transAxes, va="top", size=TICK)
    save(fig, "calibrated-gaps")


FIGS = {f.__name__.replace("_", "-"): f for f in (model, sweep, frontier, optimal, training, cells, pareto, protocol,
                                                   versatility, rating, calibrated, calibrated_gaps)}  # fmt: skip

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("names", nargs="*")
    a = p.parse_args()
    if a.collect:
        collect()
    plt.switch_backend("agg")
    S.use_mpl()
    data = jload(HERE / "data.json")
    for name in a.names or FIGS:
        FIGS[name](data)
