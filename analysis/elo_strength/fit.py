"""Ratings from the benchmark's games: Elo logistic maximum likelihood with Stockfish anchored.

python fit.py DIR [--boot 1000] [--out ratings.json] [--figure]

Anchored: every sf-E fixed at E (Ordo style). Ladder: from Stockfish's own games, its levels free
except for their mean, which checks the anchors' spacing. Joint: all games, Stockfish spacing free
(mean pinned). 95% intervals: bootstrap over opening pairs within each
pairing. Writes DIR/ratings.json (or --out) and prints the tables RESULTS.md quotes.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

K = math.log(10) / 400
SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def load(d):
    return [
        json.loads(line)
        for f in sorted(Path(d, "games").glob("*.jsonl"))
        for line in open(f)
        if line.strip()
    ]


def nominal(p):
    m = re.fullmatch(r"sf-(\d+)", p)
    return int(m[1]) if m else None


def tables(games):
    """players, and per pairing (i, j) the per-opening (games, i's points) clusters."""
    players = sorted({g[c] for g in games for c in ("white", "black")})
    idx = {p: k for k, p in enumerate(players)}
    pairs = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for g in games:
        a, b = sorted((g["white"], g["black"]))
        s = SCORE[g["result"]] if g["white"] == a else 1 - SCORE[g["result"]]
        c = pairs[idx[a], idx[b]][g["opening"]]
        c[0] += 1
        c[1] += s
    return players, {k: np.array(list(v.values())) for k, v in pairs.items()}


def fit(players, totals, fixed):
    """ML ratings; totals: (i, j, games, i's points); fixed: {index: rating}."""
    i, j, n, s = (np.array(x) for x in zip(*totals))
    free = [k for k in range(len(players)) if k not in fixed]
    r0 = np.full(len(players), np.mean(list(fixed.values())))
    for k, v in fixed.items():
        r0[k] = v

    def nll(x):
        r = r0.copy()
        r[free] = x
        p = 1 / (1 + np.exp(-K * (r[i] - r[j])))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        g = K * (s - n * p)  # d loglik / d r_i
        grad = np.zeros(len(players))
        np.add.at(grad, i, -g)
        np.add.at(grad, j, g)
        return -(s * np.log(p) + (n - s) * np.log(1 - p)).sum(), grad[free]

    x = minimize(
        nll, r0[free], jac=True, method="L-BFGS-B", bounds=[(-1500, 5000)] * len(free)
    ).x
    r = r0.copy()
    r[free] = x
    return r


def solve(players, pairs, mode, rng=None):
    totals = []
    for (a, b), c in pairs.items():
        if rng is not None:
            c = c[rng.integers(len(c), size=len(c))]
        totals.append((a, b, c[:, 0].sum(), c[:, 1].sum()))
    sf = {k: nominal(p) for k, p in enumerate(players) if nominal(p)}
    if mode == "anchored":
        return fit(players, totals, sf)
    first = min(
        sf
    )  # ladder, joint: pin one level, then shift so the levels' mean is nominal
    r = fit(players, totals, {first: sf[first]})
    return r + np.mean(list(sf.values())) - np.mean(r[list(sf)])


def rate(games, boot, mode, seed=0):
    players, pairs = tables(games)
    if mode == "ladder":  # Stockfish's own games only
        keep = {k for k, p in enumerate(players) if nominal(p)}
        pairs = {k: v for k, v in pairs.items() if set(k) <= keep}
    est = solve(players, pairs, mode)
    rng = np.random.default_rng(seed)
    reps = np.array([solve(players, pairs, mode, rng) for _ in range(boot)])
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    n, pts = defaultdict(int), defaultdict(float)
    for (a, b), c in pairs.items():
        for k, s in ((a, c[:, 1].sum()), (b, c[:, 0].sum() - c[:, 1].sum())):
            n[k] += int(c[:, 0].sum())
            pts[k] += s
    return {
        p: dict(
            elo=float(est[k]),
            lo=float(lo[k]),
            hi=float(hi[k]),
            games=n[k],
            score=pts[k] / max(n[k], 1),
        )
        for k, p in enumerate(players)
        if n[k]
    }


def label(p):
    if m := re.fullmatch(r"allie-h(\d+)", p):
        return f"Allie 2.0 human-like, conditioned {m[1]}"
    if m := re.fullmatch(r"allie-a(\d+)", p):
        return f"Allie 2.0 argmax, conditioned {m[1]}"
    if m := re.fullmatch(r"allie-s(\d+)-(\d+)", p):
        return f"Allie 2.0 argmax + search ({m[1]} sims), conditioned {m[2]}"
    if m := re.fullmatch(r"maia3-h(\d+)", p):
        return f"Maia-3 79M, conditioned {m[1]}"
    return f"Stockfish 19, UCI_Elo {nominal(p)}"


def order(p):
    """Allie human-like, argmax, search, then Maia-3, then Stockfish; by rating within each."""
    rank = ("allie-h", "allie-a", "allie-s", "maia3-h", "sf-")
    return next(i for i, f in enumerate(rank) if p.startswith(f)), int(
        re.findall(r"\d+", p)[-1]
    )


def results(games):
    """{(player, opponent): [games, points, draws]} for every non-Stockfish player."""
    by = defaultdict(lambda: [0, 0.0, 0])
    for g in games:
        w = SCORE[g["result"]]
        for me, them, s in (
            (g["white"], g["black"], w),
            (g["black"], g["white"], 1 - w),
        ):
            if not nominal(me):
                b = by[me, them]
                b[0] += 1
                b[1] += s
                b[2] += g["result"] == "1/2-1/2"
    return dict(sorted(by.items(), key=lambda kv: (order(kv[0][0]), order(kv[0][1]))))


def report(games, ratings, ladder, joint):
    span = lambda r: f"{r['elo']:.0f} [{r['lo']:.0f}, {r['hi']:.0f}]"
    print(f"{len(games)} games\n")
    print(
        "| Player | Measured Elo [95%] | Joint fit [95%] | Games | Score |\n|---|---:|---:|---:|---:|"
    )
    for p, r in sorted(ratings.items(), key=lambda kv: order(kv[0])):
        if not nominal(p):
            print(
                f"| {label(p)} | {span(r)} | {span(joint[p])} | {r['games']} | {100 * r['score']:.1f}% |"
            )
    if ladder:
        print(
            "\n| Stockfish UCI_Elo | From its own games [95%] | Joint fit [95%] |\n|---:|---:|---:|"
        )
        for p, r in sorted(ladder.items(), key=lambda kv: nominal(kv[0])):
            print(f"| {nominal(p)} | {span(r)} | {span(joint[p])} |")
    by = results(games)
    print(
        "\n| Player | Opponent | Games | Score | Elo-predicted | z | Draws |\n|---|---|---:|---:|---:|---:|---:|"
    )
    z = []
    for (me, them), (n, s, d) in by.items():
        e = 1 / (1 + 10 ** ((ratings[them]["elo"] - ratings[me]["elo"]) / 400))
        z.append((s - n * e) / math.sqrt(n * e * (1 - e)))
        print(
            f"| {me} | {them} | {n} | {100 * s / n:.0f}% | {100 * e:.0f}% | {z[-1]:+.1f} | {d} |"
        )
    z = np.array(z)
    print(
        f"\nresiduals (anchored fit): {len(z)} cells, sum z^2 {np.sum(z**2):.0f}, |z| > 2 in {np.sum(np.abs(z) > 2)}"
    )


def figure(ratings, path):
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
    ink, ink2, muted, grid, axis, blue = (
        "#27272a",
        "#52525b",
        "#71717a",
        "#e5e7eb",
        "#d4d4d8",
        "#2563eb",
    )
    plt.rcParams.update({
        "font.family": ["Nimbus Sans", "DejaVu Sans"], "font.size": 14, "text.color": ink,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "axes.labelsize": 14, "axes.labelpad": 9,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.spines.bottom": False, "xtick.labelcolor": muted, "ytick.labelcolor": muted,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 7, "ytick.major.pad": 7, "svg.fonttype": "path",
    })  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    fig.subplots_adjust(left=0.15, right=0.64, top=0.96, bottom=0.14)
    span = (600, 3000)
    ax.plot(span, span, color=axis, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.annotate(
        "y = x",
        (1050, 1050),
        xytext=(10, -12),
        textcoords="offset points",
        color=muted,
        fontsize=13,
    )
    ends, top = [], span[1]
    curves = (  # prefix, label, colour, line width, marker face, z
        ("maia3-h", "Maia-3 79M, sampled", "#85858d", 1.8, None, 3),
        ("allie-h", "Allie 2.0, sampled (T = 1)", blue, 2.6, None, 4),
        ("allie-a", "Allie 2.0, argmax", ink, 1.8, "white", 5),
    )
    for prefix, name, color, lw, face, z in curves:
        pts = sorted(
            (int(p[len(prefix) :]), r)
            for p, r in ratings.items()
            if re.fullmatch(prefix + r"\d+", p)
        )
        if not pts:
            continue
        x = np.array([q[0] for q in pts])
        y, lo, hi = (np.array([q[1][k] for q in pts]) for k in ("elo", "lo", "hi"))
        if face:  # few points: intervals as bars
            ax.vlines(x, lo, hi, color=color, lw=1.4, zorder=z)
        else:
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0, zorder=z - 1)
        ax.plot(x, y, color=color, lw=lw, zorder=z)
        ax.plot(
            x,
            y,
            "o",
            ms=8,
            mfc=face or color,
            mec=color if face else "white",
            mew=2 if face else 1.2,
            zorder=z,
        )
        ends.append((name, y[-1], color))
        top = max(top, hi.max() + 100)
    if (r := ratings.get("allie-s5-2800")) is not None:
        ax.vlines(2800, r["lo"], r["hi"], color=ink2, lw=1.4, zorder=5)
        ax.plot(2800, r["elo"], "D", ms=8, mfc="white", mec=ink2, mew=2, zorder=6)
        ends.append(("Allie 2.0, argmax + search", r["elo"], ink2))
        top = max(top, r["hi"] + 100)
    ends.sort(key=lambda t: t[1])
    gap, prev = (top - span[0]) / 16, -1e9
    for text, y0, color in ends:
        yy = max(y0, prev + gap)
        prev = yy
        ax.annotate(text, (1.0, y0), xycoords=("axes fraction", "data"), xytext=(1.03, yy), textcoords=("axes fraction", "data"), fontsize=14,
                    color=ink2, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=axis, lw=0.8, shrinkA=1, shrinkB=6))  # fmt: skip
    ax.set_xlim(*span)
    ax.set_ylim(span[0], top)
    ax.set_xticks(range(800, 3000, 400))
    ax.set_yticks(range(800, 3200, 400))
    ax.set_xlabel("Rating the model is conditioned on (both players)")
    ax.set_ylabel("Measured playing Elo\n(Stockfish UCI_Elo scale)")
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--out", default="ratings.json")
    p.add_argument("--figure", action="store_true")
    a = p.parse_args()
    games = load(a.dir)
    ratings = rate(games, a.boot, "anchored")
    ladder = (
        rate(games, a.boot, "ladder")
        if any(nominal(g["white"]) and nominal(g["black"]) for g in games)
        else {}
    )
    joint = rate(games, a.boot, "joint")
    report(games, ratings, ladder, joint)
    out = dict(games=len(games), ratings=ratings, ladder=ladder, joint=joint)
    Path(a.dir, a.out).write_text(json.dumps(out, indent=1) + "\n")
    if a.figure:
        figure(ratings, Path(a.dir, "strength"))


if __name__ == "__main__":
    main()
