"""Per-cell calibrated play (time control x 200-point mover-Elo bin): the searcher and budget whose move
distribution plays at the humans' strength (accuracy and blunder rate, in Elo through the human curves)
with the human move's cross-entropy at or below the raw policy's (T = 1 throughout). Candidates: the raw
policy, coverage's calibrated distribution at each budget, and lookahead at each call count with
pi ~ prior exp(beta Q) over a beta grid (each searcher's cross-entropy against its own run's raw policy).
Where lookahead was run, every candidate is scored on those positions only (paired). Selection minimizes
the debiased squared Elo error summed over the two metrics among candidates whose cell cross-entropy is
not above raw; cv: selected on one game half, scored on the other.

python calib_cells.py CALIB_DIR --coverage search-human-ann2c [--lookahead DIR] [--out cells.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import calib_fit as cf

COV = (8, 32, 128, 256)
CALLS = (1, 2, 4, 8, 16)
BETAS = (0.5, 1, 2, 3, 4, 6, 8, 12, 16)
K = [cf.METRICS.index(m) for m in ("accuracy", "blunder")]
# ms on the live bot (6 CPUs, 4 threads, engine alone): 14 ms a network call + 5.9 ms a leaf; coverage
# calls and leaves from the load tests' searches, lookahead from the search agent's counts
COST = {("coverage", 8): 70, ("coverage", 32): 330, ("coverage", 128): 1250, ("coverage", 256): 2350,
        ("lookahead", 1): 192, ("lookahead", 2): 296, ("lookahead", 4): 502, ("lookahead", 8): 911,
        ("lookahead", 16): 1730}  # fmt: skip


def softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


def lookahead(d):
    """position -> (legal tokens, prior, {calls: Q})."""
    out = {}
    for z in cf.chunks(d):
        calls = [c for c in CALLS if f"la_q{c}" in z]
        for n, i in enumerate(z["index"]):
            a, b = z["offsets"][n], z["offsets"][n + 1]
            out[int(i)] = (z["legal"][a:b].astype(int), z["prior"][a:b].astype(float),
                           {c: z[f"la_q{c}"][a:b].astype(float) for c in calls})  # fmt: skip
    return out


def rows(d, coverage, la):
    """Per position: cell, elo, half, and per candidate (name -> (accuracy, blunder, ce - raw ce)),
    human (accuracy, blunder)."""
    meta = np.load(d / "positions-human.npz")["meta"]
    mpv = cf.load_mpv(d / "mpv-human")
    dists = cf.load_dists([d / coverage])
    out = []
    for i, dd in dists.items():
        cps, m = mpv.get(i), meta[i]
        if cps is None or "cov_p256" not in dd or set(dd["legal"].tolist()) != set(cps):
            continue
        mm = cf.move_metrics([cps[t] for t in dd["legal"]])[K]
        h = list(dd["legal"]).index(int(m[8]))
        raw = dd["prior"].astype(float)
        ce0 = -np.log(max(raw[h], 1e-12))
        cand = {"raw": raw} | {
            f"coverage {n}": dd[f"cov_p{n}"].astype(float) for n in COV
        }
        res = {
            k: (p @ mm[0], p @ mm[1], -np.log(max(p[h], 1e-12)) - ce0)
            for k, p in cand.items()
        }
        if i in la:
            legal, prior, qs = la[i]
            order = {t: k for k, t in enumerate(legal)}
            perm = np.array([order[t] for t in dd["legal"]])
            prior, lp = prior[perm], np.log(np.maximum(prior[perm], 1e-300))
            ce1 = -np.log(max(prior[h], 1e-12))
            for c, q in qs.items():
                for beta in BETAS:
                    p = softmax(lp + beta * q[perm])
                    res[f"lookahead {c} b{beta}"] = (
                        p @ mm[0],
                        p @ mm[1],
                        -np.log(max(p[h], 1e-12)) - ce1,
                    )
        out.append(dict(cell=(int(m[2]), int(m[3])), elo=float(m[4]), half=hash((int(m[0]), int(m[1]))) % 2,
                        human=mm[:, h], res=res, la=i in la))  # fmt: skip
    return out


def score(rs, name, human, f):
    """Cell statistics of one candidate over positions rs: Elo errors (accuracy, blunder) and their
    SEs (+: the candidate plays stronger), cross-entropy gap and SE."""
    x = np.array([r["res"][name] for r in rs])
    hm = np.array([r["human"] for r in rs])
    e = float(np.mean([r["elo"] for r in rs]))
    out = {}
    for j, k in enumerate(K):
        g = x[:, j] - hm[:, j]
        gm, se = g.mean(), g.std() / np.sqrt(len(g))
        h0 = hm[:, j].mean()
        if cf.METRICS[k] in cf.LOG:
            gm, se = np.log(max(h0 + gm, 1e-9) / h0), se / h0
        s = cf.slope(
            human, f, k, e
        )  # the human curve's: gap / slope is + when it plays stronger
        out[cf.METRICS[k]] = (float(gm / s), float(abs(se / s)))
    out["ce"] = (float(x[:, 2].mean()), float(x[:, 2].std() / np.sqrt(len(x))))
    return out


def loss(st):
    return sum(st[m][0] ** 2 - st[m][1] ** 2 for m in ("accuracy", "blunder"))


def select(rs, human, f):
    names = list(rs[0]["res"])
    stats = {n: score(rs, n, human, f) for n in names}
    ok = [n for n in names if stats[n]["ce"][0] <= 0]
    return min(ok, key=lambda n: loss(stats[n])), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--coverage", default="search-human-ann2c")
    p.add_argument("--lookahead")
    p.add_argument("--out", default="cells.json")
    a = p.parse_args()
    d = Path(a.dir)
    la = lookahead(Path(a.lookahead)) if a.lookahead else {}
    R = rows(d, a.coverage, la)
    human = cf.summarize(
        cf.table(
            np.load(d / "positions-human.npz")["meta"],
            cf.load_mpv(d / "mpv-human"),
            {},
            {},
        ),
        [],
    )
    cells = defaultdict(list)
    for r in R:
        cells[r["cell"]].append(r)
    table = {}
    print(f"{len(R)} positions; {sum(r['la'] for r in R)} with lookahead")
    print(
        "cell              n  choice                  Elo acc  Elo blun   CE vs raw [95%]          cv Elo acc/blun  cv CE    ms"
    )
    for (f, b), rs in sorted(cells.items()):
        if not (800 <= b <= 2600) or len(rs) < 100:
            continue
        if any(r["la"] for r in rs):
            rs = [r for r in rs if r["la"]]
        choice, stats = select(rs, human, f)
        cv = []
        for h in (0, 1):
            tr, te = (
                [r for r in rs if r["half"] == h],
                [r for r in rs if r["half"] != h],
            )
            c, _ = select(tr, human, f)
            cv.append((c, score(te, c, human, f)))
        st = stats[choice]
        kind = choice.split()[0]
        budget = int(choice.split()[1]) if kind != "raw" else 0
        ms = COST.get((kind, budget), 0)
        cva = (
            np.mean([s["accuracy"][0] for _, s in cv]),
            np.mean([s["blunder"][0] for _, s in cv]),
        )
        cvce = np.mean([s["ce"][0] for _, s in cv])
        ce, cese = st["ce"]
        print(f"{cf.FORMATS[f]:9s} {b:4d} {len(rs):5d}  {choice:22s} {st['accuracy'][0]:+7.0f}  {st['blunder'][0]:+7.0f}"
              f"   {ce:+.4f} [{ce - 1.96 * cese:+.4f},{ce + 1.96 * cese:+.4f}]  {cva[0]:+5.0f}/{cva[1]:+5.0f}  {cvce:+.4f} {ms:5d}"
              f"   (cv picks: {cv[0][0]}; {cv[1][0]})")  # fmt: skip
        table[f"{cf.FORMATS[f]}/{b}"] = dict(n=len(rs), choice=choice, kind=kind, budget=budget,
                                             beta=float(choice.split("b")[-1]) if kind == "lookahead" else None,
                                             ms=ms, stats=stats[choice], raw=stats["raw"],
                                             cv=[dict(choice=c, stats=s) for c, s in cv],
                                             all={n: s for n, s in stats.items()})  # fmt: skip
    (d / a.out).write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
