"""Per-cell calibrated play (time control x 200-point mover-Elo bin): the searcher and budget whose move
distribution plays at the humans' strength (accuracy and blunder rate, in Elo through the human curves)
with the human move's cross-entropy at or below the raw policy's (T = 1 throughout). Candidates: the raw
policy; coverage at each budget, as its calibrated human-move distribution or as pi ~ prior exp(beta Q)
over a beta grid; lookahead at each call count as pi ~ prior exp(beta Q) (its cross-entropy against its
own run's raw policy). Where lookahead was run, every candidate is scored on those positions only
(paired). Qualifying: cell cross-entropy below raw's at 95% confidence (mean + 1.96 SE <= 0; the raw
policy always qualifies); no search in bullet unless --search-bullet (a search outlasts bullet's think
times). Selection: the cheapest qualifying candidate within TOL Elo (RMS of the two metrics,
debiased) of the most accurate one; cv: selected on one game half, scored on the other.

python calib_cells.py CALIB_DIR --coverage search-human-ann2c [--lookahead DIR] [--out cells.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import calib_fit as cf

COV = (8, 32, 128, 256)
CALLS = (1, 2, 4, 8, 16)
BETAS = (0.5, 1, 2, 3, 4, 6, 8, 12, 16)
K = [cf.METRICS.index(m) for m in ("accuracy", "blunder")]
NAMES = (["raw"] + [f"coverage {n}" for n in COV] + [f"coverage {n} b{b}" for n in COV for b in BETAS]
         + [f"lookahead {c} b{b}" for c in CALLS for b in BETAS])  # fmt: skip
# ms of one search alone, 4 threads, int8, a 6-CPU preempt node (EPYC 9354; bench/searchcost.py, median
# of 6 positions); coverage 8 extrapolated
COST = {("coverage", 8): 200, ("coverage", 32): 741, ("coverage", 128): 2832, ("coverage", 256): 5388,
        ("lookahead", 1): 648, ("lookahead", 2): 944, ("lookahead", 4): 1481, ("lookahead", 8): 2541,
        ("lookahead", 16): 5067}  # fmt: skip


def tilted(lp, q):
    """[beta, move] of pi ~ exp(lp + beta q)."""
    z = lp[None] + np.array(BETAS)[:, None] * q[None]
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


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
    """Per position: format, bin, elo, half, lookahead present; X [position, candidate, (accuracy,
    blunder, ce - raw ce)] (NaN where a candidate was not run); H [position, (accuracy, blunder)]."""
    meta = np.load(d / "positions-human.npz")["meta"]
    mpv = cf.load_mpv(d / "mpv-human")
    dists = cf.load_dists([d / coverage])
    col = {n: j for j, n in enumerate(NAMES)}
    info, X, H = [], [], []
    for i, dd in dists.items():
        cps, m = mpv.get(i), meta[i]
        if cps is None or "cov_p256" not in dd or set(dd["legal"].tolist()) != set(cps):
            continue
        mm = cf.move_metrics([cps[t] for t in dd["legal"]])[K]
        h = list(dd["legal"]).index(int(m[8]))
        raw = dd["prior"].astype(float)
        lp = np.log(np.maximum(raw, 1e-300))
        x = np.full((len(NAMES), 3), np.nan, np.float32)

        def put(names, P, ce0):
            P = np.atleast_2d(P)
            x[[col[n] for n in names]] = np.c_[
                P @ mm.T, -np.log(np.maximum(P[:, h], 1e-12)) - ce0
            ]

        ce0 = -np.log(max(raw[h], 1e-12))
        put(["raw"], raw, ce0)
        put(
            [f"coverage {n}" for n in COV],
            np.stack([dd[f"cov_p{n}"].astype(float) for n in COV]),
            ce0,
        )
        for n in COV:
            put(
                [f"coverage {n} b{b}" for b in BETAS],
                tilted(lp, dd[f"cov_q{n}"].astype(float)),
                ce0,
            )
        if i in la:
            legal, prior, qs = la[i]
            order = {t: k for k, t in enumerate(legal)}
            perm = np.array([order[t] for t in dd["legal"]])
            prior = prior[perm]
            ce1 = -np.log(max(prior[h], 1e-12))
            for c, q in qs.items():
                put(
                    [f"lookahead {c} b{b}" for b in BETAS],
                    tilted(np.log(np.maximum(prior, 1e-300)), q[perm]),
                    ce1,
                )
        info.append(
            (
                int(m[2]),
                int(m[3]),
                float(m[4]),
                hash((int(m[0]), int(m[1]))) % 2,
                i in la,
                hash((int(m[0]), int(m[1]))),
            )
        )
        X.append(x)
        H.append(mm[:, h])
    info = np.array(
        info,
        dtype=[("f", int), ("bin", int), ("elo", float), ("half", int), ("la", bool), ("game", np.int64)],
    )
    return info, np.stack(X), np.array(H)


def clustered(x, games):
    """Mean over positions and its standard error with positions clustered by game (a game gives
    several positions): x [position, ...]."""
    n, m = len(x), x.mean(0)
    _, inv = np.unique(games, return_inverse=True)
    c = np.zeros((inv.max() + 1, *x.shape[1:]))
    np.add.at(c, inv, x - m)
    return m, np.sqrt((c**2).sum(0)) / n


def score(X, H, elo, human, f, games):
    """Per candidate: Elo errors (accuracy, blunder) and SEs (+: plays stronger than the humans), and
    the cross-entropy gap and SE (SEs clustered by game); NaN for candidates not run on these
    positions."""
    out = np.full((X.shape[1], 6), np.nan)
    for j, k in enumerate(K):
        gm, se = clustered(X[:, :, j] - H[:, None, j], games)
        h0 = H[:, j].mean()
        if cf.METRICS[k] in cf.LOG:
            gm, se = np.log(np.maximum(h0 + gm, 1e-9) / h0), se / h0
        s = cf.slope(
            human, f, k, elo
        )  # the human curve's: gap / slope is + when it plays stronger
        out[:, 2 * j], out[:, 2 * j + 1] = gm / s, np.abs(se / s)
    out[:, 4], out[:, 5] = clustered(X[:, :, 2], games)
    return out


TOL = 20  # Elo: the cheapest candidate whose error is within this of the best's


def select(st, search=True):
    """The cheapest qualifying candidate whose RMS Elo error (debiased, over the two metrics) is within
    TOL of the best qualifying one's; ties: the smaller error."""
    loss = (st[:, 0] ** 2 - st[:, 1] ** 2 + st[:, 2] ** 2 - st[:, 3] ** 2) / 2
    rms = np.sqrt(np.maximum(loss, 0))
    ok = (st[:, 4] + 1.96 * st[:, 5] <= 0) & np.isfinite(loss)  # a searcher: below raw at 95%
    ok[0] = True
    ok[1:] &= search
    near = np.flatnonzero(ok & (rms <= rms[ok].min() + TOL))
    return int(min(near, key=lambda j: (describe(NAMES[j])[3], loss[j])))


def describe(name):
    p = name.split()
    kind, budget = p[0], int(p[1]) if len(p) > 1 else 0
    beta = float(p[2][1:]) if len(p) > 2 else None
    return kind, budget, beta, COST.get((kind, budget), 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--coverage", default="search-human-ann2c")
    p.add_argument("--lookahead")
    p.add_argument("--out", default="cells.json")
    p.add_argument("--search-bullet", action="store_true", help="else bullet plays the policy: a search's seconds exceed its think times")
    a = p.parse_args()
    d = Path(a.dir)
    la = lookahead(Path(a.lookahead)) if a.lookahead else {}
    info, X, H = rows(d, a.coverage, la)
    human = cf.summarize(
        cf.table(
            np.load(d / "positions-human.npz")["meta"],
            cf.load_mpv(d / "mpv-human"),
            {},
            {},
        ),
        [],
    )
    print(f"{len(X)} positions; {info['la'].sum()} with lookahead")
    print(
        "cell              n  choice                   Elo acc  Elo blun  CE vs raw [95%]           cv Elo acc/blun   cv CE     ms"
    )
    table = {}
    for f in range(4):
        for b in range(800, 2601, 200):
            sel = (info["f"] == f) & (info["bin"] == b)
            if sel.any() and info["la"][sel].any():
                sel &= info["la"]
            if sel.sum() < 100:
                continue
            elo = info["elo"][sel].mean()
            st = score(X[sel], H[sel], elo, human, f, info["game"][sel])
            c = select(st, f > 0 or a.search_bullet)
            cv = []
            for h in (0, 1):
                tr, te = sel & (info["half"] == h), sel & (info["half"] != h)
                ch = select(score(X[tr], H[tr], info["elo"][tr].mean(), human, f, info["game"][tr]), f > 0 or a.search_bullet)
                cv.append(
                    (
                        NAMES[ch],
                        score(X[te], H[te], info["elo"][te].mean(), human, f, info["game"][te])[ch],
                    )
                )
            kind, budget, beta, ms = describe(NAMES[c])
            s = st[c]
            cva = np.mean([x[0] for _, x in cv]), np.mean([x[2] for _, x in cv])
            cvce = np.mean([x[4] for _, x in cv])
            print(f"{cf.FORMATS[f]:9s} {b:4d} {sel.sum():5d}  {NAMES[c]:22s} {s[0]:+7.0f}  {s[2]:+7.0f}"
                  f"   {s[4]:+.4f} [{s[4] - 1.96 * s[5]:+.4f},{s[4] + 1.96 * s[5]:+.4f}]  {cva[0]:+5.0f}/{cva[1]:+5.0f}  {cvce:+.4f} {ms:5d}"
                  f"   (cv picks: {cv[0][0]}; {cv[1][0]})", flush=True)  # fmt: skip
            cols = (
                "elo_accuracy",
                "se_accuracy",
                "elo_blunder",
                "se_blunder",
                "ce",
                "se_ce",
            )
            table[f"{cf.FORMATS[f]}/{b}"] = dict(n=int(sel.sum()), choice=NAMES[c], kind=kind, budget=budget, beta=beta, ms=ms,
                                                 stats=dict(zip(cols, map(float, s))), raw=dict(zip(cols, map(float, st[0]))),
                                                 cv=[dict(choice=n, stats=dict(zip(cols, map(float, x)))) for n, x in cv],
                                                 all={n: dict(zip(cols, map(float, st[j]))) for j, n in enumerate(NAMES)
                                                      if np.isfinite(st[j, 0])})  # fmt: skip
    (d / a.out).write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
