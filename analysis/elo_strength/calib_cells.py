"""Per-cell calibrated play (time control x 200-point mover-Elo bin): the searcher and budget whose move
distribution plays at the humans' strength (accuracy and blunder rate, in Elo through the human curves)
with the human move's cross-entropy at or below the raw policy's (T = 1 throughout). Candidates: the raw
policy; coverage at each budget, as its calibrated human-move distribution or as pi ~ prior exp(beta Q)
over a beta grid; lookahead at each call count as pi ~ prior exp(beta Q); other coverage runs (--extra)
and header-rating offsets (--offset), each scored against its own run's raw policy; with --think, each
coverage beta as the bot plays it ("capped T bB": per position, the mixture over the rungs up to T that
the bot's drawn think time and clock allow, the policy when none does), and only those are picked.
Each cell is scored on its positions with the most candidates run (paired). Qualifying: cell
cross-entropy below raw's at 95% confidence (mean + 1.96 SE <= 0, SEs clustered by game; the raw policy
always qualifies); no search in bullet unless --search-bullet. Selection: the cheapest qualifying
candidate within TOL Elo (RMS of the two metrics, debiased) of the most accurate one; cv: selected on
one game half, scored on the other.

python calib_cells.py CALIB_DIR --coverage search-human-ann2c [--lookahead DIR] [--extra DIR ...] [--think]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

import calib_fit as cf

COV = (8, 32, 128, 256)
CALLS = (1, 2, 4, 8, 16)
BETAS = (0.5, 1, 2, 3, 4, 6, 8, 12, 16)
K = [cf.METRICS.index(m) for m in ("accuracy", "blunder")]
NAMES = []  # candidates, set by names() from BETAS


def names():
    NAMES[:] = (["raw"] + [f"coverage {n}" for n in COV] + [f"coverage {n} b{b:g}" for n in COV for b in BETAS]
                + [f"coverage {n}f b{b:g}" for n in COV for b in BETAS if FILL]
                + [f"lookahead {c} b{b:g}" for c in CALLS for b in BETAS])  # fmt: skip
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


FILL = False  # --fill: also coverage with unexpanded moves at the expanded moves' prior-weighted mean


def filled(q, heads, prior):
    """Coverage's Q with its unexpanded root moves (valued at the root's own W - L by the native
    backup) set to the prior-weighted mean of the expanded moves' Q."""
    q = q.astype(float)
    w = np.exp(heads[63:66] - heads[63:66].max())
    out = np.abs(q - (w[0] - w[2]) / w.sum()) < 1e-5
    if out.all() or not out.any():
        return q
    q[out] = np.average(q[~out], weights=prior[~out] + 1e-12)
    return q


THINK = False  # --think: also each coverage beta as the bot plays it under the think-time cap
LADDER = (8, 32, 128, 256, 1024)  # allie.lichess.calibration.LADDER["coverage"]
SIM = 0.015  # seconds a simulation (allie.lichess.calibration.COST["coverage"]; --sim)
LAG, RESERVE, SHARE = 0.3, 2.5, 0.5  # allie.lichess: Play.lag, behaviour.json's guard


def reach(time_logits, clock, inc, seconds):
    """P(the bot's think time >= each of `seconds`) and whether the clock's tenth allows them
    (allie.lichess.behaviour.think: a think-head bin, uniform within it, less the lag, under the guard)."""
    p = np.exp(time_logits - time_logits.max())
    p /= p.sum()
    b = np.arange(len(p))[:, None]
    s = np.asarray(seconds, float)[None]
    lin = np.clip(b + 0.5 - LAG - s, 0, 1)
    with np.errstate(divide="ignore"):
        log = np.clip(0.5 - (7.06 * np.log((s + LAG) / 16) - (b - 16)), 0, 1)
    out = p @ np.where(b < 16, lin, log)
    if clock is not None:
        hard = max(clock - RESERVE, 0.0)
        out = np.where((s[0] <= min(hard * SHARE + inc, hard)) & (s[0] <= hard / 10), out, 0.0)
    return out


def weights(time_logits, clock, inc, rungs):
    """P(the bot searches each rung) for rungs (ascending), the policy first: the largest rung
    whose simulations at SIM seconds fit both the drawn think time and a tenth of the clock."""
    up = reach(time_logits, clock, inc, np.array(rungs) * SIM)  # P(rung >= r)
    up = np.minimum.accumulate(up)
    return -np.diff(np.r_[1.0, up, 0.0])


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


def extra(d):
    """Another coverage run's chunks: position -> (legal tokens, prior, {tag: (P or None, Q)}) for
    every cov_q{tag} (e.g. 1024, 256v2) except the base budgets without a suffix."""
    out = {}
    for z in cf.chunks(d):
        tags = [k[5:] for k in z if k.startswith("cov_q") and k[5:] not in map(str, COV)]
        for n, i in enumerate(z["index"]):
            a, b = z["offsets"][n], z["offsets"][n + 1]
            out[int(i)] = (z["legal"][a:b].astype(int), z["prior"][a:b].astype(float),
                           {t: (z[f"cov_p{t}"][a:b].astype(float) if f"cov_p{t}" in z else None,
                                z[f"cov_q{t}"][a:b].astype(float)) for t in tags})  # fmt: skip
    return out


def offsets(d):
    """calib_offset.py's chunks: position -> (legal tokens, prior, {delta: P}) for every off_p{m200, p100, ...}."""
    out = {}
    for z in cf.chunks(d):
        tags = [k[5:] for k in z if k.startswith("off_p")]
        for n, i in enumerate(z["index"]):
            a, b = z["offsets"][n], z["offsets"][n + 1]
            out[int(i)] = (z["legal"][a:b].astype(int), z["prior"][a:b].astype(float),
                           {(-1 if t[0] == "m" else 1) * int(t[1:]): z[f"off_p{t}"][a:b].astype(float) for t in tags})  # fmt: skip
    return out


def leaves(tag):
    """Leaf evaluations of a coverage tag: simulations times views (256v2: 512; 256f: 256)."""
    n, v = re.fullmatch(r"(\d+)(?:v(\d+))?f?", tag).groups()
    return int(n) * int(v or 1)


def rows(d, coverage, la, ex=(), off=None):
    """Per position: format, bin, elo, half, lookahead present; X [position, candidate, (accuracy,
    blunder, ce - raw ce)] (NaN where a candidate was not run); H [position, (accuracy, blunder)]."""
    meta = np.load(d / "positions-human.npz")["meta"]
    mpv = cf.load_mpv(d / "mpv-human")
    dists = cf.load_dists([d / coverage])
    for e in ex:  # their candidates join NAMES
        tags = sorted({t for v in e.values() for t in v[2]}, key=leaves)
        for t in tags:
            NAMES.extend([f"coverage {t}"] + [f"coverage {t} b{b:g}" for b in BETAS])
    if off:
        NAMES.extend(f"offset {dl:+d}" for dl in sorted({dl for v in off.values() for dl in v[2]}))
    tops = list(COV) + sorted({int(t) for e in ex for v in e.values() for t in v[2] if t.isdigit()})
    if THINK:
        NAMES.extend(f"capped {t}" + (f" b{b:g}" if b is not None else "") for t in tops for b in (None, *BETAS))
    col = {n: j for j, n in enumerate(NAMES)}
    z = np.load(d / "positions-human.npz")
    tokens, starts = z["tokens"], z["offsets"]
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

        rung = {}  # simulations -> {beta or None: (accuracy, blunder, P(human move))}, raw prior

        def put(names, P, ce0, r=None, betas=None):
            P = np.atleast_2d(P)
            x[[col[n] for n in names]] = np.c_[
                P @ mm.T, -np.log(np.maximum(P[:, h], 1e-12)) - ce0
            ]
            if r is not None:
                for bt, row in zip(betas, np.c_[P @ mm.T, P[:, h]], strict=True):
                    rung.setdefault(r, {})[bt] = row

        ce0 = -np.log(max(raw[h], 1e-12))
        put(["raw"], raw, ce0)
        for n in COV:
            put([f"coverage {n}"], dd[f"cov_p{n}"].astype(float), ce0, n, [None])
            put(
                [f"coverage {n} b{b:g}" for b in BETAS],
                tilted(lp, dd[f"cov_q{n}"].astype(float)),
                ce0, n, BETAS,
            )
            if FILL:
                put([f"coverage {n}f b{b:g}" for b in BETAS], tilted(lp, filled(dd[f"cov_q{n}"], dd["heads"], raw)), ce0)
        if i in la:
            legal, prior, qs = la[i]
            order = {t: k for k, t in enumerate(legal)}
            perm = np.array([order[t] for t in dd["legal"]])
            prior = prior[perm]
            ce1 = -np.log(max(prior[h], 1e-12))
            for c, q in qs.items():
                put(
                    [f"lookahead {c} b{b:g}" for b in BETAS],
                    tilted(np.log(np.maximum(prior, 1e-300)), q[perm]),
                    ce1,
                )
        for e in ex:
            if i not in e:
                continue
            legal, prior, runs = e[i]
            order = {t: k for k, t in enumerate(legal)}
            perm = np.array([order[t] for t in dd["legal"]])
            prior = prior[perm]
            ce1, lp1 = -np.log(max(prior[h], 1e-12)), np.log(np.maximum(prior, 1e-300))
            for t, (P, q) in runs.items():
                r = int(t) if t.isdigit() else None
                if P is not None:
                    put([f"coverage {t}"], P[perm], ce1, r, [None])
                put([f"coverage {t} b{b:g}" for b in BETAS], tilted(lp1, q[perm]), ce1, r, BETAS)
        if THINK:
            tok = int(tokens[starts[i] + 2])
            clock = float(m[7]) if m[7] >= 0 else None
            rs = sorted(r for r in rung if r in LADDER)  # the bot's coverage rungs
            w = weights(dd["heads"][:63].astype(float), clock, tok - 10 if 10 <= tok <= 190 else 0, rs)
            base = np.r_[mm @ raw, raw[h]]
            for t in tops:
                if t not in rs:
                    continue
                k = rs.index(t) + 1
                ww = np.r_[w[:k], w[k:].sum()]  # rungs above the top: the top
                for bt in (None, *BETAS):
                    if any(bt not in rung[r] for r in rs[:k]):
                        continue
                    mix = ww[0] * base + sum(wi * rung[r][bt] for wi, r in zip(ww[1:], rs[:k], strict=True))
                    name = f"capped {t}" + (f" b{bt:g}" if bt is not None else "")
                    x[col[name]] = (mix[0], mix[1], -np.log(max(mix[2], 1e-12)) - ce0)
        if off and i in off:
            legal, prior, runs = off[i]
            order = {t: k for k, t in enumerate(legal)}
            perm = np.array([order[t] for t in dd["legal"]])
            ce1 = -np.log(max(prior[perm][h], 1e-12))
            for dl, P in runs.items():
                put([f"offset {dl:+d}"], P[perm], ce1)
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
    ok &= search | np.array([n.split()[0] in ("raw", "offset") for n in NAMES])  # free: allowed in bullet
    if THINK:  # the bot plays a search only as capped by the think time
        ok &= np.array([n.split()[0] in ("raw", "offset", "capped") for n in NAMES])
    near = np.flatnonzero(ok & (rms <= rms[ok].min() + TOL))
    return int(min(near, key=lambda j: (describe(NAMES[j])[3], loss[j])))


def describe(name):
    """(searcher, budget, beta, ms); budget a coverage tag (str) for the extra runs, whose ms scale
    coverage 256's per leaf evaluation."""
    p = name.split()
    kind, budget = p[0], p[1] if len(p) > 1 else "0"
    budget = int(budget) if budget.lstrip("+-").isdigit() else budget
    beta = float(p[2][1:]) if len(p) > 2 else None
    kind_ = "coverage" if kind == "capped" else kind
    n = int(budget.rstrip("f")) if isinstance(budget, str) and budget.rstrip("f").isdigit() else budget  # 128f: 128's
    ms = 0 if kind == "offset" else COST.get((kind_, n), round(COST["coverage", 256] / 256 * leaves(str(budget))) if kind_ == "coverage" else 0)
    return kind, budget, beta, ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--coverage", default="search-human-ann2c")
    p.add_argument("--lookahead")
    p.add_argument("--offset", help="calib_offset.py's chunk dir: the policy at shifted header ratings, as candidates")
    p.add_argument("--extra", nargs="*", default=[], help="other coverage runs' chunk dirs (cov_q{tag}, cov_p{tag})")
    p.add_argument("--out", default="cells.json")
    p.add_argument("--search-bullet", action="store_true", help="else bullet plays the policy: a search's seconds exceed its think times")
    p.add_argument("--sim", type=float, default=SIM, help="seconds a coverage simulation, for --think")
    p.add_argument("--think", action="store_true", help="also each beta under the bot's think-time cap (capped T bB)")
    p.add_argument("--fill", action="store_true", help="also coverage Q with unexpanded moves at the expanded mean")
    p.add_argument("--betas", help="comma-separated tilt betas (default: 0.5,1,2,3,4,6,8,12,16)")
    a = p.parse_args()
    if a.betas:
        globals()["BETAS"] = tuple(float(x) for x in a.betas.split(","))
    globals()["FILL"] = a.fill
    globals()["THINK"] = a.think
    globals()["SIM"] = a.sim
    names()
    d = Path(a.dir)
    la = lookahead(Path(a.lookahead)) if a.lookahead else {}
    info, X, H = rows(d, a.coverage, la, [extra(Path(e)) for e in a.extra], offsets(Path(a.offset)) if a.offset else None)
    present = np.isfinite(X[:, :, 0]).sum(1)
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
            if sel.any():  # the positions with the most candidates run on them (paired)
                sel &= present == present[sel].max()
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
