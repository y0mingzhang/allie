"""Human positions for the strength-calibration eval, from held-out games only (never data/test*): the golden
eval's pools (strat-eval-v1: July 2026 games cleared by the golden-eval build), or with --store a whole held-out
month store sampled with the same filters (eval.build.select, cells format x 200-point mover-Elo bin; the dev
splits' games excluded, the test splits never opened).

Up to --per positions per (format, 200-point mover-Elo bin), each a scored human move at ply >= 10:
the model's exact prefix (header with both true Elos, base, increment) and causal clock features.
From a store, meta's (row, game start) is (game index, 0) and `sites` names each position's game.

python calib_positions.py OUT.npz [--per 1000]
python calib_positions.py OUT.npz --store heldout-2026-09/store/2026-09 --per 5000 --bins 800-2600
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from allie.data import mix as cm
from allie.eval import build as ev

G = "/data/group_data/dei-group/yimingz3/allie/strat-eval-v1"
FORMATS = ("bullet", "blitz", "rapid", "classical")
BINS = list(range(400, 3001, 200))  # 400 holds < 600, 3000 holds >= 3000


def elo(digits):
    return int("".join(map(str, digits)))


def strat(min_ply):
    """Rows, their feats and (format, bin) -> [(row, game start, position, cell)] from the golden eval."""
    with np.load(f"{G}/strat.npz") as z:
        rows, labels = z["rows"].astype(np.int64), z["labels"]
    with np.load(f"{G}/feats.npz") as z:
        feats = z["feats"]
    pool = defaultdict(list)
    for r in range(len(rows)):
        bos = np.flatnonzero(rows[r] == 2348)
        for t in np.flatnonzero(labels[r] >= 0):
            g0 = bos[bos <= t][-1]
            k = t - g0 - 11
            if k < min_ply:
                continue
            w, b = elo(rows[r, g0 + 3 : g0 + 7]), elo(rows[r, g0 + 7 : g0 + 11])
            mover = w if k % 2 == 0 else b
            pool[labels[r, t] // 4, min(max(mover // 200 * 200, 400), 3000)].append(
                (r, g0, t, labels[r, t])
            )
    return rows, feats, pool, None


def store(month, per, min_ply):
    """Games, their feats and (format, bin) -> [(game, 0, position, cell)]: games drawn like the golden eval's
    with cells format x BINS and ~40 x per moves per cell, every scored move at ply >= min_ply within ROW."""
    games, _, _ = ev.select(
        month, ev.excluded(("dev", "dev_expert")), "feats", BINS[1:] + [10**4], 40 * per
    )
    pool = defaultdict(list)
    for n, (t, lab, _, _) in enumerate(games):
        for j in np.flatnonzero(lab[: cm.ROW] >= 0):
            k = j - 11
            if k < min_ply:
                continue
            f = lab[j] // len(BINS)
            mover = elo(t[3:7] if k % 2 == 0 else t[7:11])
            cell = f * 4 + int(np.searchsorted(ev.UPPER, mover, side="right"))
            pool[f, BINS[lab[j] % len(BINS)]].append((n, 0, j, cell))
    return (
        [g[0].astype(np.int64) for g in games],
        [g[2] for g in games],
        pool,
        [g[3] for g in games],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--per", type=int, default=1000)
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--store", type=Path, help="a finalized held-out month store")
    p.add_argument("--bins", default="400-3000", help="LO-HI: the mover-Elo bins kept")
    a = p.parse_args()
    rows, feats, pool, sites = (
        store(a.store, a.per, a.min_ply) if a.store else strat(a.min_ply)
    )
    lo, hi = map(int, a.bins.split("-"))
    rng = np.random.default_rng(a.seed)
    keep = []
    for key in sorted(k for k in pool if lo <= k[1] <= hi):
        c = pool[key]
        take = rng.choice(len(c), min(a.per, len(c)), replace=False)
        keep += [c[i] + key for i in sorted(take)]
        games = len({(r, g) for r, g, *_ in c})
        print(
            f"{FORMATS[key[0]]:9s} {key[1]:4d}: {len(c):6d} moves in {games:5d} games -> {len(take)}"
        )
    tokens, fts, off, meta = [], [], [0], []
    for r, g0, t, cell, f, b in keep:
        tokens.append(rows[r][g0:t])
        fts.append(feats[r][g0:t])
        off.append(off[-1] + t - g0)
        k = t - g0 - 11
        w, bl = elo(rows[r][g0 + 3 : g0 + 7]), elo(rows[r][g0 + 7 : g0 + 11])
        mover, opp = (w, bl) if k % 2 == 0 else (bl, w)
        clock = feats[r][t - 1, 0]
        meta.append((r, g0, f, b, mover, opp, k, clock, rows[r][t], cell))
    np.savez_compressed(
        a.out,
        tokens=np.concatenate(tokens).astype(np.int16),
        feats=np.concatenate(fts).astype(np.int32),
        offsets=np.array(off),
        # row, game start, format, bin, mover elo, opponent elo, ply, mover seconds, human move token, cell
        meta=np.array(meta, np.int64),
        **({"sites": np.array([sites[m[0]] for m in meta])} if sites else {}),
    )
    print(f"{len(keep)} positions -> {a.out}")


if __name__ == "__main__":
    main()
