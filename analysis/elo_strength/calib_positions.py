"""Human positions for the strength-calibration eval, from the golden-eval pools only
(strat-eval-v1: July 2026 games cleared by the golden-eval build; never data/test*).

Up to --per positions per (format, 200-point mover-Elo bin), each a scored human move at ply >= 10:
the model's exact prefix (header with both true Elos, base, increment) and causal clock features.

python calib_positions.py OUT.npz [--per 1000]
"""

import argparse
from collections import defaultdict

import numpy as np

G = "/data/group_data/dei-group/yimingz3/allie/strat-eval-v1"
FORMATS = ("bullet", "blitz", "rapid", "classical")


def elo(digits):
    return int("".join(map(str, digits)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--per", type=int, default=1000)
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261004)
    a = p.parse_args()
    with np.load(f"{G}/strat.npz") as z:
        rows, labels = z["rows"].astype(np.int64), z["labels"]
    with np.load(f"{G}/feats.npz") as z:
        feats = z["feats"]
    pool = defaultdict(list)  # (format, bin) -> (row, game start, position)
    for r in range(len(rows)):
        bos = np.flatnonzero(rows[r] == 2348)
        for t in np.flatnonzero(labels[r] >= 0):
            g0 = bos[bos <= t][-1]
            k = t - g0 - 11
            if k < a.min_ply:
                continue
            w, b = elo(rows[r, g0 + 3 : g0 + 7]), elo(rows[r, g0 + 7 : g0 + 11])
            mover = w if k % 2 == 0 else b
            pool[labels[r, t] // 4, min(max(mover // 200 * 200, 400), 3000)].append(
                (r, g0, t)
            )
    rng = np.random.default_rng(a.seed)
    keep = []
    for key in sorted(pool):
        c = pool[key]
        take = rng.choice(len(c), min(a.per, len(c)), replace=False)
        keep += [c[i] + key for i in sorted(take)]
        games = len({(r, g) for r, g, _ in c})
        print(
            f"{FORMATS[key[0]]:9s} {key[1]:4d}: {len(c):6d} moves in {games:5d} games -> {len(take)}"
        )
    tokens, fts, off, meta = [], [], [0], []
    for r, g0, t, f, b in keep:
        tokens.append(rows[r, g0:t])
        fts.append(feats[r, g0:t])
        off.append(off[-1] + t - g0)
        k = t - g0 - 11
        w, bl = elo(rows[r, g0 + 3 : g0 + 7]), elo(rows[r, g0 + 7 : g0 + 11])
        mover, opp = (w, bl) if k % 2 == 0 else (bl, w)
        clock = feats[r, t - 1, 0]
        meta.append((r, g0, f, b, mover, opp, k, clock, rows[r, t], labels[r, t]))
    np.savez_compressed(
        a.out,
        tokens=np.concatenate(tokens).astype(np.int16),
        feats=np.concatenate(fts).astype(np.int32),
        offsets=np.array(off),
        # row, game start, format, bin, mover elo, opponent elo, ply, mover seconds, human move token, cell
        meta=np.array(meta, np.int64),
    )
    print(f"{len(keep)} positions -> {a.out}")


if __name__ == "__main__":
    main()
