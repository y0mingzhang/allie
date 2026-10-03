"""Calibration positions from the July 2026 dev split (data/dev*.jsonl), disjoint from the golden
eval, which excludes every dev and test game (eval.build). Only dev files are opened.

The dev games are looked up in data-v1/2026-07's blitz shards and encoded exactly like the golden
eval (data.mix.Games tokens and feats, drop=False); the scorable moves are the golden eval's
(rated, no validation leak, human mover) and are sampled stratified by mover band.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from allie import paths
from allie.data import mix as cm

MONTH = cm.STORE / "2026-07"
SPLITS = paths.ROOT / "data"
OUT = paths.DATA / "maia3-bench/search/dev-positions.npz"
UPPER = [1400, 2000, 2400, 10**4]
PER_BAND = int(sys.argv[1]) if sys.argv[1:] else 3000


def main():
    ids = {
        json.loads(line)["game-id"].rsplit("/", 1)[-1]
        for s in ("dev", "dev_expert")
        for line in open(SPLITS / f"{s}.jsonl")
    }
    shards = [
        s
        for b in json.loads((MONTH / "buckets.json").read_text())
        if b["code"] // 10000 - 1 == 2
        for s in b["shards"]
    ]
    def shard_games(shard):
        """(site, tokens, feats, white bot, black bot, white elo, black elo) of each dev game."""
        site = pq.read_table(MONTH / shard, columns=["site"]).column("site").to_pylist()
        hit = [i for i, s in enumerate(site) if s in ids]
        if not hit:
            return []
        g = cm.Games(pq.read_table(MONTH / shard, columns=cm.COLUMNS + cm.CLOCK_COLUMNS), True)
        out = []
        for i in hit:
            if g.rated[i] and not g.leak[i] and g.fmt[i] == 2:
                t, _ = g.tokens(i, True, True)
                out.append((site[i], t, g.feats(i, len(t), drop=False), g.wbot[i], g.bbot[i], g.welo[i], g.belo[i]))
        return out

    with ThreadPoolExecutor(16) as pool:
        found = [x for part in pool.map(shard_games, shards) for x in part]
    prefixes, feats, cells, targets, games = [], [], [], [], []
    for k, (_, t, f, wbot, bbot, welo, belo) in enumerate(found):
        for m in range(len(t) - 12):
            if (wbot, bbot)[m % 2]:
                continue
            prefixes.append(t[: 11 + m])
            feats.append(f[: 11 + m])
            cells.append(4 + int(np.searchsorted(UPPER, (welo, belo)[m % 2], side="right")))
            targets.append(int(t[11 + m]))
            games.append(k)
    cells = np.array(cells)
    rng = np.random.default_rng(20260924)
    pick = np.sort(
        np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(cells == c),
                    min(PER_BAND, (cells == c).sum()),
                    replace=False,
                )
                for c in (4, 5, 6, 7)
            ]
        )
    )
    lens = np.array([len(prefixes[i]) for i in pick])
    np.savez(
        OUT,
        tokens=np.concatenate([prefixes[i] for i in pick]),
        feats=np.concatenate([feats[i] for i in pick]).astype(np.float32),
        offsets=np.r_[0, np.cumsum(lens)],
        cell=cells[pick],
        target=np.array(targets)[pick],
        game=np.array(games)[pick],
    )
    print(
        json.dumps(
            dict(
                dev_games=len(ids),
                found=len(found),
                moves=len(cells),
                per_band=np.bincount(cells, minlength=8)[4:].tolist(),
                sampled=len(pick),
            )
        )
    )


if __name__ == "__main__":
    main()
