"""Stratified evaluation set from one held-out Lichess month: strat-eval-v1 (July 2026 in data-v1) by default,
strat-eval-v2 (September 2026) with --store heldout-2026-09/store/2026-09 --out strat-eval-v2 --splits dev,dev_expert.

Cells are format {bullet, blitz, rapid, classical} x mover Elo {<1400, 1400-2000, 2000-2400, >=2400}.
Each cell gets ~TARGET scored moves drawn uniformly from rated human moves in that cell: the first
frac_cell x games of every pre-shuffled bucket. Dev/test games, validation leaks and BOT movers are
never scored. Whole games are packed into 1025-token rows padded with BOS; labels[i, j] is the cell
of token j of row i (-1 = unscored), so target j of the evaluator uses labels[:, 1:].
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from allie import paths
from allie.data import mix as cm
from allie.data.vocab import BOS

MONTH = cm.STORE / "2026-07"
OUT = paths.DATA / "strat-eval-v1"
SPLITS = Path(os.environ.get("ALLIE_PROJECT_ROOT", "/home/yimingz3/src/allie")) / "data"
FORMATS = ["bullet", "blitz", "rapid", "classical"]  # data-v1 format ids 1..4
BANDS = ["<1400", "1400-2000", "2000-2400", ">=2400"]
UPPER = [1400, 2000, 2400, 10**4]
CELLS = [f"{f}/{b}" for f in FORMATS for b in BANDS]
TARGET = 100_000


def excluded(splits=("dev", "dev_expert", "test", "test_expert")):
    ids = set()
    for s in splits:
        for line in open(SPLITS / f"{s}.jsonl"):
            ids.add(json.loads(line)["game-id"].rsplit("/", 1)[-1])
    return ids


def sides(path, skip, clock=False, upper=UPPER):
    """Shard games, their sites and the cell (format x band of upper) of each side's moves (-1 = not scorable)."""
    cols = cm.COLUMNS + (cm.CLOCK_COLUMNS if clock else [])
    g = cm.Games(pq.read_table(path, columns=cols), clock)
    site = pq.read_table(path, columns=["site"]).column("site").to_pylist()
    ok = g.rated.astype(bool) & ~g.leak & np.array([s not in skip for s in site])
    ok &= (g.fmt >= 1) & (g.fmt <= 4)
    base = (g.fmt.astype(int) - 1) * len(upper)
    cell = lambda elo, bot: np.where(
        ok & ~bot, base + np.searchsorted(upper, elo, side="right"), -1
    )
    plies = np.diff(g.off)
    return (
        g,
        site,
        cell(g.welo, g.wbot),
        cell(g.belo, g.bbot),
        (plies + 1) // 2,
        plies // 2,
    )


def per_cell(wc, bc, wn, bn, n=len(CELLS)):
    out = np.zeros(n)
    np.add.at(out, wc[wc >= 0], wn[wc >= 0])
    np.add.at(out, bc[bc >= 0], bn[bc >= 0])
    return out


def clocks(g, i, size):
    """Time left on the next mover's clock before each move (data.mix.clock_bucket), at the position predicting it:
    the base time for each side's first move, else that side's clock after its previous move; 0 elsewhere and for
    games without clocks."""
    out = np.zeros(size, np.int16)
    moves = g.off[i + 1] - g.off[i]
    c = g.cval[g.coff[i] : g.coff[i + 1]].astype(np.int64)
    if len(c) == moves and g.base[i] >= 0:
        before = np.concatenate([[g.base[i], g.base[i]], c])[:moves]
        out[10 : 10 + moves] = cm.clock_bucket(before)
    return out


def select(month, skip, side=None, upper=UPPER, target=TARGET):
    """Sampled games in bucket order: [(tokens, labels, side channel or None, site)], the sampling fractions and
    population moves per cell (format x band of upper)."""
    n = 4 * len(upper)
    buckets = [
        b
        for b in json.loads((month / "buckets.json").read_text())
        if 1 <= b["code"] // 10000 - 1 <= 4
    ]
    rate, total = {}, np.zeros(n)
    for b in buckets:
        _, _, wc, bc, wn, bn = sides(month / b["shards"][0], skip, upper=upper)
        rate[b["code"]] = per_cell(wc, bc, wn, bn, n) / len(wc)
        total += rate[b["code"]] * b["games"]
    frac = np.minimum(1.0, target / np.maximum(total, 1))
    out = []
    for b in buckets:
        # per cell, games sampled from the bucket's start
        take = np.ceil(frac * b["games"]).astype(int)
        need = int(take[rate[b["code"]] > 0].max(initial=0))
        seen = 0
        for shard in b["shards"]:
            if seen >= need:
                break
            g, site, wc, bc, _, _ = sides(month / shard, skip, side is not None, upper)
            idx = np.arange(seen, seen + len(wc))
            wsel = (wc >= 0) & (idx < take[np.maximum(wc, 0)])
            bsel = (bc >= 0) & (idx < take[np.maximum(bc, 0)])
            for i in np.flatnonzero(wsel | bsel):
                t, _ = g.tokens(i, True, True)
                lab = np.full(len(t), -1, np.int8)
                if wsel[i]:
                    lab[11 : len(t) - 1 : 2] = wc[i]
                if bsel[i]:
                    lab[12 : len(t) - 1 : 2] = bc[i]
                ch = None
                if side == "clocks":
                    ch = clocks(g, i, len(t))
                elif side == "feats":
                    ch = g.feats(i, len(t), drop=False)
                out.append((t, lab, ch, site[i]))
            seen += len(wc)
    return out, frac, total


def pack(games, side):
    """Whole games packed into ROW-token rows padded with BOS (the overflowing game truncated)."""
    rows, rlab, rclk, row, lab, clk = [], [], [], [], [], []
    for t, l, c, _ in games:
        if sum(map(len, row)) + len(t) > cm.ROW and row:
            rows.append(np.concatenate(row))
            rlab.append(np.concatenate(lab))
            rclk.append(np.concatenate(clk) if side else None)
            row, lab, clk = [], [], []
        row.append(t[: cm.ROW])
        lab.append(l[: cm.ROW])
        if side:
            clk.append(c[: cm.ROW])
    rows.append(np.concatenate(row))
    rlab.append(np.concatenate(lab))
    rclk.append(np.concatenate(clk) if side else None)
    pad = lambda x, v, dt: np.stack(
        [
            np.pad(
                r, [(0, cm.ROW - len(r))] + [(0, 0)] * (r.ndim - 1), constant_values=v
            )
            for r in x
        ]
    ).astype(dt)
    rows, rlab = pad(rows, BOS, np.int16), pad(rlab, -1, np.int8)
    if side:
        fill, dt = (0, np.int16) if side == "clocks" else (-1, np.int32)
        return rows, rlab, pad(rclk, fill, dt)
    return rows, rlab, None


def build(side=None, month=MONTH, out=OUT, splits=None):
    """side ("clocks" or "feats") reruns the same deterministic selection, checks the rows are
    byte-identical to strat.npz and writes the aligned side channel (clocks() or data.mix.Games.feats,
    no dropout) to <side>.npz."""
    skip = excluded(splits) if splits else excluded()
    games, frac, total = select(month, skip, side)
    rows, rlab, rclk = pack(games, side)
    if side:
        with np.load(out / "strat.npz") as z:
            assert np.array_equal(z["rows"], rows) and np.array_equal(z["labels"], rlab)
        np.savez(out / f"{side}.npz", **{side: rclk})
        manifest = json.loads((out / "manifest.json").read_text())
        manifest[f"{side}_sha256"] = hashlib.sha256(
            (out / f"{side}.npz").read_bytes()
        ).hexdigest()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{side}.npz written for {len(rows)} identical rows")
        return
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "strat.npz", rows=rows, labels=rlab)
    scored = np.bincount(rlab[rlab >= 0], minlength=len(CELLS))
    manifest = dict(
        source=str(month),
        cells=CELLS,
        target=TARGET,
        frac=frac.tolist(),
        population_moves=total.tolist(),
        scored_moves=scored.tolist(),
        games=len(games),
        rows=len(rows),
        excluded_games=len(skip),
        sha256=hashlib.sha256((out / "strat.npz").read_bytes()).hexdigest(),
    )
    if splits:
        manifest["excluded_splits"] = list(splits)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "sites.txt").write_text("".join(s + "\n" for *_, s in games))
    for c, n, p in zip(CELLS, scored, total, strict=True):
        print(f"{c:22} {n:8d} scored of ~{p / 1e6:8.2f}M")
    print(f"{len(games)} games, {len(rows)} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("side", nargs="?", choices=["clocks", "feats"])
    p.add_argument("--store", type=Path, default=MONTH, help="a finalized month store")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument(
        "--splits",
        type=lambda s: tuple(s.split(",")),
        help="dev/test splits whose games are excluded (default all four)",
    )
    a = p.parse_args()
    build(a.side, a.store, a.out, a.splits)
