"""Stratified evaluation set from held-out July 2026 in data-v1.

Cells are format {bullet, blitz, rapid, classical} x mover Elo {<1400, 1400-2000, 2000-2400, >=2400}.
Each cell gets ~TARGET scored moves drawn uniformly from rated human moves in that cell: the first
frac_cell x games of every pre-shuffled bucket. Dev/test games, validation leaks and BOT movers are
never scored. Whole games are packed into 1025-token rows padded with BOS; labels[i, j] is the cell
of token j of row i (-1 = unscored), so target j of the evaluator uses labels[:, 1:].
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from allie.data import mix as cm
from allie.data.vocab import BOS

MONTH = cm.STORE / "2026-07"
OUT = (
    Path(os.environ.get("ALLIE_DATA", "/data/group_data/dei-group/yimingz3/allie"))
    / "strat-eval-v1"
)
SPLITS = Path(os.environ.get("ALLIE_PROJECT_ROOT", "/home/yimingz3/src/allie")) / "data"
FORMATS = ["bullet", "blitz", "rapid", "classical"]  # data-v1 format ids 1..4
BANDS = ["<1400", "1400-2000", "2000-2400", ">=2400"]
UPPER = [1400, 2000, 2400, 10**4]
CELLS = [f"{f}/{b}" for f in FORMATS for b in BANDS]
TARGET = 100_000


def excluded():
    ids = set()
    for s in ("dev", "dev_expert", "test", "test_expert"):
        for line in open(SPLITS / f"{s}.jsonl"):
            ids.add(json.loads(line)["game-id"].rsplit("/", 1)[-1])
    return ids


def sides(path, skip, clock=False):
    """Shard games plus the cell of each side's moves (-1 = not scorable)."""
    cols = cm.COLUMNS + (cm.CLOCK_COLUMNS if clock else [])
    g = cm.Games(pq.read_table(path, columns=cols), clock)
    site = pq.read_table(path, columns=["site"]).column("site").to_pylist()
    ok = g.rated.astype(bool) & ~g.leak & np.array([s not in skip for s in site])
    ok &= (g.fmt >= 1) & (g.fmt <= 4)
    base = (g.fmt.astype(int) - 1) * len(BANDS)
    cell = lambda elo, bot: np.where(
        ok & ~bot, base + np.searchsorted(UPPER, elo, side="right"), -1
    )
    plies = np.diff(g.off)
    return g, cell(g.welo, g.wbot), cell(g.belo, g.bbot), (plies + 1) // 2, plies // 2


def per_cell(wc, bc, wn, bn):
    out = np.zeros(len(CELLS))
    np.add.at(out, wc[wc >= 0], wn[wc >= 0])
    np.add.at(out, bc[bc >= 0], bn[bc >= 0])
    return out


def build(side=None):
    """side ("clocks" or "feats") reruns the same deterministic selection, checks the rows are
    byte-identical to strat.npz and writes the aligned side channel (data.mix.Games.clock or
    Games.feats, no dropout) to <side>.npz."""
    clock = side is not None
    skip = excluded()
    buckets = [
        b
        for b in json.loads((MONTH / "buckets.json").read_text())
        if 1 <= b["code"] // 10000 - 1 <= 4
    ]
    rate, total = {}, np.zeros(len(CELLS))
    for b in buckets:
        _, wc, bc, wn, bn = sides(MONTH / b["shards"][0], skip)
        rate[b["code"]] = per_cell(wc, bc, wn, bn) / len(wc)
        total += rate[b["code"]] * b["games"]
    frac = np.minimum(1.0, TARGET / np.maximum(total, 1))
    toks, labels, clks, counts, games = [], [], [], np.zeros(len(CELLS)), 0
    for b in buckets:
        take = np.ceil(frac * b["games"]).astype(
            int
        )  # prefix of the bucket sampled per cell
        need = int(take[rate[b["code"]] > 0].max(initial=0))
        seen = 0
        for shard in b["shards"]:
            if seen >= need:
                break
            g, wc, bc, wn, bn = sides(MONTH / shard, skip, clock)
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
                toks.append(t)
                labels.append(lab)
                if clock:
                    side_of = g.clock if side == "clocks" else g.feats
                    clks.append(side_of(i, len(t), drop=False))
                games += 1
            counts += per_cell(np.where(wsel, wc, -1), np.where(bsel, bc, -1), wn, bn)
            seen += len(wc)
    rows, rlab, rclk, row, lab, clk = [], [], [], [], [], []
    for n, (t, l) in enumerate(zip(toks, labels, strict=True)):
        if sum(map(len, row)) + len(t) > cm.ROW and row:
            rows.append(np.concatenate(row))
            rlab.append(np.concatenate(lab))
            rclk.append(np.concatenate(clk) if clock else None)
            row, lab, clk = [], [], []
        row.append(t[: cm.ROW])
        lab.append(l[: cm.ROW])
        if clock:
            clk.append(clks[n][: cm.ROW])
    rows.append(np.concatenate(row))
    rlab.append(np.concatenate(lab))
    rclk.append(np.concatenate(clk) if clock else None)
    pad = lambda x, v, dt: np.stack(
        [
            np.pad(
                r, [(0, cm.ROW - len(r))] + [(0, 0)] * (r.ndim - 1), constant_values=v
            )
            for r in x
        ]
    ).astype(dt)
    rows, rlab = pad(rows, BOS, np.int16), pad(rlab, -1, np.int8)
    if clock:
        with np.load(OUT / "strat.npz") as z:
            assert np.array_equal(z["rows"], rows) and np.array_equal(z["labels"], rlab)
        fill, dt = (0, np.int16) if side == "clocks" else (-1, np.int32)
        np.savez(OUT / f"{side}.npz", **{side: pad(rclk, fill, dt)})
        manifest = json.loads((OUT / "manifest.json").read_text())
        manifest[f"{side}_sha256"] = hashlib.sha256(
            (OUT / f"{side}.npz").read_bytes()
        ).hexdigest()
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{side}.npz written for {len(rows)} identical rows")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "strat.npz", rows=rows, labels=rlab)
    scored = np.bincount(rlab[rlab >= 0], minlength=len(CELLS))
    manifest = dict(
        source=str(MONTH),
        cells=CELLS,
        target=TARGET,
        frac=frac.tolist(),
        population_moves=total.tolist(),
        scored_moves=scored.tolist(),
        games=games,
        rows=len(rows),
        excluded_games=len(skip),
        sha256=hashlib.sha256((OUT / "strat.npz").read_bytes()).hexdigest(),
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for c, n, p in zip(CELLS, scored, total, strict=True):
        print(f"{c:22} {n:8d} scored of ~{p / 1e6:8.2f}M")
    print(f"{games} games, {len(rows)} rows")


if __name__ == "__main__":
    assert sys.argv[2:] == [] and sys.argv[1:] in ([], ["clocks"], ["feats"])
    build(sys.argv[1] if sys.argv[1:] else None)
