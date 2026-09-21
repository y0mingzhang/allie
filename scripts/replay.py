"""Exposure replay of a planned run: the chessmix Sampler's own draws, cursors, acceptance and
permutation (Sampler._accepted, with each game kept as its length, bucket and (shard, row)
identity instead of its tokens), then batch()'s pop / pack order: whole games until a 1025-token
row is full, the last one cut. A repeat is a game consumed into a row after an earlier
consumption of the same (shard, row); a wrap is a bucket whose cursor took more candidates than
its pool (a traversal restart, rejected candidates included).

Usage: replay.py PLAN RUN_INDEX OUT.json [ROWS]. ROWS defaults to the whole run (steps x 512); a
shorter replay scales seen / token rates to the run but reports repeats and wraps of the
observed prefix only. Per bucket: pool games, taken candidates, seen games, repeats, passes;
totals and the format x Elo-band (stronger player) cells.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

plan_path, index, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
study = plan_path.parent
src = study / "source-ours"
sys.path.insert(0, str(src if src.exists() else Path(__file__).resolve().parent))
import chessmix as cm  # noqa: E402

run = json.loads(plan_path.read_text())["runs"][index]
target = run["steps"] * 512
rows = int(sys.argv[4]) if len(sys.argv) > 4 else target


class Light(cm.Shard):
    """A shard's policy inputs and game lengths only."""

    def __init__(self, path, aux=False, feats=False):
        cols = ["moves", "white_elo", "black_elo", "format", "rated_prefix", "val_leak"]
        t = cm.read(path, cols)
        g = type("G", (), {})()
        get = lambda k: t.column(k).to_numpy()
        g.welo, g.belo = (
            get("white_elo").astype(np.int32),
            get("black_elo").astype(np.int32),
        )
        g.fmt, g.rated, g.leak = get("format"), get("rated_prefix"), get("val_leak")
        g.len = pc.list_value_length(t.column("moves")).to_numpy() + 12
        g.n, g.code = len(g.len), int(Path(path).parent.name[1:])
        g.src, g.month = g.code // 100000, Path(path).parents[2].name
        self.path, self.g, self.phase = path, g, None


cm.Shard = Light
hist = str(study / "history-counts.json") if run.get("history") else None
s = cm.Sampler(
    run["policy"],
    run["seed"],
    total_rows=target,
    pool_frac=run["pool_frac"],
    stores=sorted({str(Path(m).parent) for m in run["months"]}),
    months=run["months"],
    history=hist,
)
pool = {c: sum(n for _, n in s.units[c]) for c in s.codes}
taken = dict.fromkeys(s.codes, 0)
ids, used = {}, []  # shard path -> id; per id, which rows were consumed


def accepted(p):
    """Sampler._accepted, games as (length, code, shard id, row)."""
    ph = s._weights(p)
    s._warm()
    got = []
    for k, pieces in s._chunk(s.rng, s.cursor):
        code = s.codes[k]
        for path, n, idx in pieces:
            shard = s._shard(k, path)
            assert n <= shard.g.n, (path, n, shard.g.n)
            shard.policy(s.fn, ph)
            ok = idx[s.rng.random(len(idx)) * s.caps[k] < shard.w[idx]]
            if path not in ids:
                ids[path] = len(used)
                used.append(np.zeros(n, bool))
            got += [(shard.g.len[i], code, ids[path], i) for i in ok]
            taken[code] += len(idx)
    assert not s.preload, "warm-up replay diverged from the draws"
    got = [got[i] for i in s.rng.permutation(len(got))]
    s.ahead = s._load(s._paths())
    return got


seen, repeats, tokens = (dict.fromkeys(s.codes, 0) for _ in range(3))
started = time.time()
for r in range(rows):
    size = 0
    while size < cm.ROW:
        if not s.pool:
            s.pool = accepted(s.seen / s.total_rows)
        n, code, j, i = s.pool.pop()
        seen[code] += 1
        repeats[code] += int(used[j][i])
        used[j][i] = True
        tokens[code] += min(n, cm.ROW - size)
        size += n
    s.seen += 1
    if r % 100000 == 0:
        print(f"row {r}/{rows} {time.time() - started:.0f}s", flush=True)
scale = target / rows
table = {}
for k in run["policy"].split("+"):
    if k.startswith("table:"):
        table = json.loads((cm.RECIPES / f"{k[6:]}.json").read_text())["weights"]
buckets = {
    str(c): dict(
        pool=pool[c],
        taken=taken[c] * scale,
        seen=seen[c] * scale,
        repeats=repeats[c],
        tokens=tokens[c] * scale,
        passes=seen[c] * scale / max(1, pool[c]),
        table=table.get(str(c)),
    )
    for c in s.codes
}
cells = {}
for c, b in buckets.items():
    c = int(c)
    fmt = "engine" if c // 100000 in cm.ENGINE else str(c // 10000 % 10 - 1)
    band = int(np.searchsorted([1400, 2000, 2400], c // 100 % 100 * 100, side="right"))
    x = cells.setdefault(
        f"{fmt}/{band}", dict(pool=0, seen=0.0, repeats=0.0, tokens=0.0)
    )
    for k in x:
        x[k] += b[k]
total = sum(b["tokens"] for b in buckets.values())
for x in cells.values():
    x["passes"], x["share"] = x["seen"] / max(1, x["pool"]), x["tokens"] / total
res = dict(
    run=run["name"],
    policy=run["policy"],
    pool_frac=run["pool_frac"],
    target_rows=target,
    replayed_rows=rows,
    repeats_observed_only=rows < target,
    replay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    chessmix_sha256=hashlib.sha256(Path(cm.__file__).read_bytes()).hexdigest(),
    infeasible=len(s.infeasible),
    seen=sum(b["seen"] for b in buckets.values()),
    repeats=sum(b["repeats"] for b in buckets.values()),
    wrapped_buckets=sum(taken[c] > pool[c] for c in s.codes),
    max_passes=max(b["passes"] for b in buckets.values()),
    seconds=time.time() - started,
    cells=dict(sorted(cells.items())),
    buckets=buckets,
)
out.write_text(json.dumps(res, indent=1) + "\n")
print(
    json.dumps(
        {k: v for k, v in res.items() if k not in ("buckets", "cells")}, indent=1
    )
)
