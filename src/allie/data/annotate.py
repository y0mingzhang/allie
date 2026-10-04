"""Backfill data.store annotations (end, evals) into a built month store.

usage: data.annotate MONTH_DIR [HF_DIR]
end is the final-position state from replaying the stored moves; evals (Stockfish, analysed games
only) come from the month's Hugging Face parquet files in HF_DIR, joined on the game id. Each games
shard is rewritten once: legacy columns verified unchanged, row order kept, atomic replace, so
running samplers are unaffected. Shards already annotated are skipped (resumable).
"""

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from allie.data.vocab import MOVES
from allie.data.store import END, align, end_state, eval_list

PUSH = [chess.Move.from_uci(m) for m in MOVES]
HF = False  # whether this pass has an eval source
# Analysed games as flat arrays (sorted 8-byte ids, offsets into VALS, per-ply evals), set before the
# shard pool forks: numpy buffers stay shared, unlike millions of small Python objects.
SITES, OFF, VALS = np.array([], "S8"), np.zeros(1, np.int64), np.array([], np.int16)


def hf_evals(path):
    """(game ids, eval counts, evals) of the analysed games in one Hugging Face file."""
    t = pq.read_table(path, columns=["Site", "movetext"])
    t = t.filter(pc.fill_null(pc.match_substring(t["movetext"], "%eval"), False))
    ids, lens, vals = [], [], []
    for site, mt in zip(t["Site"].to_pylist(), t["movetext"].to_pylist()):
        e = eval_list(mt)
        ids.append(site.rsplit("/", 1)[-1])
        lens.append(len(e))
        vals.extend(e)
    return np.array(ids, "S8"), np.array(lens, np.int64), np.array(vals, np.int16)


def join(parts):
    """Sorted flat arrays from per-file (ids, lens, vals)."""
    ids, lens, vals = (np.concatenate(x) for x in zip(*parts))
    order = np.argsort(ids, kind="stable")
    start, lens = np.cumsum(lens) - lens, lens[order]
    moved = np.repeat(start[order] - (np.cumsum(lens) - lens), lens)
    return ids[order], np.r_[0, np.cumsum(lens)], vals[moved + np.arange(len(moved))]


def annotated(path):
    """The shard's own completion record (schema metadata), not inferred from column contents."""
    meta = pq.read_schema(path).metadata or {}
    done = json.loads(meta.get(b"annotate", b"{}"))
    return bool(done.get("end") and (done.get("evals") or not HF))


def shard(path):
    if annotated(path):
        return 0, 0
    t = pq.read_table(path)
    moves = t.column("moves").combine_chunks()
    off, val = moves.offsets.to_numpy(), moves.values.to_numpy()
    sites = np.array(t.column("site").to_pylist(), "S8")
    j = np.searchsorted(SITES, sites).clip(max=max(0, len(SITES) - 1))
    hit = SITES[j] == sites if len(SITES) else np.zeros(len(t), bool)
    end, evals = np.empty(len(t), np.int8), []
    for i in range(len(t)):
        b = chess.Board()
        for m in val[off[i] : off[i + 1]]:
            b.push(PUSH[m])
        end[i] = end_state(b)
        raw = VALS[OFF[j[i]] : OFF[j[i] + 1]].tolist() if hit[i] else None
        evals.append([] if raw is None else align(raw, off[i + 1] - off[i]))
    t = t.drop_columns([c for c in ("end", "evals") if c in t.column_names])
    new = t.append_column("end", pa.array(end)).append_column(
        "evals", pa.array(evals, pa.large_list(pa.int16()))
    )
    record = json.dumps(dict(end=1, evals=int(HF))).encode()
    new = new.replace_schema_metadata(
        {**(t.schema.metadata or {}), b"annotate": record}
    )
    tmp = Path(path).with_suffix(".annotate-tmp")
    pq.write_table(new, tmp, compression="zstd")
    back = pq.read_table(tmp)
    assert back.drop_columns(["end", "evals"]).equals(t), (
        f"legacy columns changed: {path}"
    )
    assert back["end"].type == pa.int8() and 0 <= end.min() and end.max() < len(END)
    os.replace(tmp, path)
    return len(t), sum(map(bool, evals))


def main():
    global HF, SITES, OFF, VALS
    month, hf = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None
    HF = bool(hf)
    workers, start = len(os.sched_getaffinity(0)), time.monotonic()
    if hf:
        with ProcessPoolExecutor(
            min(16, workers)
        ) as ex:  # each holds a file's movetext
            SITES, OFF, VALS = join(
                ex.map(hf_evals, sorted(Path(hf).glob("*.parquet")))
            )
        print(
            json.dumps(
                dict(analysed=len(SITES), seconds=round(time.monotonic() - start))
            ),
            flush=True,
        )
    paths = sorted(month.glob("games/b*/shard-*.parquet"))
    games = analysed = 0
    with ProcessPoolExecutor(workers) as ex:  # forked after SITES, OFF, VALS are set
        for k, (n, e) in enumerate(ex.map(shard, paths), 1):
            games, analysed = games + n, analysed + e
            if k % 200 == 0 or k == len(paths):
                print(
                    json.dumps(
                        dict(shards=k, of=len(paths), games=games, analysed=analysed)
                    ),
                    flush=True,
                )
    (month / "annotate-complete.json").write_text(
        json.dumps(
            dict(
                at=time.time(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                games=games,
                analysed=analysed,
                analysed_in_source=len(SITES),
                evals=bool(hf),
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
