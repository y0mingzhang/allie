"""Backfill chessdata annotations (end, evals) into a built month store.

usage: chessdata_annotate.py MONTH_DIR [HF_DIR]
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
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chess_vocab import MOVES
from chessdata import END, align, end_state, eval_list

PUSH = [chess.Move.from_uci(m) for m in MOVES]
EVALS = {}  # game id -> raw evals, filled before the shard pool forks
HF = False  # whether this pass has an eval source


def hf_evals(path):
    t = pq.read_table(path, columns=["Site", "movetext"])
    return {
        site.rsplit("/", 1)[-1]: np.array(eval_list(mt), np.int16)
        for site, mt in zip(t["Site"].to_pylist(), t["movetext"].to_pylist())
        if mt and "%eval" in mt
    }


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
    sites = t.column("site").to_pylist()
    end, evals = np.empty(len(t), np.int8), []
    for i in range(len(t)):
        b = chess.Board()
        for m in val[off[i] : off[i + 1]]:
            b.push(PUSH[m])
        end[i] = end_state(b)
        raw = EVALS.get(sites[i])
        evals.append([] if raw is None else align(raw.tolist(), off[i + 1] - off[i]))
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
    global HF
    month, hf = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None
    HF = bool(hf)
    workers, start = len(os.sched_getaffinity(0)), time.monotonic()
    if hf:
        with ProcessPoolExecutor(workers) as ex:
            for d in ex.map(hf_evals, sorted(Path(hf).glob("*.parquet"))):
                EVALS.update(d)
        print(
            json.dumps(
                dict(analysed=len(EVALS), seconds=round(time.monotonic() - start))
            ),
            flush=True,
        )
    paths = sorted(month.glob("games/b*/shard-*.parquet"))
    games = analysed = 0
    with ProcessPoolExecutor(workers) as ex:  # forked after EVALS is filled
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
                analysed_in_source=len(EVALS),
                evals=bool(hf),
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
