"""Structured Lichess game store for data mixing research.

build     parse a monthly pgn.zst into restartable per-chunk Parquet parts
finalize  bucket games (format x max-Elo x min-Elo), pre-shuffle into small shards, write stats
gate      check on-the-fly tokenization against games tokenized by the original pipeline
verify    exit 0 iff a month's build or finalize markers parse and agree with its files
"""

import argparse
import calendar
import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from allie import paths
from allie.data.vocab import (
    BOS,
    INCREMENTS_ID,
    MOVE_ID,
    SECONDS_ID,
    TERM_NORMAL,
    TERM_OTHER,
    UNK,
)

FORMATS = ("UltraBullet", "Bullet", "Blitz", "Rapid", "Classical", "Correspondence")
TERMINATIONS = (
    "Normal",
    "Time forfeit",
    "Abandoned",
    "Rules infraction",
    "Unterminated",
)
RESULTS = ("1-0", "0-1", "1/2-1/2", "*")
# Final-position state; with termination Normal, 0 means resignation (decisive) or agreement (draw).
END = ("none", "checkmate", "stalemate", "insufficient", "repetition", "fifty")
MOVE_INDEX = {m: i - 378 for m, i in MOVE_ID.items()}
HEADER = re.compile(r'^\[(\w+) "(.*)"\]$', re.M)
STRIP = re.compile(r"\{[^}]*\}|\([^)]*\)|\$\d+|\d+\.(?:\.\.)?|1-0|0-1|1/2-1/2|\*|[?!]+")
CLOCK = re.compile(r"\[%clk (\d+):(\d+):(\d+)\]")
EVAL = re.compile(r"\[%eval (#?)(-?[\d.]+)\]")
EVAL_MISSING = -32768  # a ply of an analysed game that carries no eval
MOVETEXT = re.compile(r"\{([^}]*)\}|([^\s{}]+)")
NOT_MOVE = re.compile(r"\d+\.(?:\.\.)?|1-0|0-1|1/2-1/2|\*|\$\d+")
VAL_CACHE = paths.DATA / "validation_cache"
SCHEMA = pa.schema(
    [
        ("site", pa.string()),
        ("white", pa.string()),
        ("black", pa.string()),
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("white_diff", pa.int16()),
        ("black_diff", pa.int16()),
        ("white_title", pa.string()),
        ("black_title", pa.string()),
        ("format", pa.int8()),
        ("event_kind", pa.int8()),
        ("rated_prefix", pa.bool_()),
        ("base", pa.int32()),
        ("increment", pa.int16()),
        ("termination", pa.int8()),
        ("result", pa.int8()),
        ("utc", pa.int64()),
        ("eco", pa.string()),
        ("opening", pa.string()),
        ("moves", pa.list_(pa.uint16())),
        ("clocks", pa.list_(pa.uint32())),
        ("move_hash", pa.uint64()),
        ("token_hash", pa.uint64()),
        ("end", pa.int8()),
        ("evals", pa.list_(pa.int16())),
    ]
)


def index(options, value):
    return options.index(value) if value in options else len(options)


def hash64(values):
    digest = hashlib.blake2b(
        np.asarray(values, np.uint16).tobytes(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def utc(h):
    try:
        y, mo, d = map(int, str(h["UTCDate"]).replace("-", ".").split("."))
        hh, mm, ss = map(int, str(h["UTCTime"]).split(":"))
        return calendar.timegm((y, mo, d, hh, mm, ss))
    except (KeyError, ValueError):
        return 0


EVENT_KINDS = ("game", "tournament", "swiss")


def eval_list(movetext):
    """Stockfish eval after each ply, from the comment attached to that move: centipawns from
    White's view, mate in n = +-(32000 - n), EVAL_MISSING for plies without one; [] if none."""
    if "%eval" not in movetext:
        return []
    evals, seen = [], False
    for comment, token in MOVETEXT.findall(movetext):
        if token:
            if not NOT_MOVE.fullmatch(token):
                evals.append(EVAL_MISSING)
        elif evals and (m := EVAL.search(comment)):
            mate, v = m.groups()
            evals[-1] = (
                (32000 - abs(int(v))) * (-1 if v.startswith("-") else 1)
                if mate
                else max(-30000, min(30000, round(float(v) * 100)))
            )
            seen = True
    return evals if seen else []


def align(evals, plies):
    """Per-ply evals of an analysed game whose ply count matches the stored moves, else []."""
    evals = list(evals)
    return evals if len(evals) == plies else []


def end_state(board):
    """Index into END of the final position."""
    if board.is_checkmate():
        return 1
    if board.is_stalemate():
        return 2
    if board.is_insufficient_material():
        return 3
    if board.is_repetition(3):
        return 4
    return 5 if board.is_fifty_moves() else 0


def record(h, movetext):
    """One game from PGN headers and movetext, or (None, reason)."""
    try:
        welo, belo = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError, TypeError):
        return None, "elo"
    board, moves = chess.Board(), []
    try:
        for san in STRIP.sub(" ", movetext).split():
            m = board.parse_san(san)
            moves.append(MOVE_INDEX[m.uci()])
            board.push(m)
    except (ValueError, KeyError):
        return None, "illegal"
    clocks = [
        3600 * int(a) + 60 * int(b) + int(c) for a, b, c in CLOCK.findall(movetext)
    ]
    tc = str(h.get("TimeControl") or "-")
    try:
        base, inc = map(int, tc.split("+", 1)) if "+" in tc else (-1, -1)
    except ValueError:
        base, inc = -2, -2
    # "Rated Blitz game|tournament <url>", but swiss events are "Blitz swiss <url>".
    event = str(h.get("Event") or "").split(" ")
    rated = event[0] == "Rated"
    event = event[1:] if rated else event
    diff = lambda k: int(h[k]) if h.get(k) not in (None, "") else None
    g = dict(
        site=str(h.get("Site") or "").rsplit("/", 1)[-1],
        white=h.get("White"),
        black=h.get("Black"),
        white_elo=welo,
        black_elo=belo,
        white_diff=diff("WhiteRatingDiff"),
        black_diff=diff("BlackRatingDiff"),
        white_title=h.get("WhiteTitle") or None,
        black_title=h.get("BlackTitle") or None,
        format=index(FORMATS, event[0] if event else ""),
        event_kind=index(EVENT_KINDS, event[1] if len(event) > 1 else ""),
        rated_prefix=rated,
        base=base,
        increment=inc,
        termination=index(TERMINATIONS, h.get("Termination")),
        result=index(RESULTS, h.get("Result")),
        utc=utc(h),
        eco=h.get("ECO"),
        opening=h.get("Opening"),
        moves=moves,
        clocks=clocks if len(clocks) == len(moves) else [],
        move_hash=hash64(moves),
        end=end_state(board),
        evals=align(eval_list(movetext), len(moves)),
    )
    g["token_hash"] = hash64(tokenize(g))
    return g, None


def elo_digits(elo):
    return [int(c) for c in f"{elo:04d}"]


def tokenize(g):
    """lichess_tokens_v2 token ids for one game record."""
    sec = str(g["base"]) if g["base"] >= 0 else "*" if g["base"] == -1 else "?"
    inc = (
        str(g["increment"])
        if g["increment"] >= 0
        else "*"
        if g["increment"] == -1
        else "?"
    )
    head = [
        BOS,
        SECONDS_ID.get(sec, UNK),
        INCREMENTS_ID.get(inc, UNK),
        *elo_digits(g["white_elo"]),
        *elo_digits(g["black_elo"]),
    ]
    tail = TERM_NORMAL if g["termination"] == 0 else TERM_OTHER
    return np.array([*head, *(m + 378 for m in g["moves"]), tail], np.uint16)


def parse_chunk(i, blob, out):
    rows, bad = {f.name: [] for f in SCHEMA}, {}
    for game in re.split(r"\n\n(?=\[Event )", blob.decode("utf-8", "replace").strip()):
        head, _, movetext = game.partition("\n\n")
        g, why = record(dict(HEADER.findall(head)), movetext)
        if g is None:
            bad[why] = bad.get(why, 0) + 1
            continue
        for k, v in g.items():
            rows[k].append(v)
    path = Path(out) / f"part-{i:05d}.parquet"
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=SCHEMA), tmp, compression="zstd")
    tmp.rename(path)
    return dict(chunk=i, games=len(rows["site"]), bad=bad)


def hf_header(row):
    """PGN header strings from one Lichess/standard-chess-games (Hugging Face) row."""
    h = {k: v for k, v in row.items() if v is not None and k != "movetext"}
    for k in ("WhiteElo", "BlackElo"):
        if k in h:
            h[k] = str(h[k])
    for k in ("WhiteRatingDiff", "BlackRatingDiff"):
        if k in h:
            h[k] = f"{h[k]:+d}"
    if "UTCDate" in h:
        h["UTCDate"] = h["UTCDate"].strftime("%Y.%m.%d")
    if "UTCTime" in h:
        h["UTCTime"] = h["UTCTime"].strftime("%H:%M:%S")
    return h


def parse_hf(i, path, out):
    rows, bad = {f.name: [] for f in SCHEMA}, {}
    for row in pq.read_table(path).to_pylist():
        g, why = record(hf_header(row), row["movetext"] or "")
        if g is None:
            bad[why] = bad.get(why, 0) + 1
            continue
        for k, v in g.items():
            rows[k].append(v)
    part = Path(out) / f"part-{i:05d}.parquet"
    tmp = part.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=SCHEMA), tmp, compression="zstd")
    tmp.rename(part)
    return dict(chunk=i, source=Path(path).name, games=len(rows["site"]), bad=bad)


def chunks(path, size=1 << 26):
    """Deterministic chunks of whole games: identical boundaries on every run."""
    import zstandard

    with (
        open(path, "rb") as f,
        zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True) as r,
    ):
        buf = b""
        while True:
            block = b""
            while len(block) < size and (piece := r.read(size - len(block))):
                block += piece
            if not block:
                break
            buf += block
            cut = buf.rfind(b"\n[Event ")
            if cut > 0:
                yield buf[: cut + 1]
                buf = buf[cut + 1 :]
        if buf.strip():
            yield buf


def write_atomic(path, text):
    """Replace path by text in one step: a kill never leaves a partial marker."""
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def swap(root, new):
    """Publish the complete dir new as root/games. Only complete dirs are ever named
    games or games.new here, so a leftover games.old is never the only copy."""
    old = root / "games.old"
    shutil.rmtree(old, ignore_errors=True)
    if (root / "games").exists():
        (root / "games").rename(old)
    new.rename(root / "games")
    shutil.rmtree(old, ignore_errors=True)


def logged(root):
    """Build log lines by chunk, the last one winning; lines cut short by a kill dropped."""
    out, path = {}, root / "build-log.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            x = json.loads(line)
            out[x["chunk"]] = x
        except (ValueError, KeyError, TypeError):
            pass
    return out


def check_store(root, stage):
    """Why the markers of a built ("build") or finalized ("final") month cannot be
    trusted, or None."""
    try:
        done = json.loads((root / "build-complete.json").read_text())
        assert done["parts"] == done["chunks"] > 0, "build-complete.json"
        assert set(logged(root)) == set(range(done["chunks"])), "build log"
        if stage == "final":
            stats = json.loads((root / "stats.json").read_text())
            buckets = json.loads((root / "buckets.json").read_text())
            assert stats["buckets"] == len(buckets), "bucket count"
            assert stats["games"] == sum(b["games"] for b in buckets), "game count"
            shards = [root / x for b in buckets for x in b["shards"]]
            assert all(x.is_file() for x in shards), "missing shard"
            files = sum(x.is_file() for x in (root / "games").rglob("*"))
            assert files == len(shards), "stray file in games"
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def verify(a):
    root = Path(a.out)
    why = check_store(root, a.stage)
    if not why and a.stage == "build":
        n = len(list((root / "parts").glob("part-*.parquet")))
        k = json.loads((root / "build-complete.json").read_text())["parts"]
        why = None if n == k else f"{n} parts for {k}"
    if why:
        print(f"{root} {a.stage} not done: {why}", flush=True)
        sys.exit(1)


def build(a):
    root = Path(a.out)
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    (root / "build-complete.json").unlink(missing_ok=True)
    # a part counts as built only with its log line (a kill can fall between the two)
    have = {int(p.stem.split("-")[1]) for p in parts.glob("part-*.parquet")}
    kept = {k: v for k, v in logged(root).items() if k in have}
    write_atomic(
        root / "build-log.jsonl", "".join(json.dumps(x) + "\n" for x in kept.values())
    )
    done = set(kept)
    workers = len(os.sched_getaffinity(0))
    with (
        open(root / "build-log.jsonl", "a") as log,
        ProcessPoolExecutor(workers) as ex,
    ):
        pending = set()

        def drain(futures):
            for fut in futures:
                log.write(json.dumps(fut.result()) + "\n")
            log.flush()

        inputs = (
            ((parse_hf, f) for f in sorted(Path(a.hf).glob("*.parquet")))
            if a.hf
            else ((parse_chunk, blob) for blob in chunks(a.pgn))
        )
        for i, (fn, x) in enumerate(inputs):
            if i in done:
                continue
            pending.add(ex.submit(fn, i, x, parts))
            if len(pending) >= 2 * workers:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                drain(finished)
        drain(wait(pending)[0])
    n = len(list(parts.glob("part-*.parquet")))
    write_atomic(
        root / "build-complete.json", json.dumps(dict(parts=n, chunks=i + 1)) + "\n"
    )


def val_token_hashes():
    """Hashes of complete validation games (header, moves, termination)."""
    manifest = json.loads((VAL_CACHE / "manifest.json").read_text())
    rows = np.load(VAL_CACHE / manifest["derived_file"], mmap_mode="r").reshape(
        -1, 1025
    )
    out = set()
    for r in rows:
        b = np.flatnonzero(r == BOS)
        for s, e in zip(b, [*b[1:], len(r)]):
            if e < len(r) or r[-1] in (TERM_NORMAL, TERM_OTHER):  # complete games only
                out.add(hash64(r[s:e]))
    return np.fromiter(out, np.uint64)


def large(f):
    """f with 64-bit offsets: take() concatenates chunks, and > 2 GB of strings or list values needs them."""
    if f.type == pa.string():
        return f.with_type(pa.large_string())
    return (
        f.with_type(pa.large_list(f.type.value_type)) if pa.types.is_list(f.type) else f
    )


def with_annotations(t):
    """Parts built before the end/evals columns get end -1 (unknown) and no evals;
    data.annotate backfills them."""
    if "end" not in t.column_names:
        t = t.append_column("end", pa.array(np.full(len(t), -1, np.int8)))
    if "evals" not in t.column_names:
        t = t.append_column("evals", pa.array([[]] * len(t), pa.list_(pa.int16())))
    return t


def finalize(a):
    root = Path(a.out)
    for f in ("stats.json", "buckets.json"):  # the previous store is no longer complete
        (root / f).unlink(missing_ok=True)
    assert not (why := check_store(root, "build")), why
    parts = sorted((root / "parts").glob("part-*.parquet"))
    annotated = all({"end", "evals"} <= set(pq.read_schema(p).names) for p in parts)
    table = pa.concat_tables(with_annotations(pq.read_table(p)) for p in parts)
    table = table.cast(pa.schema([large(f) for f in table.schema]))
    welo = table["white_elo"].to_numpy().astype(np.int32)
    belo = table["black_elo"].to_numpy().astype(np.int32)
    fmt = table["format"].to_numpy().astype(np.int32)
    hi = np.clip(np.maximum(welo, belo) // 100, 6, 30)
    lo = np.clip(np.minimum(welo, belo) // 200, 3, 14)
    bucket = (fmt + 1) * 10000 + hi * 100 + lo
    leak = np.isin(table["token_hash"].to_numpy(), val_token_hashes())
    table = table.append_column("val_leak", pa.array(leak)).append_column(
        "bucket", pa.array(bucket, pa.int32())
    )
    stats = summarize(table, welo, belo, fmt, leak, root)
    order = np.lexsort((np.random.default_rng(a.seed).random(len(bucket)), bucket))
    table, bucket = table.take(pa.array(order)), bucket[order]
    games = root / "games.new"  # published by rename once complete
    if games.exists():
        shutil.rmtree(games)
    native = dict(end=1, evals=1)  # every part built with the end/evals columns
    meta = {b"annotate": json.dumps(native).encode()} if annotated else None
    starts = np.r_[0, np.flatnonzero(np.diff(bucket)) + 1, len(bucket)]
    buckets = []
    for s, e in zip(starts[:-1], starts[1:]):
        code, shards = int(bucket[s]), []
        (games / f"b{code:05d}").mkdir(parents=True)
        for k, lo_ in enumerate(range(s, e, a.shard_games)):
            rel = f"games/b{code:05d}/shard-{k:04d}.parquet"
            part = table.slice(lo_, min(a.shard_games, e - lo_))
            pq.write_table(
                part.replace_schema_metadata(meta) if meta else part,
                games / rel.split("/", 1)[1],
                compression="zstd",
            )
            shards.append(rel)
        buckets.append(
            dict(
                code=code,
                format=code // 10000 - 1,
                max_elo_bin=code // 100 % 100 * 100,
                min_elo_bin=code % 100 * 200,
                games=int(e - s),
                shards=shards,
            )
        )
    swap(root, games)
    write_atomic(root / "buckets.json", json.dumps(buckets, indent=1) + "\n")
    text = json.dumps(stats | dict(buckets=len(buckets)), indent=2)
    write_atomic(root / "stats.json", text + "\n")
    print(json.dumps(stats, indent=2))


def summarize(table, welo, belo, fmt, leak, root):
    plies = pc.list_value_length(table["moves"]).to_numpy().astype(np.int64)
    white_moves, black_moves = (plies + 1) // 2, plies // 2
    wbot = (
        pc.equal(table["white_title"], "BOT")
        .fill_null(False)
        .to_numpy(zero_copy_only=False)
    )
    bbot = (
        pc.equal(table["black_title"], "BOT")
        .fill_null(False)
        .to_numpy(zero_copy_only=False)
    )
    wdiff = table["white_diff"].fill_null(0).to_numpy().astype(np.int32)
    bdiff = table["black_diff"].fill_null(0).to_numpy().astype(np.int32)
    clocks = pc.list_value_length(table["clocks"]).to_numpy()
    expert_w, expert_b = welo >= 2400, belo >= 2400
    expert_moves = int((white_moves * expert_w).sum() + (black_moves * expert_b).sum())
    bad = {}
    for line in (root / "build-log.jsonl").read_text().splitlines():
        for k, v in json.loads(line)["bad"].items():
            bad[k] = bad.get(k, 0) + v
    return dict(
        games=len(plies),
        plies=int(plies.sum()),
        tokens=int(plies.sum() + 12 * len(plies)),
        skipped=bad,
        games_by_format={
            f: int((fmt == i).sum()) for i, f in enumerate((*FORMATS, "other"))
        },
        games_by_max_elo_100={
            int(k) * 100: int(v)
            for k, v in zip(
                *np.unique(np.maximum(welo, belo) // 100, return_counts=True)
            )
        },
        expert2400_moves_by_mover=expert_moves,
        expert2400_moves_avg_elo_rule=int(plies[(welo + belo) // 2 >= 2400].sum()),
        expert2400_moves_in_avg_elo_below_2400=int(
            (white_moves * expert_w + black_moves * expert_b)[
                (welo + belo) // 2 < 2400
            ].sum()
        ),
        expert2400_bot_moves=int(
            (white_moves * (expert_w & wbot)).sum()
            + (black_moves * (expert_b & bbot)).sum()
        ),
        expert2400_moves_rating_diff_ge_25=int(
            (white_moves * (expert_w & (np.abs(wdiff) >= 25))).sum()
            + (black_moves * (expert_b & (np.abs(bdiff) >= 25))).sum()
        ),
        expert2400_moves_rating_diff_ge_50=int(
            (white_moves * (expert_w & (np.abs(wdiff) >= 50))).sum()
            + (black_moves * (expert_b & (np.abs(bdiff) >= 50))).sum()
        ),
        games_with_bot=int((wbot | bbot).sum()),
        games_without_clocks=int(((clocks == 0) & (plies > 0)).sum()),
        val_leak_games=int(leak.sum()),
    )


def gate(a):
    """Tokens from the structured store must equal the original pipeline's tokens for the same games."""
    ref = pq.read_table(a.v2_shard, columns=["Site", "tokens"])
    want = dict(zip(ref["Site"].to_pylist(), ref["tokens"].to_pylist()))
    raw = pq.read_table(a.hf_shard)
    raw = raw.filter(pc.is_in(raw["Site"], pa.array(list(want))))
    matched = mismatched = skipped = 0
    for h in raw.to_pylist():
        g, why = record(h, h["movetext"])
        if g is None:
            skipped += 1
            continue
        got = tokenize(g).tolist()
        if got == want[h["Site"]]:
            matched += 1
        else:
            mismatched += 1
            if mismatched <= 3:
                print(
                    "MISMATCH",
                    h["Site"],
                    "want",
                    want[h["Site"]][:16],
                    "got",
                    got[:16],
                    len(want[h["Site"]]),
                    len(got),
                )
    report = dict(
        reference_games=len(want),
        compared=len(raw),
        matched=matched,
        mismatched=mismatched,
        skipped=skipped,
        passed=mismatched == 0 and skipped == 0 and matched == len(want),
    )
    print(json.dumps(report))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--pgn", help="Lichess monthly .pgn.zst")
    src.add_argument(
        "--hf", help="directory of Lichess/standard-chess-games parquet files"
    )
    b.add_argument("--out", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--out", required=True)
    f.add_argument("--shard-games", type=int, default=50000)
    f.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("gate")
    g.add_argument("--hf-shard", required=True)
    g.add_argument("--v2-shard", required=True)
    g.add_argument("--out")
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True)
    v.add_argument("--stage", choices=("build", "final"), required=True)
    a = p.parse_args()
    dict(build=build, finalize=finalize, gate=gate, verify=verify)[a.cmd](a)
