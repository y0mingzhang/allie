"""Fast month build: the store of `chessdata.py build --hf` then `finalize`, byte for
byte, in bounded memory.

build     parse the Hugging Face row groups in threads with the C core (fastbuild.c)
          and spill the games grouped by bucket
finalize  per bucket: gather, order and shard exactly like chessdata.finalize, then
          buckets.json and stats.json
check     compare the C core with chessdata.record() row by row on sampled rows

Rows the C core cannot reproduce exactly (non-ASCII movetext, unusual TimeControl,
oversized numbers) go through chessdata.record() itself.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chessdata as cd
from chess_vocab import MOVES

SRC = Path(__file__).with_name("fastbuild.c")
COLS = (
    "Event Site White Black Result WhiteTitle BlackTitle WhiteElo BlackElo "
    "WhiteRatingDiff BlackRatingDiff UTCDate UTCTime ECO Opening Termination "
    "TimeControl movetext"
).split()
TYPES = dict.fromkeys(COLS, pa.string()) | dict(
    WhiteElo=pa.int16(),
    BlackElo=pa.int16(),
    WhiteRatingDiff=pa.int16(),
    BlackRatingDiff=pa.int16(),
    UTCDate=pa.date32(),
    UTCTime=pa.time32("ms"),
)
SKIPS = {1: "elo", 2: "illegal"}
STRS = ("site", "white", "black", "white_title", "black_title", "eco", "opening")
# fastbuild.c hardcodes these tables
assert cd.FORMATS[3:] == ("Rapid", "Classical", "Correspondence")
assert cd.FORMATS[:3] == ("UltraBullet", "Bullet", "Blitz")
assert cd.EVENT_KINDS == ("game", "tournament", "swiss")
assert cd.RESULTS == ("1-0", "0-1", "1/2-1/2", "*")
assert cd.TERMINATIONS[:3] == ("Normal", "Time forfeit", "Abandoned")
assert cd.TERMINATIONS[3:] == ("Rules infraction", "Unterminated")
# fb_shard_fill() buffers: string offsets, string bytes, string validity, then the
# scalar and list columns (offsets at 34, 36, 41)
FILL = "qqqqqqqBBBBBBBBBBBBBhhhhBBbbBihbbqqHqIQQbqhBi"
P = ctypes.c_void_p
i8, i16, i32, i64 = ctypes.c_int8, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64
u8, u32, u64 = ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint64


class Col(ctypes.Structure):
    _fields_ = [("valid", P), ("voff", i64), ("off", P), ("data", P)]


class Game(ctypes.Structure):
    _fields_ = [
        ("s", P * 7),
        ("len", u32 * 7),
        ("nulls", u32),
        ("rated", u8),
        *((k, i8) for k in ("fmt", "kind", "term", "result", "end")),
        *((k, i16) for k in ("welo", "belo", "wdiff", "bdiff", "inc")),
        ("base", i32),
        ("utc", i64),
        ("move_hash", u64),
        ("token_hash", u64),
        *((k, P) for k in ("mv", "clk", "ev")),
        *((k, u32) for k in ("nm", "nc", "ne")),
    ]


def lib():
    """fastbuild.c compiled once per source version and node, like modded_board's."""
    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()[:16]
    default = f"/scratch/{os.environ['USER']}/allie/fastbuild"
    cache = Path(os.environ.get("ALLIE_FASTBUILD_CACHE", default)) / sha
    so = cache / "fastbuild.so"
    if not so.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / f"fastbuild-{os.getpid()}.so"
        cc = ["gcc", "-O3", "-march=x86-64-v2", "-std=gnu11", "-fPIC", "-shared"]
        cmd = [*cc, str(SRC), "-o", str(tmp), "-lm", "-l:libzstd.so.1"]
        subprocess.run(cmd, check=True)
        tmp.replace(so)
    L = ctypes.CDLL(str(so))
    for name, res, args in (
        ("fb_init", None, [ctypes.c_char_p, P, ctypes.c_long]),
        ("fb_run_new", P, []),
        ("fb_run_free", None, [P]),
        ("fb_skip", None, [P, ctypes.c_int]),
        ("fb_parse", ctypes.c_long, [P, P, ctypes.c_long, ctypes.c_long]),
        ("fb_append", None, [P, ctypes.POINTER(Game)]),
        ("fb_run_info", None, [P, P]),
        ("fb_run_buf", P, [P]),
        ("fb_record", u64, [P, u64, ctypes.POINTER(Game)]),
        ("fb_spill", ctypes.c_long, [P, ctypes.c_int, P, ctypes.c_int, P]),
        ("fb_load", P, [ctypes.c_int, P, ctypes.c_long, P, P, P]),
        ("fb_bucket_free", None, [P]),
        ("fb_shard_size", None, [P, ctypes.c_long, ctypes.c_long, P]),
        ("fb_shard_fill", None, [P, ctypes.c_long, ctypes.c_long, ctypes.c_int, P]),
    ):
        fn = getattr(L, name)
        fn.restype, fn.argtypes = res, args
    return L


L = lib()


def init(val=None):
    """Move table, and the sorted validation token hashes (kept alive by the caller)."""
    val = np.zeros(0, np.uint64) if val is None else val
    L.fb_init(" ".join(MOVES).encode(), val.ctypes.data, len(val))
    return val


def columns(batch):
    """C view of one record batch (the batch must outlive its use)."""
    out = (Col * len(COLS))()
    for k, name in enumerate(COLS):
        if name not in batch.schema.names:
            continue
        a = batch.column(name)
        assert a.type == TYPES[name], (name, a.type)
        b, o = a.buffers(), a.offset
        valid = b[0].address if b[0] is not None and a.null_count else None
        if a.type == pa.string():
            data = b[2].address if b[2] is not None and b[2].size else b[1].address
            out[k] = Col(valid, o, b[1].address + 4 * o, data)
        else:
            out[k] = Col(valid, o, None, b[1].address + a.type.bit_width // 8 * o)
    return out


def append(r, g):
    """A chessdata.record() dict into run r."""
    s = [g[k] for k in STRS]
    enc = [(x or "").encode() for x in s]
    mv, clk, ev = (
        np.array(g[k], t)
        for k, t in (("moves", np.uint16), ("clocks", np.uint32), ("evals", np.int16))
    )
    nulls = sum(1 << k for k, x in enumerate(s) if x is None)
    nulls |= (g["white_diff"] is None) << 7 | (g["black_diff"] is None) << 8
    bufs = [ctypes.create_string_buffer(e, len(e) + 1) for e in enc]
    x = Game(
        s=(P * 7)(*map(ctypes.addressof, bufs)),
        len=(u32 * 7)(*map(len, enc)),
        nulls=nulls,
        rated=g["rated_prefix"],
        fmt=g["format"],
        kind=g["event_kind"],
        term=g["termination"],
        result=g["result"],
        end=g["end"],
        welo=g["white_elo"],
        belo=g["black_elo"],
        wdiff=g["white_diff"] or 0,
        bdiff=g["black_diff"] or 0,
        inc=g["increment"],
        base=g["base"],
        utc=g["utc"],
        move_hash=g["move_hash"],
        token_hash=g["token_hash"],
        mv=mv.ctypes.data,
        clk=clk.ctypes.data,
        ev=ev.ctypes.data,
        nm=len(mv),
        nc=len(clk),
        ne=len(ev),
    )
    L.fb_append(r, ctypes.byref(x))


def reference(batch, i):
    row = batch.slice(i, 1).to_pylist()[0]
    return cd.record(cd.hf_header(row), row["movetext"] or "")


def parse(r, batch):
    """Rows of batch into run r; rows the C core hands back go through record()."""
    c, n, i = columns(batch), batch.num_rows, 0
    while (i := L.fb_parse(r, c, i, n)) < n:
        g, why = reference(batch, i)
        if g is None:
            L.fb_skip(r, {v: k for k, v in SKIPS.items()}[why])
        else:
            append(r, g)
        i += 1


def tasks(files, workers, most=128):
    """(seq, file index, row groups): contiguous row group ranges in store order."""
    groups = [pq.ParquetFile(f).metadata.num_row_groups for f in files]
    step = max(1, min(most, -(-sum(groups) // (4 * workers))))
    out = [
        (i, range(a, min(a + step, g)))
        for i, g in enumerate(groups)
        for a in range(0, g, step)
    ]
    return [(seq, i, rgs) for seq, (i, rgs) in enumerate(out)]


def build(a):
    root = Path(a.out)
    spill = root / "spill"
    shutil.rmtree(spill, ignore_errors=True)
    spill.mkdir(parents=True)
    files = sorted(Path(a.hf).glob("*.parquet"))
    assert files, f"no parquet files in {a.hf}"
    init()
    workers = a.workers or len(os.sched_getaffinity(0))
    fd = os.open(spill / "games.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    at = ctypes.c_uint64(0)

    def run(task):
        seq, i, rgs = task
        pf, r = pq.ParquetFile(files[i]), L.fb_run_new()
        try:
            names = [c for c in COLS if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(8192, rgs, names, use_threads=False):
                parse(r, batch)
            ent, info = np.empty((2100, 5), np.int64), np.empty(7, np.int64)
            m = L.fb_spill(r, fd, ctypes.addressof(at), a.level, ent.ctypes.data)
            assert m >= 0, "spill write failed"
            L.fb_run_info(r, info.ctypes.data)
            return seq, i, ent[:m], info
        finally:
            L.fb_run_free(r)

    with ThreadPoolExecutor(workers) as ex:
        done = list(ex.map(run, tasks(files, workers)))
    os.close(fd)
    # index rows: seq, bucket code, offset, compressed and raw bytes, games
    idx = np.concatenate([np.c_[np.full(len(e), seq), e] for seq, _, e, _ in done])
    np.save(spill / "index.npy", idx[np.lexsort((idx[:, 0], idx[:, 1]))])
    np.save(spill / "runs.npy", np.array([info[0] for *_, info in done], np.int64))
    with open(root / "build-log.jsonl", "w") as log:
        for i, f in enumerate(files):
            games, bad = 0, {}
            for info in (info for _, j, _, info in done if j == i):
                games += int(info[0])
                for kind in info[4 : 4 + info[3]]:  # bad keys in first-seen order
                    bad.setdefault(SKIPS[int(kind)], 0)
                for kind, name in SKIPS.items():
                    if info[kind]:
                        bad[name] += int(info[kind])
            line = dict(chunk=i, source=f.name, games=games, bad=bad)
            log.write(json.dumps(line) + "\n")
    parts = dict(parts=len(files), chunks=len(files))
    (root / "build-complete.json").write_text(json.dumps(parts) + "\n")


def schema():
    """The shard schema chessdata.finalize writes."""
    f = [
        x.with_type(pa.large_string())
        if x.type == pa.string()
        else x.with_type(pa.large_list(x.type.value_type))
        if pa.types.is_list(x.type)
        else x
        for x in cd.SCHEMA
    ]
    s = pa.schema([*f, ("val_leak", pa.bool_()), ("bucket", pa.int32())])
    return s.with_metadata({b"annotate": json.dumps(dict(end=1, evals=1)).encode()})


SHARD = schema()


def shard(b, code, lo, n, path):
    """Rows [lo, lo + n) of a loaded bucket as one shard, written like finalize()."""
    sz = np.empty(10, np.int64)
    L.fb_shard_size(b, lo, n, sz.ctypes.data)
    size = dict.fromkeys((*range(7), 34, 36, 41), n + 1)
    size |= dict(zip((*range(7, 14), 35, 37, 42), sz.tolist()))
    o = [np.empty(size.get(k, n), t) for k, t in enumerate(FILL)]
    L.fb_shard_fill(b, lo, n, code, (P * len(o))(*(x.ctypes.data for x in o)))

    def validity(v):
        nulls = n - int(v.sum())
        bits = np.packbits(v, bitorder="little")
        return (pa.py_buffer(bits) if nulls else None), nulls

    def string(k):
        vb, nulls = validity(o[13 + k]) if k else (None, 0)
        buf = pa.py_buffer(o[k]), pa.py_buffer(o[7 + k])
        return pa.LargeStringArray.from_buffers(n, *buf, vb, nulls)

    def int16(k):
        vb, nulls = validity(o[k + 2])
        return pa.Array.from_buffers(pa.int16(), n, [vb, pa.py_buffer(o[k])], nulls)

    def lst(k):
        return pa.LargeListArray.from_arrays(pa.array(o[k]), pa.array(o[k + 1]))

    cols = [
        *map(string, range(3)),
        *map(pa.array, o[20:22]),
        int16(22),
        int16(23),
        string(3),
        string(4),
        *map(pa.array, (o[26], o[27], o[28].view(np.bool_), *o[29:34])),
        string(5),
        string(6),
        lst(34),
        lst(36),
        *map(pa.array, o[38:41]),
        lst(41),
        pa.array(o[43].view(np.bool_)),
        pa.array(o[44]),
    ]
    table = pa.Table.from_arrays(cols, schema=SHARD)
    pq.write_table(table, path, compression="zstd")


def finalize(a):
    root = Path(a.out)
    assert (root / "build-complete.json").exists(), "build not complete"
    spill = root / "spill"
    idx, runs = np.load(spill / "index.npy"), np.load(spill / "runs.npy")
    base = np.r_[0, np.cumsum(runs)[:-1]].astype(np.int64)
    rand = np.random.default_rng(a.seed).random(int(runs.sum()))
    val = init(np.unique(cd.val_token_hashes()))
    games = root / "games.new"  # published by rename once complete
    shutil.rmtree(games, ignore_errors=True)
    stats = np.zeros(418, np.int64)
    fd = os.open(spill / "games.bin", os.O_RDONLY)
    codes, starts = np.unique(idx[:, 1], return_index=True)
    buckets, live, used = [], deque(), 0

    with ThreadPoolExecutor(a.workers or len(os.sched_getaffinity(0))) as ex:

        def load(code, n, ent):
            """One bucket into memory, then its shard writes queued behind it."""
            st = np.zeros(418, np.int64)
            ptr = [x.ctypes.data for x in (ent, base, rand, st)]
            b = L.fb_load(fd, ptr[0], len(ent), *ptr[1:])
            per = a.shard_games
            futs = [
                ex.submit(shard, b, code, lo, min(per, n - lo), games / name)
                for lo, name in zip(range(0, n, per), names(code, n, per))
            ]
            return b, st, futs

        def release():
            nonlocal used, stats
            size, fut = live.popleft()
            b, st, futs = fut.result()
            for f in futs:
                f.result()
            L.fb_bucket_free(b)
            stats += st
            used -= size

        for code, s, e in zip(map(int, codes), starts, [*starts[1:], len(idx)]):
            # fb_load rows: seq, offset, clen, rlen, games, code
            ent = np.ascontiguousarray(np.c_[idx[s:e, 0], idx[s:e, 2:], idx[s:e, 1]])
            size, n = int(ent[:, 3].sum()), int(ent[:, 4].sum())
            while live and used + size > a.mem_gb << 30:
                release()
            (games / f"b{code:05d}").mkdir(parents=True)
            live.append((size, ex.submit(load, code, n, ent)))
            used += size
            buckets.append(
                dict(
                    code=code,
                    format=code // 10000 - 1,
                    max_elo_bin=code // 100 % 100 * 100,
                    min_elo_bin=code % 100 * 200,
                    games=n,
                    shards=[f"games/{x}" for x in names(code, n, a.shard_games)],
                )
            )
        while live:
            release()
    os.close(fd)
    del val
    old = root / "games.old"
    if (root / "games").exists():
        (root / "games").rename(old)
    games.rename(root / "games")
    shutil.rmtree(old, ignore_errors=True)
    out = summary(stats, root)
    (root / "buckets.json").write_text(json.dumps(buckets, indent=1) + "\n")
    text = json.dumps(out | dict(buckets=len(buckets)), indent=2)
    (root / "stats.json").write_text(text + "\n")
    print(json.dumps(out, indent=2))


def names(code, n, per):
    return [f"b{code:05d}/shard-{k:04d}.parquet" for k in range(-(-n // per))]


def summary(st, root):
    """chessdata.summarize() from the C tallies."""
    bad = {}
    for line in (root / "build-log.jsonl").read_text().splitlines():
        for k, v in json.loads(line)["bad"].items():
            bad[k] = bad.get(k, 0) + v
    st = [int(x) for x in st]
    return dict(
        games=st[0],
        plies=st[1],
        tokens=st[1] + 12 * st[0],
        skipped=bad,
        games_by_format={f: st[2 + i] for i, f in enumerate((*cd.FORMATS, "other"))},
        games_by_max_elo_100={k * 100: v for k, v in enumerate(st[18:]) if v},
        expert2400_moves_by_mover=st[9],
        expert2400_moves_avg_elo_rule=st[10],
        expert2400_moves_in_avg_elo_below_2400=st[11],
        expert2400_bot_moves=st[12],
        expert2400_moves_rating_diff_ge_25=st[13],
        expert2400_moves_rating_diff_ge_50=st[14],
        games_with_bot=st[15],
        games_without_clocks=st[16],
        val_leak_games=st[17],
    )


def decode(g):
    """A C record as the chessdata.record() dict."""

    def arr(p, n, t):
        return (
            np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(t)), (n,)).tolist()
            if n
            else []
        )

    s = [
        None if g.nulls >> k & 1 else ctypes.string_at(g.s[k], g.len[k]).decode()
        for k in range(7)
    ]
    return dict(zip(STRS, s)) | dict(
        white_elo=g.welo,
        black_elo=g.belo,
        white_diff=None if g.nulls >> 7 & 1 else g.wdiff,
        black_diff=None if g.nulls >> 8 & 1 else g.bdiff,
        format=g.fmt,
        event_kind=g.kind,
        rated_prefix=bool(g.rated),
        base=g.base,
        increment=g.inc,
        termination=g.term,
        result=g.result,
        utc=g.utc,
        moves=arr(g.mv, g.nm, ctypes.c_uint16),
        clocks=arr(g.clk, g.nc, ctypes.c_uint32),
        move_hash=g.move_hash,
        token_hash=g.token_hash,
        end=g.end,
        evals=arr(g.ev, g.ne, ctypes.c_int16),
    )


def check(a):
    """The C core against record() on a.groups random row groups per file; rows where
    record() raises must be handed back to Python."""
    init()
    rng, info, g = np.random.default_rng(a.seed), np.empty(7, np.int64), Game()
    seen = mismatched = handed = 0
    for f in sorted(Path(a.hf).glob("*.parquet"))[: a.files]:
        pf = pq.ParquetFile(f)
        k = pf.metadata.num_row_groups
        rgs = sorted(rng.choice(k, min(a.groups, k), replace=False))
        for batch in pf.iter_batches(8192, rgs, COLS, use_threads=False):
            c, r, off = columns(batch), L.fb_run_new(), 0
            for i in range(batch.num_rows):
                try:
                    want = reference(batch, i)
                except Exception as e:
                    want = ("raises", type(e).__name__)
                L.fb_run_info(r, info.ctypes.data)
                n, bad = info[0], info[1:3].copy()
                if L.fb_parse(r, c, i, i + 1) == i:
                    handed += 1
                    continue
                L.fb_run_info(r, info.ctypes.data)
                if info[0] > n:
                    off = L.fb_record(L.fb_run_buf(r), off, ctypes.byref(g))
                    got = (decode(g), None)
                else:
                    got = (None, SKIPS[1 + int(np.flatnonzero(info[1:3] - bad)[0])])
                seen += 1
                if got != want:
                    mismatched += 1
                    if mismatched <= 5:
                        print("MISMATCH", f.name, batch.column("Site")[i], got, want)
            L.fb_run_free(r)
    print(json.dumps(dict(rows=seen, mismatched=mismatched, handed_to_python=handed)))
    sys.exit(1 if mismatched else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument(
        "--hf", required=True, help="Lichess/standard-chess-games parquet dir"
    )
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=0)
    b.add_argument("--level", type=int, default=1, help="zstd level of the spill")
    f = sub.add_parser("finalize")
    f.add_argument("--out", required=True)
    f.add_argument("--shard-games", type=int, default=50000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--workers", type=int, default=0)
    f.add_argument("--mem-gb", type=int, default=6, help="loaded buckets at once")
    c = sub.add_parser("check")
    c.add_argument("--hf", required=True)
    c.add_argument("--files", type=int, default=2)
    c.add_argument("--groups", type=int, default=5, help="row groups per file")
    c.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dict(build=build, finalize=finalize, check=check)[a.cmd](a)
