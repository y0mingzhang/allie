"""External chess game stores (ext-v1) in the data-v1 store format read by data.mix.

fetch     download one source's raw files sequentially into raw/<source>/, with a manifest
complete  exit 0 once every listed raw file is fetched and every CCRL archive extracted
parse     replay every game into restartable per-input Parquet parts (parts/<source>/)
finalize  dedupe across sources, drop validation / golden-eval leaks, bucket, shuffle, shard
"""

import argparse
import calendar
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from allie import paths
from allie.data import store as cd
from allie.data.fetch import sha256
from allie.data.vocab import BOS, SECONDS, TERM_NORMAL, TERM_OTHER
from allie.data.mix import SHARD_GAMES as SHARD

ROOT = paths.DATA / "ext-v1"
RAW, PARTS = ROOT / "raw", ROOT / "parts"
STRAT = ROOT.parent / "strat-eval-v1/strat.npz"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TCEC_REPO = "https://github.com/TCEC-Chess/tcecgames"
# source: (store, bucket code digit, raw globs); order is the dedupe preference
SOURCES = {
    "broadcast": ("otb", 3, ["broadcast/*.pgn.zst"]),
    "twic": ("otb", 2, ["twic/zips/*.zip"]),
    "pgnmentor": (
        "otb",
        1,
        [
            "pgnmentor/players/*.zip",
            "pgnmentor/openings/*.zip",
            "pgnmentor/events/*.pgn",
        ],
    ),
    "tcec": ("engine", 5, ["tcec/repo/master-archive/*.pgn"]),
    "ccrl4040": ("engine", 6, ["ccrl/4040/*.pgn"]),
    "ccrl404": ("engine", 7, ["ccrl/404/*.pgn"]),
}
# Lichess-equivalent L = a + b * FIDE per Lichess pool (format id): least-squares line through
# the ChessGoals "Lichess to OTB" table rows with FIDE >= 1800 (chessgoals.com/rating-comparison).
POOL = {
    0: (-962.2, 1.5703),
    1: (-962.2, 1.5703),
    2: (-387.6, 1.2506),
    3: (419.4, 0.8828),
    4: (709.0, 0.7132),
    5: (709.0, 0.7132),
}
SEC = np.array([int(s) for s in SECONDS[:-1]])


def cp(pawns):
    return max(-30000, min(30000, round(pawns * 100)))


HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$', re.MULTILINE)
TOKEN = re.compile(r"\{([^}]*)\}|;[^\n]*|([()])|([^\s{}();]+)")
MOVE_NUMBER = re.compile(r"^\d+(?:\.+|…)")
NOISE = re.compile(r"1-0|0-1|1/2-1/2|½-½|\*|\$\d+|[!?]+|[+=/\-±∓∞]+|e\.p\.|[ND]")
CLK = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+)(?:\.\d+)?\]")
TL = re.compile(r"(?:^|[\s,])tl=(\d+)")
WV = re.compile(r"(?:^|[\s,])wv=(-?)([M#]?)(\d+(?:\.\d+)?)")
VARIANT = re.compile(r"(?i)960|fischer|freestyle|random")
UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(h(?:ours?|rs?)?\b|min(?:ute)?s?\.?|mn|''|\"|”|″|'|’|"
    r"sec(?:ond)?s?\.?|sek|s\b|mo?u?ves?|mv|in\b|m\b)?"
)
EXTRA = [
    ("source", pa.string()),
    ("event", pa.string()),
    ("date", pa.string()),
    ("time_control", pa.string()),
    ("white_raw_elo", pa.int16()),
    ("black_raw_elo", pa.int16()),
]
PART_SCHEMA = pa.schema([*cd.SCHEMA, *EXTRA, ("name_key", pa.uint64())])


_core = [f for f in cd.SCHEMA if f.name not in ("end", "evals")]
STORE_SCHEMA = pa.schema(
    [
        cd.large(f)
        for f in [
            *_core,
            pa.field("val_leak", pa.bool_()),
            pa.field("bucket", pa.int32()),
            cd.SCHEMA.field("end"),
            cd.SCHEMA.field("evals"),
            *(pa.field(*e) for e in EXTRA),
        ]
    ]
)


def get(url, referer=None, **kw):
    h = HEADERS | ({"Referer": referer} if referer else {})
    for k in range(4):
        try:
            r = requests.get(url, headers=h, timeout=120, **kw)
            if r.status_code < 500:
                return r
        except requests.RequestException:
            pass
        time.sleep(5 * 2**k)
    return r


def urls(src):
    match src:
        case "pgnmentor":
            html = get("https://www.pgnmentor.com/files.html").text
            pat = r'href="((?:players|openings)/[^"]+\.zip|events/[^"]+\.pgn)"'
            return [
                "https://www.pgnmentor.com/" + h
                for h in dict.fromkeys(re.findall(pat, html))
            ]
        case "twic":
            html = get("https://theweekinchess.com/twic").text
            last = max(map(int, re.findall(r"twic(\d+)g\.zip", html)))
            return [
                f"https://theweekinchess.com/zips/twic{n}g.zip"
                for n in range(920, last + 1)
            ]
        case "broadcast":
            return sorted(
                get("https://database.lichess.org/broadcast/list.txt").text.split()
            )
        case "ccrl":
            base = "https://computerchess.org.uk/"
            out = []
            for l in ("4040", "404"):
                html = get(f"{base}{l}/games.html").text
                out.append(
                    f"{base}{l}/"
                    + re.search(rf'href="(CCRL-{l}\.\[\d+\]\.pgn\.7z)"', html)[1]
                )
            return out


def fetch_tcec():
    repo = RAW / "tcec" / "repo"
    if not (repo / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", TCEC_REPO, str(repo)], check=True
        )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    with open(RAW / "tcec" / "manifest.jsonl", "w") as log:
        for p in sorted((repo / "master-archive").glob("*.pgn")):
            rel = p.relative_to(repo)
            rec = dict(
                url=f"{TCEC_REPO}/blob/{commit}/{rel}",
                file=str(p.relative_to(RAW)),
                bytes=p.stat().st_size,
                sha256=sha256(p),
                status=200,
            )
            log.write(json.dumps(rec) + "\n")


def fetch(a):
    (RAW / a.source).mkdir(parents=True, exist_ok=True)
    if a.source == "tcec":
        return fetch_tcec()
    man = RAW / a.source / "manifest.jsonl"
    done = (
        {json.loads(l)["url"] for l in man.read_text().splitlines()}
        if man.exists()
        else set()
    )
    todo = urls(a.source)
    (RAW / a.source / "urls.txt").write_text("\n".join(todo) + "\n")
    referer = "https://theweekinchess.com/twic" if a.source == "twic" else None
    with open(man, "a") as log:
        for url in todo:
            if url in done:
                continue
            dest = RAW / a.source / url.split("/", 3)[3].removeprefix("broadcast/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            r = get(url, referer, stream=True)
            if r.status_code == 200:
                with open(part, "wb") as f:
                    for b in r.iter_content(1 << 20):
                        f.write(b)
                part.rename(dest)
            rec = dict(url=url, file=str(dest.relative_to(RAW)), status=r.status_code)
            if r.status_code == 200:
                rec |= dict(bytes=dest.stat().st_size, sha256=sha256(dest))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            time.sleep(a.delay)


def extracted(d):
    log = RAW / "ccrl" / d / "extract.log"
    return log.exists() and "Everything is Ok" in log.read_text()


def complete(a):
    missing = {}
    for src in ("pgnmentor", "twic", "broadcast", "ccrl"):
        want = set((RAW / src / "urls.txt").read_text().split())
        man = RAW / src / "manifest.jsonl"
        got = {json.loads(l)["url"] for l in man.read_text().splitlines()}
        if want - got:
            missing[src] = len(want - got)
    missing |= {f"ccrl{d}-extract": 1 for d in ("4040", "404") if not extracted(d)}
    print(json.dumps(missing))
    sys.exit(1 if missing else 0)


def decode(blob):
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return blob.decode("latin-1")


def load(path, lo, hi):
    """Decoded PGN text of one raw file, or of bytes [lo, hi) of a plain PGN."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return "\n\n".join(
                decode(z.read(n)) for n in z.namelist() if n.lower().endswith(".pgn")
            )
    with open(path, "rb") as f:
        if path.suffix == ".zst":
            import zstandard

            r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
            return decode(r.read())
        f.seek(lo)
        return decode(f.read(None if hi is None else hi - lo))


def boundaries(path, size, step=1 << 25):
    cuts = [0]
    with open(path, "rb") as f:
        for t in range(step, size, step):
            f.seek(t)
            if (i := f.read(1 << 20).find(b"\n[Event ")) >= 0 and t + i + 1 > cuts[-1]:
                cuts.append(t + i + 1)
    return cuts + [size]


def tasks():
    """(source, raw path, lo, hi, cost): whole files, big plain PGNs cut at game starts."""
    for src, (_, _, globs) in SOURCES.items():
        if src.startswith("ccrl") and not extracted(src[4:]):
            continue
        for g in globs:
            for p in sorted(RAW.glob(g)):
                n = p.stat().st_size
                if p.suffix == ".pgn" and n > 1 << 26:
                    cuts = boundaries(p, n)
                    yield from ((src, p, a, b, b - a) for a, b in zip(cuts, cuts[1:]))
                else:
                    yield src, p, 0, None, n * (8 if p.suffix == ".zst" else 4)


def part_path(src, path, lo):
    name = path.relative_to(RAW).as_posix().replace("/", "__")
    return PARTS / src / f"{name}@{lo}.parquet"


def walk(movetext):
    """Mainline SAN tokens and the comment text attached to each."""
    sans, notes, depth = [], [], 0
    for note, paren, tok in TOKEN.findall(movetext):
        if paren:
            depth += 1 if paren == "(" else -1
        elif depth > 0:
            continue
        elif tok:
            tok = MOVE_NUMBER.sub("", tok).rstrip("!?")
            if tok and not NOISE.fullmatch(tok):
                sans.append(tok)
                notes.append("")
        elif note and notes:
            notes[-1] += " " + note
    return sans, notes


def minutes(x, unit):
    if unit.startswith("h"):
        return x * 3600
    if unit.startswith(("s", "''", '"', "”", "″")):
        return x
    if unit.startswith(("min", "mn", "m", "'", "’")):
        return x * 60
    return x * 60 if x <= 150 else x


def parse_tc(s, engine):
    """(base, increment) seconds of the first period, or None. OTB relays often write minutes."""
    s = s.lower().replace('\\"', '"').strip().strip('"').strip()
    if not s or s in ("-", "?", "*"):
        return None
    if m := re.fullmatch(r"(?:(\d+)/)?(\d+)(?:\+(\d+(?:\.\d+)?))?(?:[:_].*)?", s):
        moves, t = int(m[1] or 0), int(m[2])
        t = moves if moves > 60 else t  # "5400/40" = seconds / moves
        return (t if engine else minutes(t, "")), float(m[3] or 0)
    if engine:
        return None
    nums = [
        (float(x), u, s[: m.start()].rstrip().endswith("+"))
        for m in UNIT.finditer(s)
        for x, u in [m.groups("")]
        if not u.startswith(("mo", "mv", "in"))
    ]
    if not nums:
        return None
    (x, u, _), rest = nums[0], nums[1:]
    inc = next(
        (
            y
            for y, v, plus in rest
            if v.startswith(("s", "''", '"', "”", "″")) or (plus and not v and y <= 60)
        ),
        0,
    )
    return minutes(x, u), inc


def event_tc(event, kind, engine):
    e = f"{event} {kind}".lower()
    if engine and (m := re.search(r"(\d+)/(\d+)", e)):
        return int(m[2]) * 60, 0  # CCRL 40/15: moves / minutes
    for keys, bi in (
        (("bullet",), (60, 0)),
        (("blitz", "armageddon", "titled tue"), (180, 2)),
        (("rapid",), (900, 10)),
    ):
        if any(k in e for k in keys):
            return bi
    return 5400, 30


def speed(seconds):
    """Lichess speed of an estimated game duration base + 40 * increment."""
    return int(np.searchsorted([30, 180, 480, 1500, 21600], seconds, side="right"))


def rating(v):
    try:
        r = int(str(v).strip())
    except ValueError:
        return None
    return r if 0 < r < 10000 else None


def person(name, engine):
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    n = n.lower()
    if engine:
        return re.sub(r"[^a-z0-9]", "", n)
    n = n.split(",")[0] if "," in n else (n.split() or [""])[-1]
    return re.sub(r"[^a-z]", "", n)


def utc(h):
    if t := cd.utc(h):
        return t
    try:
        y, m, d = (int(x) if x.isdigit() else 1 for x in h["Date"].split("."))
        return calendar.timegm((y, max(1, m), max(1, d), 0, 0, 0)) if y > 1000 else 0
    except (KeyError, ValueError):
        return 0


def termination(t, result, plies):
    t = (t or "").lower()
    if not plies and result < 2:
        return 2  # forfeit before a move
    if not t:
        return 0 if result < 3 else 4
    for i, keys in (
        (1, ("time",)),
        (2, ("abandon", "stalled", "disconnect", "crash", "forfeit")),
        (3, ("illegal", "rules", "infraction")),
        (4, ("unterminated",)),
    ):
        if any(k in t for k in keys):
            return i
    return 0


def clocks_of(notes, base):
    """Seconds left after each ply from %clk or TCEC tl= (ms); TCEC book plies keep the base."""
    out = []
    for n in notes:
        if m := CLK.search(n):
            out.append(3600 * int(m[1]) + 60 * int(m[2]) + int(m[3]))
        elif m := TL.search(n):
            out.append(int(m[1]) // 1000)
        elif n.lstrip().startswith("book"):
            out.append(base)
        else:
            return []
    return out


def evals_of(notes):
    """White-view centipawns after each ply from %eval or TCEC wv= (data.store.eval_list units)."""
    out, seen = [], False
    for n in notes:
        v = cd.EVAL_MISSING
        if m := cd.EVAL.search(n):
            mate, x = m.groups()
            s = -1 if x.startswith("-") else 1
            v = s * (32000 - min(abs(int(x)), 1999)) if mate else cp(float(x))
        elif m := WV.search(n):
            s, mate, x = -1 if m[1] else 1, m[2], float(m[3])
            v = s * (32000 - min(int(x), 1999)) if mate else s * cp(x)
        seen |= v != cd.EVAL_MISSING
        out.append(v)
    return out if seen else []


def record(game, src):
    """One game dict in PART_SCHEMA, or (None, reason)."""
    engine = SOURCES[src][0] == "engine"
    lines = game.split("\n")
    k = next(
        (
            i
            for i, l in enumerate(lines)
            if l.strip() and not l.lstrip().startswith("[")
        ),
        len(lines),
    )
    h = dict(HEADER.findall("\n".join(lines[:k])))
    event = h.get("Event", "")
    if (
        h.get("SetUp") == "1"
        or h.get("FEN")
        or h.get("Variant", "standard").lower() not in ("standard", "chess", "")
        or VARIANT.search(event)
    ):
        return None, "variant"
    if not engine and "BOT" in (h.get("WhiteTitle"), h.get("BlackTitle")):
        return None, "bot"
    raw = rating(h.get("WhiteElo")), rating(h.get("BlackElo"))
    if None in raw:
        return None, "elo"
    sans, notes = walk("\n".join(lines[k:]))
    board, moves = chess.Board(), []
    try:
        for san in sans:
            m = board.parse_san(san)
            moves.append(cd.MOVE_INDEX[m.uci()])
            board.push(m)
    except (ValueError, KeyError):
        return None, "illegal"
    tc = h.get("TimeControl")
    b, i = parse_tc(tc or "", engine) or event_tc(event, h.get("EventType", ""), engine)
    base, inc = int(SEC[np.abs(SEC - b).argmin()]), int(min(180, max(0, round(i))))
    fmt = speed(base + 40 * inc)
    a, s = POOL[fmt]
    elo = raw if engine else [int(min(3999, max(400, round(a + s * r)))) for r in raw]
    title = lambda c: "BOT" if engine else h.get(f"{c}Title") or None
    result = cd.index(cd.RESULTS, h.get("Result"))
    key = f"{person(h.get('White'), engine)}|{person(h.get('Black'), engine)}"
    g = dict(
        site=None,
        white=h.get("White"),
        black=h.get("Black"),
        white_elo=elo[0],
        black_elo=elo[1],
        white_diff=None,
        black_diff=None,
        white_title=title("White"),
        black_title=title("Black"),
        format=fmt,
        event_kind=len(cd.EVENT_KINDS),
        rated_prefix=True,
        base=base,
        increment=inc,
        termination=termination(h.get("Termination"), result, len(moves)),
        result=result,
        utc=utc(h),
        eco=h.get("ECO") or None,
        opening=h.get("Opening") or None,
        moves=moves,
        clocks=clocks_of(notes, base),
        move_hash=cd.hash64(moves),
        end=cd.end_state(board),
        evals=evals_of(notes),
        source=src,
        event=event or None,
        date=h.get("Date") or h.get("UTCDate"),
        time_control=tc,
        white_raw_elo=raw[0],
        black_raw_elo=raw[1],
        name_key=int.from_bytes(
            hashlib.blake2b(key.encode(), digest_size=8).digest(), "little"
        ),
    )
    g["token_hash"] = cd.hash64(cd.tokenize(g))
    return g, None


def parse_task(src, path, lo, hi):
    out = part_path(src, path, lo)
    try:
        blob = load(path, lo, hi).lstrip("﻿").replace("\r\n", "\n")
        rows, bad = {f.name: [] for f in PART_SCHEMA}, {}
        for i, game in enumerate(re.split(r"\n(?=\[Event )", blob)):
            if not game.strip():
                continue
            g, why = record(game, src)
            if g is None:
                bad[why] = bad.get(why, 0) + 1
                continue
            g["site"] = f"{src}:{path.name}:{lo}:{i}"
            for k, v in g.items():
                rows[k].append(v)
        t = pa.table(rows, schema=PART_SCHEMA)
        t = t.replace_schema_metadata({"bad": json.dumps(bad)})
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        pq.write_table(t, tmp, compression="zstd")
        tmp.rename(out)
        return dict(part=out.name, games=len(t), bad=bad)
    except Exception as e:
        return dict(part=out.name, error=repr(e))


def parse(a):
    todo = sorted(
        (t for t in tasks() if not part_path(*t[:3]).exists()), key=lambda t: -t[4]
    )
    print(f"{len(todo)} parse tasks", flush=True)
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    with (
        ProcessPoolExecutor(len(os.sched_getaffinity(0))) as ex,
        open(PARTS / "parse-log.jsonl", "a") as log,
    ):
        futs = [ex.submit(parse_task, *t[:4]) for t in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            if n % 100 == 0 or "error" in r:
                print(n, f"{time.time() - t0:.0f}s", json.dumps(r), flush=True)


def strat_hashes():
    """Token hashes of every complete game in the golden eval rows."""
    out = set()
    for r in np.load(STRAT)["rows"]:
        starts = np.flatnonzero((r == BOS) & (np.r_[r[1:], BOS] != BOS))
        ends = np.flatnonzero((r == TERM_NORMAL) | (r == TERM_OTHER))
        for s in starts:
            if (j := np.searchsorted(ends, s)) < len(ends):
                out.add(cd.hash64(r[s : ends[j] + 1]))
    return np.fromiter(out, np.uint64)


BANDS = ("<1400", "1400-2000", "2000-2400", ">=2400")


def band_table(fmt, top, games=None):
    """Games per format x max-Elo band (data.mix.BAND)."""
    cell = np.searchsorted([1400, 2000, 2400], top, side="right")
    w = np.ones(len(fmt), np.int64) if games is None else games
    return {
        f: {b: int(w[(fmt == i) & (cell == j)].sum()) for j, b in enumerate(BANDS)}
        for i, f in enumerate(cd.FORMATS)
        if ((fmt == i) & (w > 0)).any()
    }


def write_store(t, src, bad, dropped, seed):
    store, digit, _ = SOURCES[src]
    root = ROOT / store / f"20xx-{src}"
    root.mkdir(parents=True, exist_ok=True)
    welo = t["white_elo"].to_numpy().astype(np.int32)
    belo = t["black_elo"].to_numpy().astype(np.int32)
    fmt = t["format"].to_numpy().astype(np.int32)
    hi = np.clip(np.maximum(welo, belo) // 100, 6, 30)
    lo = np.clip(np.minimum(welo, belo) // 200, 3, 14)
    bucket = digit * 100000 + (fmt + 1) * 10000 + hi * 100 + lo
    (root / "build-log.jsonl").write_text(json.dumps(dict(source=src, bad=bad)) + "\n")
    stats = cd.summarize(t, welo, belo, fmt, np.zeros(len(t), bool), root)
    order = np.lexsort((np.random.default_rng(seed).random(len(t)), bucket))
    t, bucket = t.take(pa.array(order)), bucket[order]
    t = t.append_column("val_leak", pa.array(np.zeros(len(t), bool)))
    t = t.append_column("bucket", pa.array(bucket, pa.int32()))
    t = t.select(STORE_SCHEMA.names).cast(STORE_SCHEMA)
    meta = {b"annotate": json.dumps(dict(end=1, evals=1)).encode()}
    games = root / "games.new"
    shutil.rmtree(games, ignore_errors=True)
    starts = np.r_[0, np.flatnonzero(np.diff(bucket)) + 1, len(bucket)]
    buckets = []
    for s, e in zip(starts[:-1], starts[1:]):
        code, shards = int(bucket[s]), []
        (games / f"b{code:05d}").mkdir(parents=True)
        for k, a in enumerate(range(s, e, SHARD)):
            rel = f"games/b{code:05d}/shard-{k:04d}.parquet"
            part = t.slice(a, min(SHARD, e - a)).replace_schema_metadata(meta)
            pq.write_table(part, games / rel.split("/", 1)[1], compression="zstd")
            shards.append(rel)
        buckets.append(
            dict(
                code=code,
                format=code // 10000 % 10 - 1,
                max_elo_bin=code // 100 % 100 * 100,
                min_elo_bin=code % 100 * 200,
                games=int(e - s),
                shards=shards,
                source=src,
            )
        )
    cd.swap(root, games)
    plies = pc.list_value_length(t["moves"]).to_numpy()
    stats |= dict(
        buckets=len(buckets),
        source=src,
        dropped=dropped,
        games_with_clocks=int((pc.list_value_length(t["clocks"]).to_numpy() > 0).sum()),
        games_with_evals=int((pc.list_value_length(t["evals"]).to_numpy() > 0).sum()),
        mean_plies=float(plies.mean()) if len(plies) else 0.0,
        games_by_format_max_elo_band=band_table(fmt, np.maximum(welo, belo)),
    )
    (root / "buckets.json").write_text(json.dumps(buckets, indent=1) + "\n")
    (root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return {str(b["code"]): b["games"] for b in buckets}, stats


def repeats(mask, pref, *keys):
    """Rows of mask whose keys equal a preferred row's (pref: most significant first)."""
    idx = np.flatnonzero(mask)
    o = idx[np.lexsort([k[idx] for k in (*pref[::-1], *keys[::-1])])]
    same = np.ones(max(len(o) - 1, 0), bool)
    for k in keys:
        same &= k[o][1:] == k[o][:-1]
    out = np.zeros(len(mask), bool)
    out[o[1:][same]] = True
    return out


def finalize(a):
    parts = sorted(PARTS.glob("*/*.parquet"))
    bad, tables = {s: {} for s in SOURCES}, []
    for p in parts:
        f = pq.ParquetFile(p)
        for k, v in json.loads(f.schema_arrow.metadata[b"bad"]).items():
            bad[p.parent.name][k] = bad[p.parent.name].get(k, 0) + v
        tables.append(f.read())
    t = pa.concat_tables(tables)
    t = t.cast(pa.schema([cd.large(f) for f in t.schema]))
    print(f"{len(t)} parsed games from {len(parts)} parts", flush=True)
    prio = pc.index_in(t["source"], pa.array(list(SOURCES))).to_numpy()
    plies = pc.list_value_length(t["moves"]).to_numpy()
    clocked = pc.list_value_length(t["clocks"]).to_numpy() > 0
    evald = pc.list_value_length(t["evals"]).to_numpy() > 0
    mh, nk = t["move_hash"].to_numpy(), t["name_key"].to_numpy()
    res = t["result"].to_numpy()
    dates = t["date"].combine_chunks()
    dk = dates.dictionary_encode().indices.fill_null(-1).to_numpy()
    full = (pc.utf8_length(dates).fill_null(0).to_numpy() == 10) & ~pc.match_substring(
        dates, "?"
    ).fill_null(True).to_numpy(zero_copy_only=False)
    # duplicates: same moves and players (normalised surnames); else same moves, full date and
    # result (>= 20 plies: relays spell names differently); else same moves over >= 60 plies
    pref = (prio, ~clocked, ~evald)
    dup = repeats(np.ones(len(t), bool), pref, mh, nk)
    dated = repeats(~dup & full & (plies >= 20), pref, mh, dk, res)
    long = repeats(~dup & ~dated & (plies >= 60), pref, mh)
    keep = ~(dup | dated | long)
    leakset = np.union1d(cd.val_token_hashes(), strat_hashes())
    leak = keep & np.isin(t["token_hash"].to_numpy(), leakset)
    keep &= ~leak
    print(f"leak hashes {len(leakset)}", flush=True)
    counts, summary = {}, dict(sources={})
    for i, src in enumerate(SOURCES):
        m = prio == i
        dropped = bad[src] | dict(
            duplicate=int((dup & m).sum()),
            duplicate_date_result=int((dated & m).sum()),
            duplicate_long_moves_only=int((long & m).sum()),
            leak=int((leak & m).sum()),
        )
        if not (keep & m).any():
            summary["sources"][src] = dict(games=0, dropped=dropped)
            continue
        c, stats = write_store(
            t.filter(pa.array(keep & m)), src, bad[src], dropped, a.seed
        )
        counts |= c
        summary["sources"][src] = dict(
            games=stats["games"],
            store=str(ROOT / SOURCES[src][0] / f"20xx-{src}"),
            dropped=dropped,
            games_with_clocks=stats["games_with_clocks"],
            games_with_evals=stats["games_with_evals"],
            games_by_format_max_elo_band=stats["games_by_format_max_elo_band"],
        )
        print(src, json.dumps(summary["sources"][src]), flush=True)
    (ROOT / "history-counts.json").write_text(
        json.dumps(dict(games=sum(counts.values()), counts=counts), indent=1) + "\n"
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("source", choices=["pgnmentor", "twic", "broadcast", "tcec", "ccrl"])
    f.add_argument("--delay", type=float, default=1.0)
    sub.add_parser("complete")
    sub.add_parser("parse")
    z = sub.add_parser("finalize")
    z.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dict(fetch=fetch, complete=complete, parse=parse, finalize=finalize)[a.cmd](a)
