"""Training-time mixing over the structured Lichess store (data-v1).

A policy maps a shard's game columns (and training progress) to a sampling weight and per-side
loss masks. The sampler draws buckets with probability proportional to games x weight cap, takes
each drawn bucket's next pre-shuffled games and accepts each with probability weight / cap, so
games are drawn exactly in proportion to their weight. Accepted games are shuffled, tokenized on
the fly and packed into 1025-token rows like the original corpus: the overflowing game is truncated.
"""

import copy
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chess_vocab import BOS, INCREMENTS_ID, SECONDS_ID, TERM_NORMAL, TERM_OTHER, UNK

STORE = Path("/data/group_data/dei-group/yimingz3/allie/data-v1")
RECIPES = Path("/data/group_data/dei-group/yimingz3/allie/results/recipe10x/recipes")
COLUMNS = [
    "moves",
    "white_elo",
    "black_elo",
    "white_title",
    "black_title",
    "format",
    "rated_prefix",
    "base",
    "increment",
    "termination",
    "val_leak",
]
ROW = 1025
SHARD_GAMES = 50000  # chessdata.py finalize --shard-games
CLOCK_COLUMNS = ["clocks", "token_hash"]
AUX_COLUMNS = ["result"]
POOL = dict(
    tokens=np.int64,
    mask=bool,
    time=np.int16,
    wdl=np.int8,
    feat=np.int32,
)


def clock_bucket(seconds):
    """0 = no clock; 1..16 = exact seconds 0..15 left on the mover's clock, where bullet
    scrambles live; 17..63 = log-spaced (~15% wide) from 16 s up to 3 h and beyond."""
    s = np.asarray(seconds, np.float64)
    log = np.minimum(63, 17 + np.round(7.06 * np.log(np.maximum(s, 16) / 16)))
    return np.where(s < 16, 1 + s, log).astype(np.int16)


# chess-v2 process_hf keep ratios by (format, 100-point average-Elo bucket); missing buckets keep 1.
KEEP_TABLE = {
    1: (
        400,
        [
            1,
            1,
            1,
            0.35023249,
            0.13953627,
            0.06992065,
            0.04321353,
            0.03173244,
            0.02422802,
            0.01970780,
            0.01694768,
            0.01484727,
            0.01369574,
            0.01344519,
            0.01384097,
            0.01534650,
            0.01851822,
            0.02520809,
            0.03838693,
            0.06063675,
            0.10931648,
            0.21127398,
            0.39336253,
            0.73892405,
        ],
    ),
    2: (
        400,
        [
            1,
            1,
            1,
            0.72235112,
            0.30747959,
            0.16453221,
            0.10575061,
            0.08115952,
            0.06360318,
            0.05203546,
            0.04565920,
            0.04089603,
            0.03869786,
            0.03841250,
            0.04017861,
            0.04789621,
            0.06477250,
            0.10071058,
            0.18222968,
            0.35476887,
            0.76532284,
        ],
    ),
    3: (
        400,
        [
            1,
            1,
            1,
            1,
            0.80628453,
            0.44433873,
            0.28256792,
            0.21764459,
            0.16804908,
            0.13959675,
            0.12149435,
            0.11385523,
            0.11718799,
            0.13019962,
            0.15212717,
            0.21074483,
            0.35586375,
            0.69701493,
        ],
    ),
    4: (600, [1, 1, 1, 1, 1, 1, 1, 1, 0.99701110, 0.86754598]),
    0: (
        500,
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0.73197492,
            0.46403021,
            0.33870032,
            0.28031212,
            0.28934325,
            0.34767719,
            0.48767753,
            0.67446563,
        ],
    ),
}
KEEP = np.ones((6, 100))
for fmt, (lo, ratios) in KEEP_TABLE.items():
    KEEP[fmt, lo // 100 : lo // 100 + len(ratios)] = ratios


def lookup(x, ids):
    u, inverse = np.unique(x, return_inverse=True)
    return np.array(
        [ids.get(str(v) if v >= 0 else "*" if v == -1 else "?", UNK) for v in u],
        np.int16,
    )[inverse]


class Games:
    """One shard's columns as compact numpy arrays, with precomputed headers."""

    def __init__(self, table, clock=False, aux=False, feats=False):
        moves = table.column("moves").combine_chunks()
        self.off = moves.offsets.to_numpy()
        self.values = moves.values.to_numpy()
        get = lambda k: table.column(k).to_numpy()
        self.welo, self.belo = (
            get("white_elo").astype(np.int32),
            get("black_elo").astype(np.int32),
        )
        self.fmt, self.term = get("format"), get("termination")
        self.rated, self.leak = get("rated_prefix"), get("val_leak")
        bot = lambda k: (
            pc.equal(table.column(k), "BOT")
            .fill_null(False)
            .to_numpy(zero_copy_only=False)
        )
        self.wbot, self.bbot = bot("white_title"), bot("black_title")
        self.n = len(self.welo)
        digits = lambda e: np.stack(
            [e // 1000 % 10, e // 100 % 10, e // 10 % 10, e % 10], 1
        ).astype(np.int16)
        self.head = np.concatenate(
            [
                np.full((self.n, 1), BOS, np.int16),
                lookup(get("base"), SECONDS_ID)[:, None],
                lookup(get("increment"), INCREMENTS_ID)[:, None],
                digits(self.welo),
                digits(self.belo),
            ],
            1,
        )
        if clock or aux or feats:
            c = table.column("clocks").combine_chunks()
            self.coff, self.cval = c.offsets.to_numpy(), c.values.to_numpy()
            self.base, self.inc = get("base"), get("increment")
            self.dropped = (
                get("token_hash") % 10 == 0
            )  # these games train without clocks
        if aux:
            self.result = get("result")

    def clock(self, i, size, drop=True):
        """Time left on the next mover's clock before each move, at the position predicting it:
        the base time for each side's first move, else that side's clock after its previous move.
        0 on other positions and for games without clocks (or dropped)."""
        out = np.zeros(size, np.int16)
        moves = self.off[i + 1] - self.off[i]
        c = self.cval[self.coff[i] : self.coff[i + 1]].astype(np.int64)
        if len(c) == moves and self.base[i] >= 0 and not (drop and self.dropped[i]):
            before = np.concatenate([[self.base[i], self.base[i]], c])[:moves]
            out[10 : 10 + moves] = clock_bucket(before)
        return out

    def aux(self, i, size):
        """Targets at the position predicting each move, -1 elsewhere: the mover's think time as
        clock bucket - 1 (from the third ply, games with clocks) and the result for the mover
        (0 win, 1 draw, 2 loss)."""
        time, wdl = np.full(size, -1, np.int16), np.full(size, -1, np.int8)
        moves, r = self.off[i + 1] - self.off[i], self.result[i]
        if 0 <= r < 3:
            wdl[10 : 10 + moves] = np.array([[0, 2], [2, 0], [1, 1]])[r][
                np.arange(moves) % 2
            ]
        c = self.cval[self.coff[i] : self.coff[i + 1]].astype(np.int64)
        if len(c) == moves > 2 and self.base[i] >= 0:
            spent = c[:-2] - c[2:] + self.inc[i]
            time[12 : 10 + moves] = np.where(
                spent >= 0, clock_bucket(np.maximum(spent, 0)) - 1, -1
            )
        return time, wdl

    def feats(self, i, size, drop=True):
        """Raw seconds at the position predicting each move, -1 = none: the mover's time left,
        the opponent's time left, and the mover's think time on their previous move (from the
        fifth ply; the first move of each side does not tick the clock)."""
        out = np.full((size, 3), -1, np.int32)
        moves = self.off[i + 1] - self.off[i]
        c = self.cval[self.coff[i] : self.coff[i + 1]].astype(np.int64)
        if len(c) == moves and self.base[i] >= 0 and not (drop and self.dropped[i]):
            b = self.base[i]
            out[10 : 10 + moves, 0] = np.concatenate([[b, b], c])[:moves]
            out[10 : 10 + moves, 1] = np.concatenate([[b], c])[:moves]
            if moves > 4:
                prev = c[: moves - 4] - c[2 : moves - 2] + self.inc[i]
                out[14 : 10 + moves, 2] = np.where(prev >= 0, prev, -1)
        return out

    def tokens(self, i, white_mask, black_mask):
        moves = self.values[self.off[i] : self.off[i + 1]].astype(np.int64) + 378
        toks = np.concatenate(
            [
                self.head[i].astype(np.int64),
                moves,
                [TERM_NORMAL if self.term[i] == 0 else TERM_OTHER],
            ]
        )
        mask = np.ones(len(toks), bool)
        mask[11 : 11 + len(moves) : 2] = white_mask
        mask[12 : 11 + len(moves) : 2] = black_mask
        return toks, mask


def keep(g, elo):
    return KEEP[np.clip(g.fmt, 0, 5), np.clip(elo // 100, 0, 99)] * g.rated


def control(g, p):
    return keep(g, (g.welo + g.belo) // 2), True, True


def upsampled(k):
    """control, with games whose stronger player is rated >= 2400 weighted k times."""
    return lambda g, p: (
        control(g, p)[0] * np.where(np.maximum(g.welo, g.belo) >= 2400, k, 1),
        True,
        True,
    )


def relaxed(r):
    """control with the down-sampling of sub-expert games relaxed r times (keep ratio
    min(1, r x keep)): fresh abundant tokens instead of repeating a pass; >= 2400 games
    keep control's weight."""

    def fn(g, p):
        w = control(g, p)[0]
        top = np.maximum(g.welo, g.belo)
        return np.where(top < 2400, np.minimum(1.0, r * w), w), True, True

    return fn


# ext-v1 source digits: pgnmentor, twic, broadcast; tcec, ccrl
OTB, ENGINE = (1, 2, 3), (5, 6, 7)
EXT = (("otb", OTB), ("engine", ENGINE))


def sources(srcs, k):
    """control, with games from the given external sources weighted k times."""
    return lambda g, p: (
        control(g, p)[0] * np.where(np.isin(getattr(g, "src", 0), srcs), k, 1),
        True,
        True,
    )


def dated(before, k):
    """control, with games from Lichess months before `before` (YYYY-MM) weighted k times."""
    assert 0 <= k <= 1, "bucket caps are control's"
    return lambda g, p: (
        control(g, p)[0]
        * (k if g.src == 0 and getattr(g, "month", before) < before else 1),
        True,
        True,
    )


def cooldown(policy, start, before=control):
    """before until training progress start, then policy (list its name in PHASES)."""
    return lambda g, p: before(g, p) if p < start else policy(g, p)


POLICIES = dict(
    control=control,
    natural=lambda g, p: (np.ones(g.n), True, True),
    mover_rule=lambda g, p: (keep(g, np.maximum(g.welo, g.belo)), True, True),
    **{f"up{k}": upsampled(k) for k in (2, 4, 8)},
    **{f"relax{r}": relaxed(r) for r in (2, 3, 4, 8)},
    **{f"{n}_x{k}": sources(s, k) for n, s in EXT for k in (2, 4, 10, 30)},
    **{f"{n}_d{k}": sources(s, 1 / k) for n, s in EXT for k in (2, 3, 10)},
    **{f"no{n}": sources(s, 0) for n, s in EXT},
    pre2108_d2=dated("2021-08", 0.5),
)
# policy -> the training progress marks where its weights change (cooldown policies)
PHASES = {}


def recipe(name):
    """Recipe NAME.json, whose name ends in the first 12 hex digits of its sha256
    (pool_inventory.py recipe writes them): the frozen copy next to a study's source-ours if
    there is one, else RECIPES."""
    frozen = Path(__file__).resolve().parents[1] / "recipes" / f"{name}.json"
    raw = (frozen if frozen.exists() else RECIPES / f"{name}.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:12] == name[-12:], f"{name} edited"
    return json.loads(raw)


def table(name):
    """table:NAME, a recipe's weight per bucket code."""
    w = {int(c): v for c, v in recipe(name)["weights"].items()}
    return lambda g, p: (w[g.code], True, True)


COOL = re.compile(r"cool(\d+)(?:\(([^()]+)\))?:(.+)")


def cool(name):
    """coolNN:AFTER or coolNN(BEFORE):AFTER as (NN / 100, BEFORE or None, AFTER); None if not a cooldown name."""
    m = COOL.fullmatch(name)
    if not m:
        assert not name.startswith("cool"), name
        return None
    start, before, after = int(m[1]) / 100, m[2], m[3]
    assert 0 < start < 1 and not any(
        (x or "").startswith("cool") for x in (before, after)
    ), name
    return start, before, after


def resolve(name):
    """A policy name: POLICIES key, table:NAME, or coolNN[(BEFORE)]:AFTER (BEFORE, control by default, until NN% of
    training, then AFTER)."""
    if c := cool(name):
        start, before, after = c
        return cooldown(resolve(after), start, resolve(before) if before else control)
    return (
        table(name.removeprefix("table:"))
        if name.startswith("table:")
        else POLICIES[name]
    )


def marks(name):
    c = cool(name)
    return (0.0, c[0]) if c else PHASES.get(name, (0.0,))


def phase(policy, p):
    return max(x for k in policy.split("+") for x in marks(k) if x <= p)


def compose(name):
    """'a+b+...': policy a, times each later policy's weight relative to control, masks
    and-ed. E.g. mover_rule+up4+noengine+otb_x4."""
    first, *rest = map(resolve, name.split("+"))

    def fn(g, p):
        w, mw, mb = first(g, p)
        w = np.asarray(w, float)
        if rest:
            base = np.asarray(control(g, p)[0], float)
            # factors are relative to control, so games control drops cannot be re-weighted
            assert not ((base == 0) & (np.broadcast_to(w, base.shape) > 0)).any(), name
        for f in rest:
            wk, mwk, mbk = f(g, p)
            w = w * np.divide(
                np.asarray(wk, float), base, out=np.zeros_like(base), where=base > 0
            )
            mw, mb = mw & mwk, mb & mbk
        return w, mw, mb

    return fn


class Grid:
    """Synthetic games spanning a bucket's Elo ranges, to bound the policy weight."""

    def __init__(self, code):
        fmt, hi, lo = code // 10000 % 10 - 1, code // 100 % 100, code % 100
        high = np.arange(
            0 if hi == 6 else hi * 100, 3500 if hi == 30 else hi * 100 + 100, 10
        )
        low = np.arange(
            0 if lo == 3 else lo * 200, 3500 if lo == 14 else lo * 200 + 200, 10
        )
        w, b = [x.ravel() for x in np.meshgrid(high, low)]
        keep = b <= w
        self.welo, self.belo, self.n = w[keep], b[keep], int(keep.sum())
        self.fmt = np.full(self.n, fmt)
        self.code, self.src = code, code // 100000
        self.rated = np.ones(self.n, bool)


LOCAL = pafs.LocalFileSystem()


def read(path, cols):
    """One shard's columns. Opening the file directly skips read_table's per-call filesystem
    resolution and dataset discovery, which dominate the sampler once pools span many shards."""
    with LOCAL.open_input_file(str(path)) as f:
        return pq.ParquetFile(f).read(columns=cols)


class Shard:
    """The resident shard of one bucket plus its policy outputs for the current phase."""

    def __init__(self, path, aux=False, feats=False):
        cols = COLUMNS + CLOCK_COLUMNS * (aux or feats) + AUX_COLUMNS * aux
        self.path, self.g, self.phase = (
            path,
            Games(read(path, cols), aux=aux, feats=feats),
            None,
        )
        self.g.code = int(Path(path).parent.name[1:])
        self.g.src = self.g.code // 100000  # 0 lichess, else ext source
        self.g.month = Path(path).parents[2].name

    def policy(self, fn, ph):
        if self.phase != ph:
            w, mw, mb = fn(self.g, ph)
            n = self.g.n
            self.w = np.where(self.g.leak, 0, np.broadcast_to(w, n))
            self.mw, self.mb, self.phase = (
                np.broadcast_to(mw, n),
                np.broadcast_to(mb, n),
                ph,
            )


class Sampler:
    def __init__(
        self,
        policy,
        seed=42,
        total_rows=1,
        exclude=("2026-07",),
        stores=(STORE,),
        pool_frac=1.0,
        chunk=4096,
        aux=False,
        history=None,
        feats=False,
        months=None,
    ):
        self.init = {
            k: v for k, v in locals().items() if k != "self"
        }  # to rebuild elsewhere
        self.aux, self.last = aux, {}
        self.feats = feats
        self.channels = ["tokens", "mask"] + ["time", "wdl"] * aux + ["feat"] * feats
        self.policy, self.fn, self.total_rows, self.chunk = (
            policy,
            POLICIES[policy] if policy in POLICIES else compose(policy),
            max(1, total_rows),
            chunk,
        )
        months = (
            sorted(months)
            if months
            else sorted(
                str(p.parent)
                for s in stores
                for p in Path(s).glob("20*/stats.json")
                if p.parent.name not in exclude
            )
        )
        self._index(
            months,
            pool_frac,
            history and json.loads(Path(history).read_text())["counts"],
        )
        self.rng = np.random.default_rng(seed)
        self.cursor = {}  # code -> [epoch seed, shard position, row position]
        self.seen = 0
        self.pool = []  # accepted, tokenized games not yet packed, in pop order

    def _index(self, months, pool_frac, history=None):
        """Shards per bucket as (path, rows used). Bucket weights keep the full game counts;
        pool_frac only shrinks each month-bucket to its first pool_frac of pre-shuffled games.
        With history ({code: all-history games}), weights use those counts and each bucket keeps
        pool_frac of its all-history count, spread over its months; buckets whose target exceeds
        the games on disk keep them all and are listed in self.infeasible."""
        self.months, self.pool_frac, self.history = months, pool_frac, history
        buckets = [
            (m, b)
            for m in months
            for b in json.loads((Path(m) / "buckets.json").read_text())
        ]
        games = {}
        for _, b in buckets:
            games[b["code"]] = games.get(b["code"], 0) + b["games"]
        self.missing = 0  # all-history games in buckets absent from the stores
        if history:
            self.missing = sum(v for k, v in history.items() if int(k) not in games)
            scale = {
                c: pool_frac * history.get(str(c), 0) / n for c, n in games.items()
            }
            games = {c: history.get(str(c), 0) for c in games}
        else:
            scale = dict.fromkeys(games, pool_frac)
        self.infeasible = sorted(c for c, f in scale.items() if f > 1)
        self.units = {}
        for m, b in buckets:
            left = math.ceil(min(1, scale[b["code"]]) * b["games"])
            for k, s in enumerate(b["shards"]):
                rows = min(SHARD_GAMES, b["games"] - k * SHARD_GAMES, left)
                if rows <= 0:
                    break
                self.units.setdefault(b["code"], []).append((str(Path(m) / s), rows))
                left -= rows
        self.codes = sorted(self.units)
        self.games = np.array([games[c] for c in self.codes], float)
        self.paths = [p for c in self.codes for p, _ in self.units[c]]
        self.ends = [int(self.games.sum())]
        self.resident, self.weights_phase = {}, None  # code -> Shard
        self.preload, self.ahead = {}, {}  # path -> Shard / future, read ahead of use
        self.ex = getattr(self, "ex", None)

    def _weights(self, p):
        ph = phase(self.policy, p)
        if ph != self.weights_phase:
            self.caps = np.array(
                [np.max(self.fn(Grid(c), ph)[0], initial=0) for c in self.codes]
            )
            total = self.games * self.caps
            self.cdf, self.weights_phase = np.cumsum(total) / total.sum(), ph
        return ph

    def _take(self, k, count, rng, cursor):
        """The next `count` games of bucket k as (path, rows, index array) pieces."""
        code, pieces = self.codes[k], []
        shards = self.units[code]
        while count:
            cur = cursor.setdefault(code, [int(rng.integers(2**63)), 0, 0])
            path, rows = shards[
                np.random.default_rng(cur[0]).permutation(len(shards))[cur[1]]
            ]
            t = min(count, rows - cur[2])
            pieces.append((path, rows, np.arange(cur[2], cur[2] + t)))
            cur[2] += t
            count -= t
            if cur[2] == rows:
                cur[1], cur[2] = cur[1] + 1, 0
                if cur[1] == len(shards):
                    cur[:] = [int(rng.integers(2**63)), 0, 0]
        return pieces

    def _chunk(self, rng, cursor):
        """One chunk of draws as (bucket, pieces), lazily: callers draw each bucket's acceptance
        before the next bucket's take, which fixes the order of rng draws."""
        drawn = np.searchsorted(self.cdf, rng.random(self.chunk), side="right")
        for k, count in zip(*np.unique(drawn, return_counts=True)):
            yield int(k), self._take(int(k), int(count), rng, cursor)

    def _paths(self):
        """Non-resident shards the next chunk opens, found by replaying its draws on copies of the
        rng and cursors."""
        rng, cursor, need = copy.deepcopy(self.rng), copy.deepcopy(self.cursor), {}
        for k, pieces in self._chunk(rng, cursor):
            for path, _, idx in pieces:
                rng.random(len(idx))
                r = self.resident.get(self.codes[k])
                if r is None or r.path != path:
                    need[path] = None
        return list(need)

    def _load(self, paths):
        """Futures reading shards on a shared thread pool (cold NFS reads are latency-bound)."""
        self.ex = self.ex or ThreadPoolExecutor(32)
        return {p: self.ex.submit(Shard, p, self.aux, self.feats) for p in paths}

    def _warm(self):
        """Read the shards the next chunk opens in parallel, reusing reads started ahead of it."""
        need, ahead = self._paths(), self.ahead
        ahead |= self._load([p for p in need if p not in ahead])
        self.preload, self.ahead = {p: ahead[p].result() for p in need}, {}

    def _shard(self, k, path):
        code = self.codes[k]
        if code not in self.resident or self.resident[code].path != path:
            self.resident[code] = self.preload.pop(path, None) or Shard(
                path, self.aux, self.feats
            )
        return self.resident[code]

    def _accepted(self, p):
        """One chunk of candidate draws, returned as accepted tokenized games in random order."""
        ph = self._weights(p)
        self._warm()
        out = []
        for k, pieces in self._chunk(self.rng, self.cursor):
            for path, rows, idx in pieces:
                shard = self._shard(k, path)
                assert rows <= shard.g.n, (path, rows, shard.g.n)
                shard.policy(self.fn, ph)
                ok = idx[self.rng.random(len(idx)) * self.caps[k] < shard.w[idx]]
                for i in ok:
                    game = shard.g.tokens(i, shard.mw[i], shard.mb[i])
                    if self.aux:
                        game += shard.g.aux(i, len(game[0]))
                    if self.feats:
                        game += (shard.g.feats(i, len(game[0])),)
                    out.append(game)
        assert not self.preload, "warm-up replay diverged from the draws"
        out = [out[i] for i in self.rng.permutation(len(out))]
        # start reading the chunk after this one; a hint only, its own replay decides what is used
        self.ahead = self._load(self._paths())
        return out

    def batch(self, n, rank=0, world=1):
        cols = [[] for _ in self.channels]
        for r in range(n * world):
            parts, size = [[] for _ in cols], 0
            while size < ROW:
                if not self.pool:
                    self.pool = self._accepted((self.seen + r) / self.total_rows)
                game = self.pool.pop()
                for p, x in zip(parts, game):
                    p.append(x)
                size += len(game[0])
            for col, p in zip(cols, parts):
                col.append(np.concatenate(p)[:ROW])
        rows, *rest = (np.stack(c) for c in cols)
        self.seen += n * world
        part = slice(rank * n, (rank + 1) * n)
        self.last = {k: v[part] for k, v in zip(self.channels[1:], rest)}
        return rows[part]

    def snapshot(self):
        """Cheap copy of the mutable state; state_dict serializes it."""
        return dict(
            rng=copy.deepcopy(self.rng.bit_generator.state),
            cursor=copy.deepcopy(self.cursor),
            seen=self.seen,
            pool=list(self.pool),
        )

    def state_dict(self, snap=None):
        snap = snap or self.snapshot()
        pool = snap["pool"]
        return dict(
            policy=self.policy,
            months=self.months,
            pool_frac=self.pool_frac,
            history=self.history,
            rng=snap["rng"],
            cursor=snap["cursor"],
            seen=snap["seen"],
            pool_lens=[len(g[0]) for g in pool],
            **{
                f"pool_{k}": np.concatenate([g[j] for g in pool] or [[]]).tolist()
                for j, k in enumerate(self.channels)
            },
        )

    def load_state_dict(self, state):
        if state["policy"] != self.policy:
            raise ValueError("Checkpoint mixing policy differs")
        self._index(
            state["months"], state["pool_frac"], state.get("history")
        )  # resume on the checkpoint's own pool
        self.rng.bit_generator.state = state["rng"]
        self.cursor, self.seen = copy.deepcopy(state["cursor"]), state["seen"]
        cut = np.cumsum(state["pool_lens"])[:-1]
        chans = [
            np.split(np.array(state[f"pool_{k}"], POOL[k]), cut) for k in self.channels
        ]
        self.pool = list(zip(*chans)) if state["pool_lens"] else []


def _produce(init, state, args, keep, q, req, ans):
    """Child process: rebuild the sampler at `state`, stream numbered batches, and answer
    state requests for any recently produced batch number."""
    try:
        s = Sampler(**init)
        s.load_state_dict(state)
        snaps, lock = {}, threading.Lock()

        def serve():
            while True:
                k = req.get()
                with lock:
                    snap = snaps[k]
                ans.put(s.state_dict(snap))

        threading.Thread(target=serve, daemon=True).start()
        for seq in itertools.count(1):
            rows = s.batch(*args)
            with lock:
                snaps[seq] = s.snapshot()
                snaps.pop(seq - keep, None)
            q.put((seq, rows, s.last, s.seen))
    except BaseException as e:  # surface in the training process instead of hanging it
        q.put(e)
        ans.put(e)


class Prefetch:
    """Builds the sampler's batches in a child process, so batch building never competes with
    the training loop for the GIL; same batches and states as the sampler itself."""

    def __init__(self, sampler, depth=256):
        self.sampler, self.depth, self.args = sampler, depth, None
        self.last, self.seq, self._seen = {}, 0, sampler.seen
        self.waited = 0.0  # seconds training blocked on the child

    def batch(self, n, rank=0, world=1):
        if self.args is None:
            self.args = (n, rank, world)
            ctx = mp.get_context("spawn")
            self.q, self.req, self.ans = ctx.Queue(self.depth), ctx.Queue(), ctx.Queue()
            state, keep = self.sampler.state_dict(), 2 * self.depth + 4
            self.proc = ctx.Process(
                target=_produce,
                args=(
                    self.sampler.init,
                    state,
                    self.args,
                    keep,
                    self.q,
                    self.req,
                    self.ans,
                ),
                daemon=True,
            )
            self.proc.start()
        assert self.args == (n, rank, world), "prefetch needs fixed batch arguments"
        t = time.monotonic()
        self.seq, rows, self.last, self._seen = self._get(self.q)
        self.waited += time.monotonic() - t
        return rows

    def _get(self, q):
        """Next item from the child; raises if it failed or died (e.g. OOM-killed) instead of hanging."""
        while True:
            try:
                item = q.get(timeout=10)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(
                        f"prefetch worker died (exit {self.proc.exitcode})"
                    ) from None
        if isinstance(item, BaseException):
            raise RuntimeError("prefetch worker failed") from item
        return item

    @property
    def seen(self):
        return self._seen

    def __getattr__(self, k):
        return getattr(self.sampler, k)

    def state_dict(self):
        """State after the last batch handed to training (the child runs ahead of it)."""
        if self.args is None:
            return self.sampler.state_dict()
        self.req.put(self.seq)
        return self._get(self.ans)

    def load_state_dict(self, state):
        assert self.args is None, "load state before the first batch"
        self.sampler.load_state_dict(state)
        self._seen = self.sampler.seen
