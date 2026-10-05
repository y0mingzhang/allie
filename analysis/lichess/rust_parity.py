"""The Rust engine against the C++ kernels on the real model (int8 CPU, as the bot loads it): bitwise
parity of every step's logits and the final caches over live games' positions, then single-token step
timings at several batch sizes and thread counts. Runs on a CPU job, never the login node.

  python analysis/lichess/rust_parity.py --model .../allie-2.0-annealed --pgn '.../live-games/*.pgn' \
      --games 20 --threads 4,8,16 --batches 1,8,24 --out results/lichess-search/rust_parity.jsonl
"""

import argparse
import glob
import json
import os
import platform
import statistics
import time
from types import SimpleNamespace

import chess.pgn
import numpy as np
import torch

from allie.lichess.engine import Game
from allie.lichess.fast import Fast
from allie.lichess.fastrs import RustFast
from allie.lichess.model import Cache, Model
from allie.lichess.tokens import HEADER


def read_pgn(path):
    g = chess.pgn.read_game(open(path))
    h = g.headers
    base, inc = map(int, h["TimeControl"].split("+"))
    speed = next(
        (
            s
            for s in ("bullet", "blitz", "rapid", "classical")
            if s in h.get("Event", "").lower()
        ),
        "rapid",
    )
    moves, clocks, node = [], [], g
    while node.variations:
        node = node.variation(0)
        moves.append(node.move.uci())
        clocks.append(None if node.clock() is None else int(node.clock()))
    return dict(white=int(h["WhiteElo"]), black=int(h["BlackElo"]), base=base, inc=inc, speed=speed,
                moves=moves, clocks=clocks, site=h.get("Site", path))  # fmt: skip


def positions(model, g):
    """(ids, feats, boards) of a game's tokens, as the bot builds them (Game needs only engine.model)."""
    game = Game(
        SimpleNamespace(model=model),
        g["white"],
        g["black"],
        g["base"],
        g["inc"],
        g["speed"],
    )
    moves, clocks = g["moves"], g["clocks"]
    for k in range(1, len(moves) + 1):
        j = k - 1
        own, other = clocks[j], clocks[j - 1] if j else None
        wt, bt = (own, other) if j % 2 == 0 else (other, own)
        game.update(moves[:k], wt, bt)
    return (torch.tensor(game.tokens), torch.tensor(game.features(), dtype=torch.float32),
            torch.tensor(np.frombuffer(b"".join(game.boards), np.uint8).reshape(-1, 68).copy()))  # fmt: skip


def item(cache, x, a, b):
    return (cache, x[0][a:b], x[1][a:b], x[2][a:b])


def diff(a, b):
    return int((a != b).sum())


def compare(a, b, what, report):
    """Counts differing elements; records the first difference."""
    bad = diff(a, b)
    report["elements"] += a.numel()
    report["mismatches"] += bad
    if bad and report["first"] is None:
        i = int((a != b).flatten().nonzero()[0])
        report["first"] = dict(
            what=what,
            index=i,
            cxx=float(b.flatten()[i]),
            rust=float(a.flatten()[i]),
            count=bad,
        )
    return bad


def cache_rows(c):
    return dict(k=c.k[:, :, : c.n], v=c.v[:, :, : c.n], e=c.e[: c.n])


def parity(model, cxx, rust, games, log):
    """Each game: the header as one step, then one token a step, on both engines; every step's logits
    and the final caches compared bitwise. Then all games prefilled in one batched step."""
    report = dict(
        games=len(games), positions=0, steps=0, elements=0, mismatches=0, first=None
    )
    for n, x in enumerate(games):
        caches = Cache(model), Cache(model)
        za = cxx.step([item(caches[0], x, 0, HEADER)])
        zb = rust.step([item(caches[1], x, 0, HEADER)])
        compare(zb, za, f"game {n} header logits", report)
        for p in range(HEADER, len(x[0])):
            a, b = cxx.step([item(caches[0], x, p, p + 1)]), rust.step([item(caches[1], x, p, p + 1)])
            report["steps"] += 1
            compare(b, a, f"game {n} step {p} logits", report)
        report["positions"] += len(x[0])
        for k, (ra, rb) in enumerate(zip(cache_rows(caches[0]).values(), cache_rows(caches[1]).values())):
            compare(rb, ra, f"game {n} cache {'kve'[k]}", report)
        log(f"game {n}: {len(x[0])} positions, {report['mismatches']} mismatching elements so far"
            + (f"; first {report['first']}" if report["first"] else ""))
    # one batched prefill of every game: the multi-token attention path, large token counts
    ca = [Cache(model) for _ in games]
    cb = [Cache(model) for _ in games]
    za = cxx.step([item(c, x, 0, len(x[0])) for c, x in zip(ca, games)])
    zb = rust.step([item(c, x, 0, len(x[0])) for c, x in zip(cb, games)])
    report["steps"] += 1
    compare(zb, za, "batched prefill logits", report)
    for a, b in zip(ca, cb):
        for k, (ra, rb) in enumerate(
            zip(cache_rows(a).values(), cache_rows(b).values())
        ):
            compare(rb, ra, f"batched prefill cache {'kve'[k]}", report)
    return report


def timing(model, games, threads, batches, reps, log):
    """Median ms of a single-token step at each batch size, both engines, each thread count. Each
    repetition steps the same token again after rewinding the caches, so every step does the same work."""
    rows = []
    for nt in threads:
        for name, cls in (("cxx", Fast), ("rust", RustFast)):
            eng = cls(model, threads=nt)
            for b in batches:
                xs = [games[j % len(games)] for j in range(b)]
                caches = [Cache(model) for _ in xs]
                pos = [
                    min(len(x[0]) - 1, HEADER + 30 + j % 7) for j, x in enumerate(xs)
                ]
                eng.step([item(c, x, 0, p) for c, x, p in zip(caches, xs, pos)])
                items = [item(c, x, p, p + 1) for c, x, p in zip(caches, xs, pos)]
                for _ in range(3):
                    for c, p in zip(caches, pos):
                        c.truncate(p)
                    eng.step(items)
                ms = []
                for _ in range(reps):
                    for c, p in zip(caches, pos):
                        c.truncate(p)
                    t0 = time.perf_counter()
                    eng.step(items)
                    ms.append((time.perf_counter() - t0) * 1e3)
                row = dict(engine=name, threads=eng.threads, batch=b, median_ms=statistics.median(ms), min_ms=min(ms),
                           isa=getattr(eng, "isa", None) or eng.lib.allie_isa().decode())  # fmt: skip
                rows.append(row)
                log(
                    f"{name:4s} threads {eng.threads:2d} batch {b:2d}: {row['median_ms']:.2f} ms median ({row['min_ms']:.2f} min)"
                )
            del eng
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--pgn",
        default="/data/group_data/dei-group/yimingz3/allie/lichess/live-games/*.pgn",
    )
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--threads", default="4,8,16")
    ap.add_argument("--batches", default="1,8,24")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--parity-threads", type=int, default=8)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    log(f"{platform.node()} {open('/proc/cpuinfo').read().split('model name')[1].split(chr(10))[0].strip(': ')} "
        f"cpus {len(os.sched_getaffinity(0))}")  # fmt: skip
    t0 = time.time()
    model = Model(
        args.model, "cpu", torch.bfloat16, None, True, "fast", args.parity_threads
    )
    cxx = model.fast
    rust = RustFast(model, threads=args.parity_threads)
    log(
        f"model loaded in {time.time() - t0:.0f}s; cxx isa {cxx.lib.allie_isa().decode()} rust isa {rust.isa}"
    )
    paths = sorted(glob.glob(args.pgn))[: args.games]
    games = [positions(model, read_pgn(p)) for p in paths]
    log(f"{len(games)} games, {sum(len(x[0]) for x in games)} positions")
    report = parity(model, cxx, rust, games, log)
    log(f"parity: {json.dumps(report)}")
    result = dict(
        node=platform.node(),
        cxx_isa=cxx.lib.allie_isa().decode(),
        rust_isa=rust.isa,
        parity=report,
        timing=[],
    )
    if not args.skip_timing:
        del cxx, rust
        model.fast = None
        result["timing"] = timing(model, games, [int(t) for t in args.threads.split(",")],
                                  [int(b) for b in args.batches.split(",")], args.reps, log)  # fmt: skip
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
    log("done")


if __name__ == "__main__":
    main()
