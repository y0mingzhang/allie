"""Offline checks with the real model: bots on the mock server, and per-move latency."""

import copy
import resource
import threading
import time

import numpy as np

from .bot import Bot
from .client import Lichess
from .engine import Engine, Game
from .mock import MockLichess


def selfplay(config, model, search, games, opponent, base, inc, timeout=7200):
    """Bot A (and bot B, or the server's random mover) play `games` games through the
    protocol layer; returns every game's record and the decision latencies."""
    engine = Engine(model)
    config = copy.deepcopy(config)
    config.max_games = max(config.max_games, games)
    tokens = {"token-a": "allie-a"} | (
        {"token-b": "allie-b"} if opponent == "self" else {}
    )
    mock = MockLichess(tokens)
    bots = [Bot(config, Lichess(t, mock.url), engine, search) for t in tokens]
    for b in bots:
        threading.Thread(target=b.run, daemon=True).start()
    while any(b.me is None for b in bots):
        time.sleep(0.05)
    challenger = "allie-b" if opponent == "self" else "random"
    for i in range(games):
        mock.challenge(
            challenger, "allie-a", base, inc, color=("white", "black")[i % 2]
        )
    start = time.monotonic()
    while len(mock.games) < games or any(
        g.status == "started" for g in mock.games.values()
    ):
        if time.monotonic() - start > timeout:
            break
        time.sleep(0.5)
    time.sleep(1)  # let the game threads log their ends
    for b in bots:
        b.stop()
    mock.close()
    for b in bots:
        b.join()
    engine.close()
    records = [
        dict(
            id=g.id,
            white=g.white,
            black=g.black,
            status=g.status,
            winner=g.winner,
            plies=len(g.board.move_stack),
            moves=" ".join(m.uci() for m in g.board.move_stack),
            clock_ms=[int(c) for c in g.clock],
        )
        for g in mock.games.values()
    ]
    ms = [1000 * s for b in bots for st in b.finished.values() for s in st]
    summary = dict(
        games=len(records),
        finished=sum(r["status"] != "started" for r in records),
        statuses={
            s: sum(r["status"] == s for r in records)
            for s in {r["status"] for r in records}
        },
        plies=sum(r["plies"] for r in records),
        rejected_moves=mock.rejected,
        declined=mock.declined,
        decision_ms=percentiles(ms),
        clocks_left_ms=[r["clock_ms"] for r in records],
        forwards=engine.forwards,
        tokens=engine.tokens,
        max_rss_gb=rss(),
    )
    return dict(summary=summary, games=records)


def bench(config, model, search, moves, concurrent=1, base=180, inc=2):
    """Per-move latency as a live game sees it: each side's decision (the opponent's move joins
    the cache, then the choice) and the follow-up that appends the bot's own move."""
    engine = Engine(model)
    play = config.play
    rows, games = [], []

    def one(seed):
        rng = np.random.default_rng(seed)
        elo = play.rating if isinstance(play.rating, int) else 1500
        sides = [Game(engine, elo, elo, base, inc, "blitz", seed + j) for j in range(2)]
        clocks, moves_so_far = [float(base), float(base)], []
        for ply in range(moves):
            me = sides[ply % 2]
            if me.board.is_game_over():
                break
            t0 = time.perf_counter()
            d = me.decide(play, search, clocks[ply % 2])
            t1 = time.perf_counter()
            if ply >= 2:
                spent = (
                    min(d.think, clocks[ply % 2] - 1)
                    if play.think_time
                    else rng.uniform(1, 5)
                )
                clocks[ply % 2] = max(clocks[ply % 2] - spent, 1.0) + inc
            moves_so_far.append(d.move)
            for g in sides:
                g.update(list(moves_so_far), *clocks)
            t2 = time.perf_counter()
            me.sync()
            t3 = time.perf_counter()
            rows.append((ply, 1000 * (t1 - t0), 1000 * (t3 - t2)))
        board = sides[0].board
        games.append(
            dict(moves=" ".join(moves_so_far), result=board.result(claim_draw=True))
        )

    t0 = time.perf_counter()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    engine.close()
    decide = [r[1] for r in rows if r[0] >= 2]
    return dict(
        device=str(model.device),
        dtype=str(model.dtype),
        experts=model.keep,
        threads=__import__("torch").get_num_threads(),
        concurrent=concurrent,
        search=play.search if play.mode == "strongest" else 0,
        decisions=len(rows),
        decide_ms=percentiles(decide),
        first_move_ms=[round(r[1]) for r in rows if r[0] < 2],
        own_move_ms=percentiles([r[2] for r in rows]),
        moves_per_second=len(rows) / wall,
        max_rss_gb=rss(),
        games=games,
    )


def percentiles(x):
    if not x:
        return {}
    q = np.percentile(x, [50, 90, 99])
    return dict(n=len(x), median=round(q[0], 1), p90=round(q[1], 1), p99=round(q[2], 1),
                max=round(max(x), 1))  # fmt: skip


def rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
