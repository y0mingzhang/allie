"""Tree reuse between moves on live games: at each of White's turns from --start, the coverage search (budget
--budget) then the KL search, each with reuse on (treers' Coverage(reuse=True) / KL(reuse=True): the last tree
re-rooted at the grandchild the game reached) and off, on one game at a time; then the game's actual two plies
are played. Reports per searcher the reused fraction (kept evaluated leaves / budget: mean, median, p10, p90, the
share of turns that started fresh) and the search time per move off vs on. The Rust backend (the all-Rust loop
through the Server); a CPU job, never the login node.

usage: reuse_fraction.py --model DIR --pgns 'GLOB' [--games 20] [--start 10] [--turns 12] [--budget 128]
       [--threads 16] [--out FILE]
"""

import argparse
import glob
import json
import sys
import time

import numpy as np
import torch

from allie.lichess import treers
from allie.lichess.engine import Engine, Game
from allie.lichess.model import Model

from search_profile import read_pgn


def advance(game, g, k):
    """The game brought to ply k with the PGN's clocks, one move at a time (as search_golden.load_game)."""
    moves, clocks = g["moves"], g["clocks"]
    for j in range(len(game.moves), k):
        own, other = clocks[j], clocks[j - 1] if j else None
        wt, bt = (own, other) if j % 2 == 0 else (other, own)
        game.update(moves[: j + 1], wt, bt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pgns", required=True)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--start", type=int, default=10)
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(1)
    m = Model(a.model, "cpu", torch.bfloat16, None, True, "rust", a.threads)
    engine = Engine(m)
    searchers = {
        name: [cls(reuse=False), cls(reuse=True)]
        for name, cls in (("coverage", treers.Coverage), ("kl", treers.KL))
    }
    rows = {
        name: [] for name in searchers
    }  # per turn: reused, evaluated, kept_pulls, ms off, ms on
    files = [
        f
        for f in sorted(glob.glob(a.pgns))
        if len(read_pgn(f, 10**6)["moves"]) >= a.start + 2 * a.turns + 1
    ]
    t0 = time.perf_counter()
    for f in files[: a.games]:
        g = read_pgn(f, 10**6)
        game = Game(engine, g["white"], g["black"], g["base"], g["inc"], g["speed"])
        for turn in range(a.turns):
            k = a.start + 2 * turn  # White to move
            advance(game, g, k)
            if game.board.is_game_over() or game.board.legal_moves.count() < 2:
                break
            for name, (off, on) in searchers.items():
                ms = {}
                for s, label in ((off, "off"), (on, "on"))[
                    :: 1 if turn % 2 == 0 else -1
                ]:
                    t1 = time.perf_counter()
                    res = s(game, a.budget)
                    ms[label] = 1000 * (time.perf_counter() - t1)
                    assert res is not None
                    if label == "on":
                        st = game.last_search
                rows[name].append(
                    dict(
                        reused=st["reused"],
                        evaluated=st["evaluated"],
                        kept_pulls=st.get("kept_pulls", st["reused"]),
                        off=ms["off"],
                        on=ms["on"],
                    )
                )
        print(
            json.dumps(
                dict(
                    game=g["site"],
                    turns=len(rows["coverage"]),
                    seconds=round(time.perf_counter() - t0, 1),
                )
            ),
            flush=True,
        )
    out = dict(
        model=a.model,
        budget=a.budget,
        threads=m.fast.threads,
        games=min(a.games, len(files)),
        start=a.start,
        turns=a.turns,
    )
    for name, r in rows.items():
        frac = np.array([x["reused"] for x in r]) / a.budget
        later = (
            np.array([x["reused"] for x in r[1:]]) / a.budget if len(r) > 1 else frac
        )  # the first turn of a game is always fresh
        q = lambda v: dict(
            mean=round(float(v.mean()), 3),
            median=round(float(np.median(v)), 3),
            p10=round(float(np.percentile(v, 10)), 3),
            p90=round(float(np.percentile(v, 90)), 3),
        )  # noqa: E731
        out[name] = dict(
            turns=len(r),
            reused_fraction=q(frac),
            reused_fraction_after_first=q(later),
            fresh_starts=int((frac == 0).sum()),
            leaves_evaluated=dict(
                off=a.budget, on=round(float(np.mean([x["evaluated"] for x in r])), 1)
            ),
            ms_per_move=dict(
                off=q(np.array([x["off"] for x in r])),
                on=q(np.array([x["on"] for x in r])),
            ),
        )
    out["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(out) + "\n")
    engine.close()


if __name__ == "__main__":
    sys.exit(main())
