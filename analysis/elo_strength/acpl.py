"""Move quality per player, independent of any rating scale: Stockfish 19 at fixed depth scores
every position of a sample of games; a move's loss is the eval before it minus the eval after it,
both from the mover's view, in centipawns capped at 1000 (mate = 1000).

python acpl.py DIR [--per 100 --depth 12 --workers 12 --anchors 1320,...] [--out acpl.json]
"""

import argparse
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import numpy as np

from fit import load
from play import stockfish

CAP = 1000


def evals(moves, depth, local):
    """Eval of each position (before each move and after the last) from its side to move."""
    if getattr(local, "sf", None) is None:
        local.sf = stockfish()
    board, out = chess.Board(), []
    for k in range(len(moves) + 1):
        if board.is_checkmate():
            out.append(-CAP)
        elif board.is_game_over(claim_draw=True):
            out.append(0)
        else:
            cp = local.sf.go(moves[:k], depth=depth)[1]
            out.append(max(-CAP, min(CAP, cp)))
        if k < len(moves):
            board.push_uci(moves[k])
    return out


def losses(game, depth, local, book=6):
    """(mover, centipawn loss, position undecided) for each move after the book."""
    moves = game["moves"].split()
    e = evals(moves, depth, local)
    return [
        (
            game["white"] if k % 2 == 0 else game["black"],
            max(0, e[k] + e[k + 1]),
            abs(e[k]) < 500,
        )
        for k in range(book, len(moves))
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--per", type=int, default=100, help="games sampled per player")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument(
        "--anchors",
        default="1320,1700,2100,2500,2900",
        help="opponents (round 1's: all players)",
    )
    p.add_argument("--out", default="acpl.json")
    a = p.parse_args()
    games = load(a.dir)
    by = defaultdict(list)
    common = {f"sf-{e}" for e in a.anchors.split(",")}
    for g in games:
        for c, o in (("white", "black"), ("black", "white")):
            if g[o] in common and not g[c].startswith("sf-") and g["opening"] < 10:
                by[g[c]].append(g)
    rng = random.Random(0)
    sample = {
        g["id"]: g for gs in by.values() for g in rng.sample(gs, min(a.per, len(gs)))
    }
    local = threading.local()
    with ThreadPoolExecutor(a.workers) as pool:
        rows = [
            r
            for rs in pool.map(lambda g: losses(g, a.depth, local), sample.values())
            for r in rs
        ]
    out = {}
    for player in sorted(by):
        x = np.array([(loss, live) for who, loss, live in rows if who == player], float)
        if not len(x):
            continue
        live = x[x[:, 1] == 1, 0]  # positions not already decided (|eval| < 500)
        out[player] = dict(
            moves=len(x),
            acpl=float(x[:, 0].mean()),
            acpl_live=float(live.mean()),
            blunders=float((live >= 300).mean()),
            mistakes=float(((live >= 100) & (live < 300)).mean()),
        )
        print(f"{player:16s} moves {len(x):5d}  acpl {out[player]['acpl']:5.0f}  live {out[player]['acpl_live']:5.0f}  "
              f"blunders {100 * out[player]['blunders']:4.1f}%  mistakes {100 * out[player]['mistakes']:4.1f}%")  # fmt: skip
    Path(a.dir, a.out).write_text(
        json.dumps(dict(depth=a.depth, games=len(sample), players=out), indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
