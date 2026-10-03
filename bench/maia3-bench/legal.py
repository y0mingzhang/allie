"""Ragged legal-move ids (0..1967) and row/position indices for every sampled position."""
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, "/home/yimingz3/src/allie/scripts")
from chess_vocab import MOVES, MOVE_ID  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/group_data/dei-group/yimingz3/allie/maia3-bench")

with np.load(DATA / "games.npz") as z:
    moves, off, meta, sel, keep = (z[k] for k in ("moves", "offsets", "meta", "sel", "keep"))
s = sel[keep]
want = {}
for j, (g, m, *_) in enumerate(s):
    want.setdefault(int(g), []).append((int(m), j))

n = len(s)
pos = np.zeros((n, 2), np.int64)  # row, index of the token that predicts the move
tgt = np.zeros(n, np.int64)
lists, offs = [], [0]
order = []
for g in sorted(want):
    board = chess.Board()
    plies = dict(want[g])
    uci = [MOVES[k] for k in moves[off[g] : off[g + 1]]]
    for m, u in enumerate(uci):
        if m in plies:
            j = plies[m]
            pos[j] = (meta[g, 4], meta[g, 5] + 10 + m)
            tgt[j] = MOVE_ID[u] - 378
            lists.append(np.array([MOVE_ID[x.uci()] - 378 for x in board.legal_moves], np.int16))
            order.append(j)
        board.push(chess.Move.from_uci(u))
order = np.array(order)
inv = np.argsort(order)
lists = [lists[i] for i in inv]
offs = np.cumsum([0] + [len(x) for x in lists])
np.savez(DATA / "legal.npz", pos=pos, target=tgt, legal=np.concatenate(lists), offsets=offs)
print(f"{n} positions, {offs[-1]} legal moves, mean {offs[-1] / n:.1f}")
