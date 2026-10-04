"""Cross-check our batched Maia-3 scorer against the authors' own UCI engine.

Feeds one reconstructed position to `maia3.uci --use_uci_history` and compares its
MultiPV move probabilities with the ones our scorer computes for the same position.
"""
import argparse
import sys

import chess
import numpy as np
import torch

from allie import paths
from allie.data.vocab import MOVES
from allie.eval.maia3.score_maia3 import load

sys.path.insert(0, str(paths.MAIA3_REPO))
from maia3.dataset import tokenize_board  # noqa: E402
from maia3.utils import get_all_possible_moves, mirror_move  # noqa: E402

DATA = paths.DATA / "maia3-bench"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="maia3-5m")
    p.add_argument("--index", type=int, default=12345)
    a = p.parse_args()
    with np.load(DATA / "games.npz") as z:
        moves, off, meta, sel, keep = (z[k] for k in ("moves", "offsets", "meta", "sel", "keep"))
    g, ply = int(sel[keep[a.index], 0]), int(sel[keep[a.index], 1])
    uci = [MOVES[k] for k in moves[off[g] : off[g + 1]]]
    board, hist = chess.Board(), []
    for u in uci[:ply]:
        hist.append(tokenize_board(board))
        board.push(chess.Move.from_uci(u))
    hist.append(tokenize_board(board))
    w, b = int(meta[g, 2]), int(meta[g, 3])
    self_elo, oppo_elo = (w, b) if board.turn else (b, w)

    model, cfg, _, _ = load(a.model, 4)
    h = hist[-cfg.history :]
    t = torch.cat(h, dim=1)
    if len(h) < cfg.history:
        t = torch.cat([h[0].repeat(1, cfg.history - len(h)), t], dim=1)
    vocab = {m: i for i, m in enumerate(get_all_possible_moves())}
    mask = torch.zeros(len(vocab), dtype=torch.bool)
    for lm in board.legal_moves:
        mask[vocab[lm.uci() if board.turn else mirror_move(lm.uci())]] = True
    with torch.no_grad():
        logits, _, _ = model(t[None], torch.tensor([self_elo]), torch.tensor([oppo_elo]))
    pr = torch.softmax(logits[0].float().masked_fill(~mask, float("-inf")), 0)
    top = torch.topk(pr, 5)
    mine = {}
    for prob, i in zip(top.values.tolist(), top.indices.tolist()):
        u = [k for k, v in vocab.items() if v == i][0]
        mine[u if board.turn else mirror_move(u)] = prob

    # their own engine object, driven exactly as the UCI loop drives it
    from maia3.uci import Maia3UCIEngine as Engine, parse_args
    eng = Engine(parse_args(["--model", a.model, "--use_uci_history", "--device", "cpu",
                             "--temperature", "0"]))
    eng.ensure_model_loaded()
    eng.self_elo, eng.oppo_elo, eng.multipv = self_elo, oppo_elo, 5
    eng.cmd_position("position startpos moves " + " ".join(uci[:ply]))
    _, top_moves = eng.score_moves()
    theirs = {it["move"].uci(): round(it["policy"], 3) for it in top_moves}

    print("game", g, "ply", ply, "fen", board.fen())
    print("self/oppo elo", self_elo, oppo_elo)
    print("ours  ", {k: round(v, 3) for k, v in mine.items()})
    print("engine", theirs)


if __name__ == "__main__":
    main()
