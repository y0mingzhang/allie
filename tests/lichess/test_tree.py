import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from allie.lichess.engine import Engine, Game, Play
from allie.lichess.model import Cache, step
from allie.lichess.tokens import MOVE_ID

from .test_tokens import random_game

tree = pytest.importorskip("allie.lichess.tree")


def test_nodes_match_prefill(tiny):
    game = Game(Engine(tiny), 1500, 1600, 180, 2)
    game.update(random_game(4, 21), 170, 165)
    feats = np.array(game.features(), np.float32)
    nodes = tree.Tree(game, game.sync(), 16).handles([game.tokens], [feats])
    n, tok = len(game.tokens), lambda m: MOVE_ID[m.uci()]
    first = list(game.board.legal_moves)[:2]
    replies = []
    for m in first:
        b = game.board.copy()
        b.push(m)
        replies.append(next(iter(b.legal_moves)))
    z1 = nodes([[1, 0, tok(first[0]), n + 1], [2, 0, tok(first[1]), n + 1]])
    z2 = nodes([[3, 1, tok(replies[0]), n + 2], [4, 2, tok(replies[1]), n + 2]])
    for path, z in zip(([1], [2], [1, 3], [2, 4]), (*z1, *z2)):
        ids = torch.tensor(game.tokens + [int(nodes.token[i]) for i in path])
        f = torch.tensor(game.features() + [nodes.feats[i].tolist() for i in path])
        b = b"".join(game.boards + [nodes.board[i] for i in path])
        b = torch.tensor(np.frombuffer(b, np.uint8).reshape(-1, 68))
        ref = step(tiny, [(Cache(tiny), ids, f.float(), b)])[0]
        np.testing.assert_allclose(z, ref.double().numpy(), atol=3e-5)


native = Path(os.environ.get("ALLIE_CHESS_INCLUDE", "vendor/chess-library/include"))


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
def test_coverage_search_plays_legal_moves(tiny):
    pytest.importorskip("pybind11")
    game = Game(Engine(tiny), 2800, 2800, 180, 2)
    game.update(random_game(6, 12), 175, 176)
    search = tree.Coverage()
    play = Play(mode="strongest", search=5)
    moves, p = search(game, 5)
    assert sorted(moves) == sorted(m.uci() for m in game.board.legal_moves)
    assert abs(p.sum() - 1) < 1e-9
    assert game.decide(play, search).move in moves


def test_ragged_paths_ignore_unwritten_slots(tiny):
    game = Game(Engine(tiny), 1500, 1600, 180, 2)
    game.update(random_game(8, 15), 170, 165)
    feats = np.array(game.features(), np.float32)
    t = tree.Tree(game, game.sync(), capacity=16)
    t.k[:, :, 1:], t.v[:, :, 1:] = float("nan"), float("nan")  # never-written slots
    nodes = t.handles([game.tokens], [feats])
    n, tok = len(game.tokens), lambda m: MOVE_ID[m.uci()]
    first, second = list(game.board.legal_moves)[:2]
    b = game.board.copy()
    b.push(first)
    reply = next(iter(b.legal_moves))
    nodes([[1, 0, tok(first), n + 1]])
    z = nodes([[2, 1, tok(reply), n + 2], [3, 0, tok(second), n + 1]])  # depths 2 and 1
    assert np.isfinite(z).all()
