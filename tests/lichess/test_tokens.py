import random

import chess
import numpy as np
import pyarrow as pa
import pytest

from allie.data.mix import Games
from allie.lichess.tokens import HEADER, MOVE_ID, START, advance, features, header


def random_game(seed, plies=300):
    rng, board = random.Random(seed), chess.Board()
    while not board.is_game_over() and len(board.move_stack) < plies:
        moves = list(board.legal_moves)
        special = [
            m
            for m in moves
            if m.promotion or board.is_castling(m) or board.is_en_passant(m)
        ]
        board.push(rng.choice(special if special and rng.random() < 0.5 else moves))
    return [m.uci() for m in board.move_stack]


def test_header():
    assert header(180, 2, 1500, 2850) == [2348, 199, 12, 1, 5, 0, 0, 2, 8, 5, 0]
    assert header(None, None, 800, 99999)[:3] == [2348, 377, 191]
    assert header(17, 0, 1500, 1500)[1] == 2349  # no such base-time token


def test_boards_match_native_encoder():
    board = pytest.importorskip("allie.search.board")
    for seed in range(40):
        moves = random_game(seed)
        tokens = header(180, 2, 1500, 1500) + [MOVE_ID[m] for m in moves]
        states = [START] * HEADER
        for t in tokens[HEADER:]:
            states.append(advance(states[-1], t))
        ours = np.frombuffer(b"".join(states), np.uint8).reshape(-1, 68)
        np.testing.assert_array_equal(ours, board.encode(np.array([tokens]))[0])


def test_features_match_training_loader():
    rng = np.random.default_rng(0)
    for base, inc, n in [(180, 2, 61), (60, 0, 3), (600, 5, 1), (15, 0, 9)]:
        spent = rng.integers(0, 9, n)
        clocks, left = [], [base, base]
        for k in range(n):
            if k >= 2:
                left[k % 2] = max(left[k % 2] - spent[k], 0) + inc
            clocks.append(left[k % 2])
        table = pa.table(dict(
            moves=[list(range(n))], white_elo=[1500], black_elo=[1600], white_title=[None],
            black_title=[None], format=[2], rated_prefix=[1.0], base=[base], increment=[inc],
            termination=[0], val_leak=[False], clocks=[clocks], token_hash=[1],
        ))  # fmt: skip
        expect = Games(table, feats=True).feats(0, HEADER + n, drop=False)
        ours = [features(k, base, inc, clocks) for k in range(n)]
        np.testing.assert_array_equal(
            np.array(ours), expect[HEADER - 1 : HEADER - 1 + n]
        )


def test_features_unknown():
    assert features(5, None, None, [None] * 5) == [-1, -1, -1]
    assert features(4, 60, 0, [60, 60, None, 58]) == [-1, 58, -1]


def test_vocabulary_matches_training():
    from allie.data import vocab
    from allie.lichess import tokens

    assert tokens.MOVES == vocab.MOVES and tokens.MOVE_ID == vocab.MOVE_ID
    assert tokens.BOS == vocab.BOS and tokens.UNK == vocab.UNK
    for k, i in vocab.SECONDS_ID.items():
        assert tokens.SECONDS_ID[None if k == "*" else int(k)] == i
    for k, i in vocab.INCREMENTS_ID.items():
        assert tokens.INCREMENTS_ID[None if k == "*" else int(k)] == i


def test_fill_clocks():
    from allie.lichess.tokens import fill

    # each side on its own: white from 180 (before move 0) to 170 (move 2), then held;
    # black from 180 (before move 1) to 150 (move 5)
    assert fill([None, None, 170, None, None, 150], 180) == [175, 170, 170, 160, 170, 150]
    assert fill([None, None, 170, None], None) == [170, None, 170, None]  # no base, no black
    assert fill([], 60) == []
