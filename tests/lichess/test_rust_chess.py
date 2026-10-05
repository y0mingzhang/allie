"""The Rust chess module (rust/allie-fast/src/chess.rs) against the native search's
Position (allie.search.native: the gold move order and outcome), python-chess, the
model's board encoder (allie.search.board) and allie.lichess.tokens, on random games
and scripted edge cases. ALLIE_CHESS_CORPUS=N scales the random corpus (default 300)."""

import os
import random
import shutil
from collections import Counter

import chess
import numpy as np
import pytest

from allie.data import vocab
from allie.lichess.tokens import HEADER, MOVE_ID, MOVES, START, advance, header

from .test_tokens import random_game

fast = pytest.importorskip("allie_fast")
board = pytest.importorskip("allie.search.board")
CORPUS = int(os.environ.get("ALLIE_CHESS_CORPUS", 300))


@pytest.fixture(scope="module")
def native():
    pytest.importorskip("pybind11")
    if not shutil.which("c++"):
        pytest.skip("no c++ for the native search")
    from allie.search import native

    return native.load()


def check_game(native, moves):
    """Every position of a game (UCI moves) against the native Position, python-chess,
    the board encoder and tokens.advance; returns the positions compared and
    python-chess's final outcome."""
    tokens = header(180, 2, 1500, 1500) + [MOVE_ID[m] for m in moves]
    ours, gold, py = fast.Position(), native.Position(), chess.Board()
    state, states = START, [START] * HEADER
    for k, t in enumerate(tokens[HEADER:]):
        legal = ours.legal()
        assert legal == gold.legal(), (k, py.fen())
        assert set(legal) == {MOVE_ID[m.uci()] for m in py.legal_moves}
        assert ours.outcome() == gold.outcome(), (k, py.fen())
        assert ours.fen() == gold.fen() and ours.white == gold.white == py.turn
        ours.push(t)
        gold.push(t)
        py.push_uci(moves[k])
        state = fast.advance_board(state, t)
        states.append(advance(states[-1], t))
        assert state == states[-1], (k, py.fen())
    assert ours.legal() == gold.legal() and ours.outcome() == gold.outcome()
    assert ours.fen() == gold.fen() and ours.white == gold.white
    last = fast.Position.from_tokens(tokens)
    assert last.fen() == ours.fen() and last.legal() == ours.legal()
    assert last.outcome() == ours.outcome()
    ref = board.encode(np.array([tokens]))[0]
    np.testing.assert_array_equal(fast.encode_boards(np.array(tokens)), ref)
    np.testing.assert_array_equal(fast.encode_boards(tokens), ref)
    np.testing.assert_array_equal(
        np.frombuffer(b"".join(states), np.uint8).reshape(-1, 68), ref
    )
    return len(moves) + 1, py.outcome(claim_draw=False)


def san_game(sans):
    b = chess.Board()
    return [b.push_san(s).uci() for s in sans.split()]


def boards(moves):
    return fast.encode_boards(header(180, 2, 1500, 1500) + [MOVE_ID[m] for m in moves])


def test_vocabulary():
    assert fast.moves() == MOVES == vocab.MOVES and len(MOVES) == 1968
    assert fast.MOVE_START == 378 and fast.HEADER == HEADER and fast.START == START


def test_random_corpus(native):
    positions, ends = 0, Counter()
    for seed in range(CORPUS):
        n, outcome = check_game(native, random_game(seed, 150))
        positions += n
        ends[outcome.termination.name if outcome else "unfinished"] += 1
    print(f"\n{CORPUS} games, {positions} positions compared, endings {dict(ends)}")
    assert positions > 50 * CORPUS


def test_castling_each_side(native):
    for sans, rights in (
        ("e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 Re1 O-O", [0, 12]),
        ("d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7 O-O-O O-O-O", [0, 12]),
    ):
        moves = san_game(sans)
        check_game(native, moves)
        b = boards(moves)
        assert [b[-1, 65], b[-2, 65]] == rights
    b = boards(san_game("e4 e5 Nf3 Nc6 Bc4 Bc5 O-O"))[-1]
    assert list(b[4:8]) == [0, 4, 6, 0] and b[65] == 12
    b = boards(san_game("d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7 O-O-O O-O-O"))[-1]
    assert list(b[56:61]) == [0, 0, 12, 10, 0]
    # rook moves and captures of a rook's square clear its right
    b = boards(san_game("a4 h5 Ra3 Rh6 Ra1 Rh8"))[-1]
    assert b[65] == 1 + 8
    b = boards(san_game("b3 g6 Bb2 Bg7 g4 Bxb2 Bg2 Bxa1 Bxb7 Nf6 Bxa8"))[-1]
    assert b[65] == 1 + 4


def test_en_passant(native):
    moves = san_game("e4 a6 e5 d5 exd6 a5 b4 axb4 a4 bxa3")
    check_game(native, moves)
    b = boards(moves)[HEADER - 1 :]  # b[k]: after move k
    assert b[1, 66] == 5  # e4: the flag is set with no capture possible
    after_e4 = fast.Position.from_tokens(header(180, 2, 1500, 1500) + [MOVE_ID["e2e4"]])
    assert after_e4.fen().split()[3] == "-"
    assert b[4, 66] == 4 and b[5, 66] == 0  # d5, then exd6
    assert b[5, 35] == 0 and b[5, 43] == 1  # d5's pawn gone, white pawn on d6
    assert b[10, 24] == 0 and b[10, 16] == 7  # a4's pawn gone, black pawn on a3


def test_promotions(native):
    for sans in (
        "a4 h5 a5 h4 a6 h3 axb7 hxg2 bxa8=N gxh1=B",  # underpromotion captures
        "a4 h5 a5 h4 a6 h3 axb7 hxg2 bxc8=R gxf1=Q+",  # rook and queen captures, check
        "h4 Nf6 h5 a5 h6 a4 hxg7 a3 g8=Q axb2 Nc3 b1=R",  # plain pushes
    ):
        moves = san_game(sans)
        check_game(native, moves)
    b = boards(san_game("a4 h5 a5 h4 a6 h3 axb7 hxg2 bxa8=N gxh1=B"))[-1]
    assert b[56] == 2 and b[7] == 9
    b = boards(san_game("h4 Nf6 h5 a5 h6 a4 hxg7 a3 g8=Q axb2 Nc3 b1=R"))[-1]
    assert b[62] == 5 and b[1] == 10


def test_checkmate_and_stalemate(native):
    for sans, score in (
        ("f3 e5 g4 Qh4#", 0.0),
        ("e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#", 1.0),
        (
            "e3 a5 Qh5 Ra6 Qxa5 h5 h4 Rah6 Qxc7 f6 Qxd7+ Kf7 Qxb7 Qd3 "
            "Qxb8 Qh7 Qxc8 Kg6 Qe6",
            0.5,
        ),
    ):
        moves = san_game(sans)
        check_game(native, moves)
        p = fast.Position.from_tokens(
            header(180, 2, 1500, 1500) + [MOVE_ID[m] for m in moves]
        )
        assert p.outcome() == score and p.legal() == []


def test_fivefold_repetition(native):
    moves = san_game("Nf3 Nf6 Ng1 Ng8 " * 4)
    check_game(native, moves)
    tokens = header(180, 2, 1500, 1500) + [MOVE_ID[m] for m in moves]
    assert fast.Position.from_tokens(tokens[:-4]).outcome() == -1
    p = fast.Position.from_tokens(tokens)
    assert p.outcome() == 0.5 and p.legal() != []
    assert p.child(MOVE_ID["e2e4"]).outcome() == -1 and p.outcome() == 0.5


def shuffle_game(seed):
    """Reversible moves only and never a fifth repetition, mate or stalemate: a 75-move
    draw."""
    rng, b = random.Random(seed), chess.Board()
    while not b.is_game_over():
        moves = [
            m
            for m in b.legal_moves
            if not b.is_capture(m) and b.piece_type_at(m.from_square) != chess.PAWN
        ]
        rng.shuffle(moves)
        for m in moves:
            b.push(m)
            if b.is_fivefold_repetition() or b.is_checkmate() or b.is_stalemate():
                b.pop()
            else:
                break
        else:
            return None
    return b if b.is_seventyfive_moves() else None


def test_seventy_five_move_rule(native):
    b = next(filter(None, map(shuffle_game, range(100))))
    moves = [m.uci() for m in b.move_stack]
    assert b.halfmove_clock == 150 and len(moves) == 150
    _, outcome = check_game(native, moves)
    assert outcome.termination == chess.Termination.SEVENTYFIVE_MOVES
    tokens = header(180, 2, 1500, 1500) + [MOVE_ID[m] for m in moves]
    assert fast.Position.from_tokens(tokens[:-1]).outcome() == -1
    assert fast.Position.from_tokens(tokens).outcome() == 0.5


def capture_game(seed, plies=400):
    rng, b = random.Random(seed), chess.Board()
    while not b.is_game_over() and len(b.move_stack) < plies:
        moves = list(b.legal_moves)
        captures = [m for m in moves if b.is_capture(m)]
        b.push(rng.choice(captures or moves))
    return b


def test_insufficient_material(native):
    found = 0
    for seed in range(200):
        b = capture_game(seed)
        if not b.is_insufficient_material():
            continue
        _, outcome = check_game(native, [m.uci() for m in b.move_stack])
        assert outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL
        found += 1
        if found == 3:
            break
    assert found == 3


def test_errors():
    p = fast.Position()
    for token in (
        5,
        377,
        378 + 1968,
        MOVE_ID["e2e5"],
        MOVE_ID["e1g1"],
        MOVE_ID["e7e5"],
    ):
        with pytest.raises(ValueError):
            p.push(token)
    assert p.fen() == chess.STARTING_FEN
    child = p.child(MOVE_ID["e2e4"])
    assert not child.white and p.white and p.fen() == chess.STARTING_FEN
    prefix = header(180, 2, 1500, 1500)
    with pytest.raises(ValueError):
        fast.Position.from_tokens(prefix + [MOVE_ID["e2e4"], MOVE_ID["e2e4"]])
    with pytest.raises(ValueError):
        fast.advance_board(START[:67], MOVE_ID["e2e4"])
    with pytest.raises(ValueError):
        fast.advance_board(START, MOVE_ID["e7e5"])
    with pytest.raises(ValueError):
        fast.encode_boards(prefix + [5])
    assert fast.encode_boards(prefix[:3]).shape == (3, 68)
