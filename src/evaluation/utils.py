import functools
import math
import shutil

import chess
import chess.engine


@functools.cache
def get_stockfish_engine():
    assert shutil.which(
        "stockfish"
    ), "check that stockfish is installed and visible in $PATH"

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Threads": 16, "Hash": 8192})

    return engine


@functools.cache
def compute_white_centipawn_score(board_fen: str, nodes=10**6) -> float:
    engine = get_stockfish_engine()
    board = chess.Board()
    board.set_fen(board_fen)
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))

    score = info["score"].white().score()

    if score is None:
        assert info["score"].is_mate()
        # a forced mate is worth +-1000
        score = 1000 if info["score"].white().mate() > 0 else -1000

    # clamp score to [-1000, 1000]
    score = max(-1000, min(1000, score))

    return score


def compute_win_percents_from_eval(eval: float) -> float:
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * eval)) - 1)


def compute_accuracy_from_win_percents(prev: float, curr: float) -> float:
    if curr >= prev:
        return 100.0
    return max(0.0, 103.1668 * math.exp(-0.04354 * (prev - curr)) - 3.1669)


def compute_centipawn_loss_and_accuracy(
    board: chess.Board, move: str
) -> tuple[float, float]:
    prev_eval = compute_white_centipawn_score(board.fen())
    board.push_uci(move)
    curr_eval = compute_white_centipawn_score(board.fen())

    if board.turn == chess.WHITE:
        prev_eval, curr_eval = -prev_eval, -curr_eval

    loss = prev_eval - curr_eval
    loss = max(0, loss)

    # https://lichess.org/page/accuracy
    win_percents_prev = compute_win_percents_from_eval(prev_eval)
    win_percents_curr = compute_win_percents_from_eval(curr_eval)

    accuracy = compute_accuracy_from_win_percents(win_percents_prev, win_percents_curr)

    # CPL is nonnegative
    board.pop()

    return loss, accuracy
