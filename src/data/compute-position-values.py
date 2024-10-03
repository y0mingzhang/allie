import math
import sys

import chess
import pandas as pd
from tqdm.auto import tqdm

from evaluation.utils import compute_white_centipawn_score, get_stockfish_engine


def evaluate_moves(moves_uci: str) -> list[float]:
    board = chess.Board()
    scores = []
    for move_uci in moves_uci.split():
        board.push_uci(move_uci)
        white_score = compute_white_centipawn_score(board.fen(), nodes=10**7)
        if board.turn == chess.WHITE:
            # was a move by black
            score = -white_score
        else:
            # was a move by white
            score = white_score
        scores.append(score)
    return scores


def compute_state_value(value: float) -> float:
    return 2 / (1 + math.exp(-0.00368208 * value)) - 1


def compute_state_values(evals: list[float]) -> list[float]:
    return [compute_state_value(e) for e in evals]


def main():
    tqdm.pandas()

    df = pd.read_json(sys.argv[1], lines=True)
    df["moves-eval"] = df["moves-uci"].progress_map(evaluate_moves)
    df["position-values"] = df["moves-eval"].map(compute_state_values)
    df.to_json(sys.argv[1], lines=True, orient="records")

    get_stockfish_engine().quit()
    print("Done!")


if __name__ == "__main__":
    main()
