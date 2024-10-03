import json
import time

from evaluation.decode import DecodeStrategy
from modeling.data import Game, UCITokenizer, json_to_partial_game


def algorithm_test(alg: DecodeStrategy) -> None:
    test_game_data = {
        "game-id": "DeepBlue vs. Kasparov, 1997 Game 2",
        "moves-uci": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 h7h6 d2d4 f8e8 b1d2 e7f8 d2f1 c8d7 f1g3 c6a5 b3c2 c7c5 b2b3 a5c6 d4d5 c6e7 c1e3 e7g6 d1d2 f6h7 a2a4 g6h4 f3h4 d8h4 d2e2 h4d8 b3b4 d8c7 e1c1 c5c4 a1a3 e8c8 c1a1 c7d8 f2f4 h7f6 f4e5 d6e5 e2f1 f6e8 f1f2 e8d6 e3b6 d8e8 a3a2 f8e7 b6c5 e7f8 g3f5 d7f5 e4f5 f7f6 c5d6 f8d6 a4b5 a6b5 c2e4 a8a2 f2a2 e8d7 a2a7 c8c7 a7b6 c7b7 a1a8 g8f7 b6a6 d7c7 a6c6 c7b6 g1f1 b7b8 a8a6",
        "moves-seconds": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "result": "1-0",
        "white-elo": "2800",
        "black-elo": "2840",
        "termination": "Normal",
        "time-control": "5400+0",
    }

    moves_made = 0
    t0 = time.time()
    for partial_idx in range(0, len(test_game_data["moves-seconds"]), 2):
        partial_game = json_to_partial_game(test_game_data, partial_idx)
        result = alg.decode_full(partial_game)
        moves_made += 1

        if partial_idx == 10:
            print(f"At ply 10, board state is: {partial_game.board.fen()}")
            print("Model output:")
            print(json.dumps(result.__dict__, indent=2))

    dt = time.time() - t0
    print(f"self test passed: {dt:.1f} seconds spent on {moves_made} moves")
