import json
import random

import chess


def generate_random_game() -> dict:
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 80:
        board.push(random.choice(list(board.legal_moves)))

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result = outcome.result()
    else:
        result = "1/2-1/2"
    moves = " ".join([m.uci() for m in board.move_stack])

    return {
        "moves-uci": moves,
        "moves-seconds": [0] * len(moves),
        "result": result,
        "white-elo": 1500,
        "black-elo": 1500,
        "termination": "Normal",
        "time-control": "180+2",
    }


def main():
    with open("/home/yimingz3/src/chess-lm/data/random-games/test.jsonl", "w") as f:
        for i in range(2000):
            game = generate_random_game()
            json.dump(game, f)
            f.write("\n")


if __name__ == "__main__":
    main()
