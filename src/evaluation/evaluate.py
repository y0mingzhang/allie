import argparse
import json
import os
import random
import sys
from os.path import join

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from evaluation.decode import (
    DECODE_STRATEGIES,
    DecodeStrategy,
    GPT35Turbo,
    Maia,
    Policy,
)
from evaluation.utils import (
    compute_centipawn_loss_and_accuracy,
    compute_white_centipawn_score,
    get_stockfish_engine,
)
from modeling.data import UCITokenizer, json_to_game, json_to_partial_game
from modeling.model import initialize_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def evaluate_value(
    strategy: Policy,
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    games = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), file=sys.stdout):
        game_dict = row.to_dict()
        game = json_to_game(game_dict)
        moves = game.moves
        game_dict["model-move"] = []
        game_dict["stockfish-eval"] = []
        game_dict["model-white-advantage"] = []

        for i in range(len(moves)):
            prefix = json_to_partial_game(game_dict, i)
            legal_moves = prefix.legal_moves
            score_results = strategy.score_full(prefix, legal_moves)

            if score_results.white_advantage is not None:
                game_dict["model-white-advantage"].append(score_results.white_advantage)
                game_dict["stockfish-eval"].append(
                    compute_white_centipawn_score(prefix.board.fen())
                )
        games.append(game_dict)

    return {
        "games": games,
    }


def evaluate_quick(
    strategy: Policy,
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    games = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), file=sys.stdout):
        game_dict = row.to_dict()
        game = json_to_game(game_dict)
        moves = game.moves
        game_dict["model-move"] = []
        game_dict["human-move"] = []
        game_dict["top-move-legal"] = []
        game_dict["sampled-move-legal"] = []
        game_dict["human-resigned"] = []
        game_dict["model-resigned"] = []
        game_dict["model-think-time"] = []
        game_dict["model-white-advantage"] = []

        for i in range(len(moves)):
            prefix = json_to_partial_game(game_dict, i)
            legal_moves = prefix.legal_moves
            score_results = strategy.score_full(prefix, legal_moves)

            legal_moves_probs = score_results.legal_move_probabilities

            if strategy.sample:
                choice = legal_moves_probs.multinomial(1)[0].item()
            else:
                choice = legal_moves_probs.argmax().item()

            model_move = legal_moves[choice]
            human_move = moves[i]

            game_dict["model-move"].append(model_move)
            game_dict["human-move"].append(human_move)

            if score_results.top_move_legal is not None:
                game_dict["top-move-legal"].append(score_results.top_move_legal)

            if score_results.sampled_move_legal is not None:
                game_dict["sampled-move-legal"].append(score_results.sampled_move_legal)

            if score_results.resigned is not None:
                game_dict["human-resigned"].append(False)

            if score_results.resigned is not None:
                game_dict["model-resigned"].append(score_results.resigned)

            if score_results.think_time is not None:
                game_dict["model-think-time"].append(score_results.think_time)

            if score_results.white_advantage is not None:
                game_dict["model-white-advantage"].append(score_results.white_advantage)

        if game.normal_termination and game.board.outcome(claim_draw=True) is None:
            # human has resigned
            decode_results = strategy.decode_full(game)
            game_dict["human-resigned"].append(True)
            game_dict["model-resigned"].append(decode_results.resigned)
        games.append(game_dict)

    return {
        "games": games,
    }


@torch.inference_mode(True)
def evaluate(
    strategy: DecodeStrategy,
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    games = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), file=sys.stdout):
        game_dict = row.to_dict()
        game = json_to_game(game_dict)
        moves = game.moves
        game_dict["model-centipawn-loss"] = []
        game_dict["model-lichess-accuracy"] = []
        game_dict["model-move"] = []
        game_dict["human-move"] = []
        game_dict["top-move"] = []
        game_dict["top-move-legal"] = []
        game_dict["sampled-move-legal"] = []

        for i in range(len(moves)):
            prefix = json_to_partial_game(game_dict, i)
            legal_moves = prefix.legal_moves
            score_results = strategy.score_full(prefix, legal_moves)

            legal_moves_probs = score_results.legal_move_probabilities

            top_choice = legal_moves_probs.argmax().item()
            if strategy.sample:
                choice = legal_moves_probs.multinomial(1)[0].item()
            else:
                choice = top_choice

            model_move = legal_moves[choice]
            human_move = moves[i]

            board = prefix.board
            model_loss, model_acc = compute_centipawn_loss_and_accuracy(
                board, model_move
            )

            game_dict["model-centipawn-loss"].append(model_loss)
            game_dict["model-lichess-accuracy"].append(model_acc)
            board.push_uci(human_move)

            game_dict["model-move"].append(model_move)
            game_dict["top-move"].append(legal_moves[top_choice])
            game_dict["human-move"].append(human_move)

            if score_results.top_move_legal is not None:
                game_dict["top-move-legal"].append(score_results.top_move_legal)

            if score_results.sampled_move_legal is not None:
                game_dict["sampled-move-legal"].append(score_results.sampled_move_legal)

        games.append(game_dict)

    get_stockfish_engine().quit()
    return {
        "games": games,
    }


@torch.inference_mode(True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--value", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--alias", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_rows", type=int, default=-1)
    parser.add_argument("--decode", type=str, default="policy")
    parser.add_argument("--decode_args", type=json.loads, default={})
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--maia_elo", type=int, default=None)

    args = parser.parse_args()

    if not args.debug:
        assert args.output_file is not None

    if args.decode == "maia":
        if not args.alias:
            if args.maia_elo is None:
                args.alias = "maia-closest"
            else:
                args.alias = f"maia-{args.maia_elo}"
        strategy = Maia(args.maia_elo)
    elif args.decode == "gpt-3.5-turbo":
        strategy = GPT35Turbo()
    else:
        assert args.decode in DECODE_STRATEGIES
        assert args.config

        config = OmegaConf.load(args.config)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"setting manual seed = {SEED}")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        tokenizer = UCITokenizer(**config.data_config.tokenizer_config)

        model = initialize_model(tokenizer, **config.model_config)
        model = model.to(DEVICE)
        model.eval()
        ckpt_path = join(config.trainer_config.output_dir, "best.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])

        strategy = DECODE_STRATEGIES[args.decode](
            model,
            tokenizer,
            sample=args.sample,
            precision="float32",
            **args.decode_args,
        )

    dataframe = pd.read_json(args.dataset, lines=True, orient="records")
    if args.debug:
        dataframe = dataframe[:5]
    if args.n_rows >= 0:
        dataframe = dataframe[: args.n_rows]

    assert not (args.quick and args.value)
    if args.quick:
        eval_fn = evaluate_quick
    elif args.value:
        eval_fn = evaluate_value
    else:
        eval_fn = evaluate
    eval_outputs = eval_fn(strategy, dataframe)

    eval_outputs = {
        "decode": args.decode,
        "alias": args.alias,
        "n_rows": args.n_rows,
        "sample": args.sample,
        **args.decode_args,
        **eval_outputs,
        **strategy.run_stats(),
    }

    if args.debug:
        eval_outputs.pop("games", None)
        print(json.dumps(eval_outputs, indent=2))

    if not args.debug:
        output_dir = os.path.dirname(args.output_file)
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "a") as f:
            json.dump(eval_outputs, f)
            f.write("\n")

    if isinstance(strategy, Maia):
        strategy.cleanup()


if __name__ == "__main__":
    main()
