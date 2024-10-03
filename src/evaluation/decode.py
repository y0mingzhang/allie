import functools
import math
import os
import random
import shutil
from abc import ABC
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Type

import chess
import chess.engine
import numpy as np
import openai
import torch
import torch.nn.functional as F
from openai_api_cache import OpenAIAPICache

from evaluation.carlini import ChessLLM
from modeling.data import (
    TIME_MEAN,
    TIME_STDEV,
    Game,
    UCITokenizer,
    apply_time_normalization,
    undo_time_normalization,
)
from modeling.model import CausalLMWithRegressionHead, Model
from modeling.moves import NUM_CHESS_MOVES
from modeling.utils import to_device

PRECISIONS = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class DecodeResult:
    move: str
    think_time: float | None
    resigned: bool


@dataclass(frozen=True)
class ScoreResult:
    legal_move_probabilities: torch.Tensor
    think_time: float | None = None
    resigned: bool = None
    all_move_probabilities: torch.Tensor | None = None
    top_move_legal: bool | None = None
    sampled_move_legal: float | None = None
    white_advantage: float | None = None


class DecodeStrategy(ABC):
    def __init__(
        self,
        model: Model,
        tokenizer: UCITokenizer,
        sample: bool | str = False,
        device: Optional[str] = None,
        precision: str = "float32",
        **kwargs: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sample = sample
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=PRECISIONS[precision])
        )

        self.init(**kwargs)
        self.check()

    def init(self, *args, **kwargs) -> None: ...

    def check(self) -> None: ...

    def decode(self, game: Game) -> str:
        return self.decode_full(game).move

    def decode_full(self, game: Game) -> DecodeResult:
        legal_moves = game.legal_moves
        score_result = self.score_full(game, legal_moves)
        probs = score_result.legal_move_probabilities
        think_time = score_result.think_time
        resigned = score_result.resigned

        if self.sample not in (True, False):
            assert self.sample == "rating-based"
            opponent_elo = (
                game.white_elo if game.board.turn == chess.WHITE else game.black_elo
            )
            if opponent_elo >= 2200 and len(game.board.move_stack) > 8:
                sample = False
            else:
                sample = True
        else:
            sample = self.sample

        if sample:
            choice = probs.multinomial(1)[0].item()
        else:
            choice = probs.argmax()
        return DecodeResult(legal_moves[choice], think_time, resigned)

    def score_full(
        self,
        game: Game,
        legal_moves: list[str],
    ) -> ScoreResult:
        raise NotImplementedError

    def run_stats(self) -> dict:
        return {}

    def cleanup(self) -> None: ...


class Maia(DecodeStrategy):
    def __init__(
        self,
        maia_elo: int | None = None,
        maia_weights_dir: str = "maia-weights",
        opening_strategy: DecodeStrategy | None = None,
    ) -> None:
        # maia_elo set to None -> take closest model to opponent ELO
        assert shutil.which("lc0"), "check that lc0 is installed and visible in $PATH"

        if maia_elo is None:
            maia_elo_list = list(range(1100, 2000, 100))
        else:
            maia_elo_list = [maia_elo]

        self.engines = {}
        for elo in maia_elo_list:
            engine = chess.engine.SimpleEngine.popen_uci("lc0")
            engine.configure(
                {
                    "Threads": 4,
                    "WeightsFile": os.path.join(maia_weights_dir, f"maia-{elo}.pb"),
                }
            )
            self.engines[elo] = engine
        self.opening_strategy = opening_strategy
        self.sample = False

    def score_full(
        self,
        game: Game,
        legal_moves: list[str],
    ) -> ScoreResult:
        if self.opening_strategy is not None and len(game.board.move_stack) < 10:
            return self.opening_strategy.score_full(game, legal_moves)

        elo = game.white_elo if game.board.turn == chess.WHITE else game.black_elo
        assert elo is not None, game
        closest_elo = min(self.engines, key=lambda e: abs(e - elo))
        play_result = self.engines[closest_elo].play(
            game.board, chess.engine.Limit(nodes=1)
        )

        move_idx = legal_moves.index(play_result.move.uci())
        move_probabilities = torch.zeros(len(legal_moves))
        move_probabilities[move_idx] = 1.0

        return ScoreResult(move_probabilities)

    def cleanup(self):
        for engine in self.engines.values():
            engine.quit()

    def run_stats(self):
        return {}


class Policy(DecodeStrategy):
    def init(
        self,
        temperature: float = 1.0,
        time_prediction: bool = True,
    ) -> None:
        self.temperature = max(temperature, 1e-5)
        self.time_prediction = time_prediction

    def score_full(
        self,
        game: Game,
        legal_moves: list[str],
    ) -> ScoreResult:
        batch = self.tokenizer.pad_and_collate([game])
        batch = to_device(batch, self.device)

        with self.ctx:
            outputs = self.model(**batch)
        logits = outputs["logits"][0, -1].float().cpu()
        legal_moves_tensor = torch.tensor(
            [self.tokenizer.token_to_id[m] for m in legal_moves],
            dtype=torch.int64,
        )

        move_logits = logits[legal_moves_tensor]
        time_encoding = outputs["time_logits"][0, -1].item()
        think_time = undo_time_normalization(time_encoding)

        if not self.time_prediction:
            think_time = None

        pov = 1 if game.board.turn == chess.WHITE else -1
        value = pov * outputs["value_logits"][0, -1].item()

        resigned_logit = logits[self.tokenizer.resigned_token_id].item()
        resigned = resigned_logit > move_logits.max().item() and value < -0.9

        all_moves_logits = logits[:NUM_CHESS_MOVES]
        all_moves_probabilities = F.softmax(all_moves_logits / self.temperature, dim=0)

        top_move_idx = all_moves_probabilities.argmax().item()
        top_move_legal = self.tokenizer.tokens[top_move_idx] in legal_moves
        sampled_move_legal = all_moves_probabilities[legal_moves_tensor].sum().item()

        return ScoreResult(
            F.softmax(move_logits / self.temperature, dim=0),
            think_time,
            resigned,
            all_moves_probabilities,
            top_move_legal,
            sampled_move_legal,
            outputs["value_logits"][0, -1].item(),
        )


class ActionValue(Policy):
    def init(self, batch_size: int = 16, **kwargs):
        super().init(**kwargs)
        self.batch_size = batch_size

    def score_full(
        self,
        game: Game,
        legal_moves: list[str],
    ) -> ScoreResult:

        policy_score = super().score_full(game, legal_moves)
        games = [game.make_move(m) for m in legal_moves]
        action_values = []
        for i in range(0, len(games), self.batch_size):
            batch = self.tokenizer.pad_and_collate(games[i : i + self.batch_size])
            batch = to_device(batch, self.device)
            with self.ctx:
                outputs = self.model(**batch)
            pov = 1 if game.board.turn == chess.WHITE else -1
            value = pov * outputs["value_logits"][:, -1].cpu().numpy()
            action_values.append(value)

        action_values = np.concatenate(action_values)
        move_probabilities = torch.zeros(len(legal_moves))
        move_probabilities[action_values.argmax()] = 1.0

        return ScoreResult(
            move_probabilities, policy_score.think_time, policy_score.resigned
        )


class MCTSNode:
    def __init__(self, prior: float, game: Game):
        self.prior = prior
        self.children: dict[str, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.game = game

    @functools.cached_property
    def board(self) -> chess.Board:
        return self.game.board

    @functools.cached_property
    def legal_moves(self) -> list[str]:
        return self.game.legal_moves

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def __repr__(self) -> str:
        return str(
            {
                "prior": self.prior,
                "turn": "white" if self.turn else "back",
                "visit_count": self.visit_count,
                "value": self.value(),
                "children": list(self.children),
            }
        )


class MCTS(Policy):
    def init(
        self,
        n_sims: int = 50,
        depth_limit: int = 100,
        c_puct: float = 1.25,
        c_base: float = 19652.0,
        dummy_value_function: bool = False,
        kl_reg_solution: bool = True,
        **kwargs,
    ) -> None:
        super().init(**kwargs)
        self.n_sims = n_sims
        self.depth_limit = depth_limit

        self.c_puct = c_puct
        self.c_base = c_base
        self.dummy_value_function = dummy_value_function
        self.kl_reg_solution = kl_reg_solution

        self.total_rollouts = 0

    def check(self) -> None:
        assert isinstance(
            self.model, CausalLMWithRegressionHead
        ), "model does not provide a value prediction"

    def evaluate_node(
        self,
        node: MCTSNode,
    ) -> float | tuple[float, float | None]:
        # note: value tracks how good a state is from parent's pov
        # because we maximize this value during child selection

        # check whether game state is terminal
        # if so, use actual outcome as value
        if node.board.is_game_over():
            outcome = node.board.outcome()
            assert outcome is not None
            if outcome.winner is None:
                # draw
                return 0.0

            assert outcome.winner != node.board.turn
            return 1.0

        # otherwise, use NN estimate
        batch = self.tokenizer.pad_and_collate([node.game])
        batch = to_device(batch, self.device, blocking=True)

        with self.ctx:
            outputs = self.model(**batch)
            legal_moves_tensor = torch.tensor(
                [self.tokenizer.token_to_id[m] for m in node.legal_moves],
                dtype=torch.int64,
                device=self.device,
            )
            logits = outputs["logits"][0, -1][legal_moves_tensor]
            policy = F.softmax(logits, 0).tolist()
            time_encoding = outputs["time_logits"][0, -1].item()
            think_time = undo_time_normalization(time_encoding)

            # get pov value
            pov = 1 if node.board.turn == chess.BLACK else -1
            value = pov * outputs["value_logits"][0, -1].item()

        for action, prior in zip(node.legal_moves, policy):
            child_state = node.game.make_move(action)
            node.children[action] = MCTSNode(prior, child_state)

        return 0.0 if self.dummy_value_function else value

    def select_child(self, node: MCTSNode) -> MCTSNode:
        return max(
            node.children.values(), key=lambda child: self.ucb_score(node, child)
        )

    def ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        pb_c = (
            math.log((parent.visit_count + self.c_base + 1) / self.c_base) + self.c_puct
        )
        pb_c *= math.sqrt(parent.visit_count) / (1 + child.visit_count)
        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    def backup(self, search_path: list[MCTSNode], value: float) -> None:
        for node in search_path[::-1]:
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def score_full(
        self,
        game: Game,
        legal_moves: list[str],
    ) -> ScoreResult:

        policy_score = super().score_full(game, legal_moves)

        root = MCTSNode(0.0, game)
        self.evaluate_node(root)
        for _ in range(self.n_sims):
            node = root
            search_path = [node]

            # keep going down the path until:
            # leaf node is reached (unexplored / terminal)
            # OR max search depth reached
            while node.expanded() and len(search_path) <= self.depth_limit:
                node = self.select_child(node)
                search_path.append(node)

            value = self.evaluate_node(node)
            self.backup(search_path, value)

        self.total_rollouts += self.n_sims

        visit_counts = torch.FloatTensor(
            [root.children[move].visit_count for move in legal_moves]
        )

        if not self.kl_reg_solution:
            return ScoreResult(
                visit_counts / visit_counts.sum(),
                policy_score.think_time,
                policy_score.resigned,
            )

        if self.kl_reg_solution:
            total_visits = visit_counts.sum().item()
            lambda_N = self.c_puct * total_visits / (len(visit_counts) + total_visits)

            children_priors = torch.FloatTensor(
                [root.children[move].prior for move in legal_moves]
            )
            children_values = torch.FloatTensor(
                [root.children[move].value() for move in legal_moves]
            )

            alpha_min = (children_values + lambda_N * children_priors).max()
            alpha_max = (children_values + lambda_N).max()

            while True:
                alpha = (alpha_min + alpha_max) / 2
                pi_bar = lambda_N * (children_priors) / (alpha - children_values)
                if pi_bar.sum() > 1.0:
                    alpha_min = alpha
                else:
                    alpha_max = alpha

                if torch.isclose(
                    pi_bar.sum(), torch.tensor(1.0), rtol=1e-3
                ) or torch.isclose(alpha_min, alpha_max, rtol=1e-3):
                    return ScoreResult(
                        pi_bar / pi_bar.sum(),
                        policy_score.think_time,
                        policy_score.resigned,
                    )

    def run_stats(self) -> dict[str, int]:
        return {"total_rollouts": self.total_rollouts}


class AdaptiveMCTS(MCTS):
    def init(
        self,
        mean_n_sims: int = 50,
        depth_limit: int = 100,
        scale_c_puct: bool = True,
        c_puct: float = 1.25,
        c_base: float = 19652.0,
        dummy_value_function: bool = False,
        kl_reg_solution: bool = True,
        use_human_move_time: bool = False,
        conversion_strategy: str = "linear",
        **kwargs,
    ) -> None:
        Policy.init(self, **kwargs)
        self.mean_n_sims = mean_n_sims
        self.depth_limit = depth_limit
        self.mean_n_sims = mean_n_sims
        self.scale_c_puct = scale_c_puct
        self.unscaled_c_puct = c_puct
        self.c_puct = c_puct
        self.c_base = c_base
        self.dummy_value_function = dummy_value_function
        self.kl_reg_solution = kl_reg_solution
        self.use_human_move_time = use_human_move_time
        assert conversion_strategy in ["linear", "binary"]
        self.conversion_strategy = conversion_strategy

        self.total_rollouts = 0

    def convert_time_to_sims(self, time_encoding: float) -> int:
        if self.conversion_strategy == "linear":
            scale = self.mean_n_sims * TIME_STDEV / TIME_MEAN
            n_sims = round(scale * time_encoding + self.mean_n_sims)

        elif self.conversion_strategy == "binary":
            n_sims = 0 if time_encoding < 0 else self.mean_n_sims

        if n_sims < 0:
            return 0
        if n_sims > 200:
            return 200

        return n_sims

    def evaluate_root(
        self,
        node: MCTSNode,
    ) -> tuple[float, int, float]:
        # note: value tracks how good a state is from parent's pov
        # because we maximize this value during child selection

        # check whether game state is terminal
        # if so, use actual outcome as value
        if node.board.is_game_over():
            outcome = node.board.outcome()
            assert outcome is not None
            if outcome.winner is None:
                # draw
                return 0.0, 1, 0.0

            assert outcome.winner != node.board.turn
            return 1.0, 1, 0.0

        # otherwise, use NN estimate
        batch = self.tokenizer.pad_and_collate([node.game])
        batch = to_device(batch, self.device, blocking=True)

        with self.ctx:
            outputs = self.model(**batch)
            legal_moves_tensor = torch.tensor(
                [self.tokenizer.token_to_id[m] for m in node.legal_moves],
                dtype=torch.int64,
                device=self.device,
            )
            logits = outputs["logits"][0, -1][legal_moves_tensor]

            policy = F.softmax(logits, 0).tolist()

            time_encoding = outputs["time_logits"][0, -1].item()
            think_time = undo_time_normalization(time_encoding)

            # get pov value
            pov = 1 if node.board.turn == chess.BLACK else -1
            value = pov * outputs["value_logits"][0, -1].item()

            # get time prediction
            if self.use_human_move_time:
                assert (
                    node.game.next_move_seconds is not None
                ), "missing human move seconds"
                time_encoding = apply_time_normalization(node.game.next_move_seconds)
            else:
                time_encoding = outputs["time_logits"][0, -1].item()
            n_sims = self.convert_time_to_sims(time_encoding)

        for action, prior in zip(node.legal_moves, policy):
            child_state = node.game.make_move(action)
            node.children[action] = MCTSNode(prior, child_state)

        return 0.0 if self.dummy_value_function else value, n_sims, think_time

    def score_full(
        self,
        game: Game,
        legal_moves: list[str],
    ) -> ScoreResult:

        policy_score = Policy.score_full(self, game, legal_moves)

        root = MCTSNode(0.0, game)
        _, n_sims, _ = self.evaluate_root(root)
        self.total_rollouts += n_sims

        if n_sims == 0:
            # just use prior when no simulations are run
            return ScoreResult(
                torch.FloatTensor([root.children[move].prior for move in legal_moves]),
                policy_score.think_time,
                policy_score.resigned,
            )

        if self.scale_c_puct:
            if n_sims == 0:
                self.c_puct = self.unscaled_c_puct
            else:
                self.c_puct = (
                    self.unscaled_c_puct
                    * math.sqrt(self.mean_n_sims)
                    / math.sqrt(n_sims)
                )
        for _ in range(n_sims):
            node = root
            search_path = [node]

            # keep going down the path until:
            # leaf node is reached (unexplored / terminal)
            # OR max search depth reached
            while node.expanded() and len(search_path) <= self.depth_limit:
                node = self.select_child(node)
                search_path.append(node)

            value = self.evaluate_node(node)
            self.backup(search_path, value)

        visit_counts = torch.FloatTensor(
            [root.children[move].visit_count for move in legal_moves]
        )

        if not self.kl_reg_solution:
            return ScoreResult(
                visit_counts / visit_counts.sum(),
                policy_score.think_time,
                policy_score.resigned,
            )

        if self.kl_reg_solution:
            total_visits = visit_counts.sum().item()
            lambda_N = self.c_puct * total_visits / (len(visit_counts) + total_visits)

            children_priors = torch.FloatTensor(
                [root.children[move].prior for move in legal_moves]
            )
            children_values = torch.FloatTensor(
                [root.children[move].value() for move in legal_moves]
            )

            alpha_min = (children_values + lambda_N * children_priors).max()
            alpha_max = (children_values + lambda_N).max()

            while True:
                alpha = (alpha_min + alpha_max) / 2
                pi_bar = lambda_N * (children_priors) / (alpha - children_values)
                if pi_bar.sum() > 1.0:
                    alpha_min = alpha
                else:
                    alpha_max = alpha

                if torch.isclose(
                    pi_bar.sum(), torch.tensor(1.0), rtol=1e-3
                ) or torch.isclose(alpha_min, alpha_max, rtol=1e-3):
                    return ScoreResult(
                        pi_bar / pi_bar.sum(),
                        policy_score.think_time,
                        policy_score.resigned,
                    )


class GPT35Turbo(DecodeStrategy):
    def __init__(
        self,
        dryrun=False,
    ) -> None:
        with open("/home/yimingz3/secrets/openai-api-key") as f:
            opeani_client = openai.OpenAI(api_key=f.read().strip())
        with open("/home/yimingz3/secrets/redis-auth") as f:
            redis_pw = f.read()
        self.redis = OpenAIAPICache(
            opeani_client, mode="completion", port=12321, password=redis_pw
        )
        self.chessllm = ChessLLM(None, {"num_lookahead_tokens": 10})
        self.dryrun = dryrun
        self.sample = False
        self.cache = {}

    def request(self, pgn: str) -> str:
        resp = self.redis.generate(
            prompt=pgn,
            model="gpt-3.5-turbo-instruct",
            temperature=0.0,
            max_tokens=20,
        )
        return resp.choices[0].text

    def serialize_board(self, board: chess.Board) -> str:
        return " ".join(move.uci() for move in board.move_stack)

    def make_move(self, board: chess.Board) -> chess.Move:

        board_state = self.serialize_board(board)
        if board_state in self.cache:
            return self.cache[board_state]

        pgn = self.chessllm.get_query_pgn(board)
        next_moves = []

        output = self.request(pgn)
        if output[:2] == "-O":
            output = self.request(pgn + " ")
        next_moves = self.chessllm.try_moves(board, output)

        move_to_play = (
            board.parse_san(next_moves[0])
            if next_moves
            else random.choice(list(board.legal_moves))
        )

        board = board.copy()
        for move_san in next_moves:
            board_state = self.serialize_board(board)
            move = board.parse_san(move_san)
            assert move is not None
            self.cache[board_state] = move
            board.push(move)

        return move_to_play

    def score_full(
        self,
        game: Game,
        legal_moves: list[str],
    ) -> ScoreResult:

        move = self.make_move(game.board)
        move_idx = legal_moves.index(move.uci())
        move_probabilities = torch.zeros(len(legal_moves))
        move_probabilities[move_idx] = 1.0

        return ScoreResult(move_probabilities)


DECODE_STRATEGIES: dict[str, Type[DecodeStrategy]] = {
    "maia": Maia,
    "policy": Policy,
    "mcts": MCTS,
    "adaptive-mcts": AdaptiveMCTS,
    "gpt-3.5-turbo": GPT35Turbo,
}
