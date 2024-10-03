import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import chess
import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from modeling.moves import CHESS_MOVES

Batch = dict[str, torch.Tensor]

logger = logging.getLogger(__name__)

"""
Design of the chess tokenizer

Vocab contains:
- valid moves under UCI - 1968 tokens
- blitz time controls - 24 tokens
- resignation token
- special tokens (BOS, EOS, PAD, UNK)

Packing game info into 32 bits:
- bits 0-15 encode move time as {seconds + 1} (used for time modeling)
  0x00000000 says that the next token is not a move, so no move time needed.
- bits 16-17 encode game outcome (used for value modeling)
  0x0000 -> white wins
  0x4000 -> draw
  0x8000 -> black wins
- remaining bits encode token idx
- when (token & 0x3FFF) > len(tokenizer), the token encodes player ELO.
  Recover the exact ELO by subtracting len(tokenizer).
"""


@dataclass
class Game:
    time_control: str | None
    white_elo: int | None
    black_elo: int | None
    outcome: str | None
    normal_termination: bool
    moves: list[str]
    moves_seconds: list[int]
    next_move_seconds: int | None = None

    @cached_property
    def board(self) -> chess.Board:
        board = chess.Board()
        for move in self.moves:
            board.push_uci(move)
        return board

    @cached_property
    def legal_moves(self) -> list[str]:
        legal_moves = [move.uci() for move in self.board.legal_moves]
        return legal_moves

    def make_move(self, move: str) -> Self:
        return Game(
            self.time_control,
            self.white_elo,
            self.black_elo,
            self.outcome,
            self.normal_termination,
            self.moves + [move],
            self.moves_seconds + [0],
            None,
        )

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)


@dataclass(frozen=True)
class PlayResult:
    move: str
    move_time: float | None


def json_to_game(d: dict) -> Game:
    outcome = d["result"] if d["result"] in OUTCOME_MASKS else "1/2-1/2"
    return Game(
        time_control=d["time-control"],
        white_elo=int(d["white-elo"]),
        black_elo=int(d["black-elo"]),
        outcome=outcome,
        normal_termination=d["termination"] == "Normal",
        moves=d["moves-uci"].split(),
        moves_seconds=d["moves-seconds"],
    )


def json_to_partial_game(d: dict, halfmove: int) -> Game:
    return Game(
        time_control=d["time-control"],
        white_elo=int(d["white-elo"]),
        black_elo=int(d["black-elo"]),
        outcome=None,
        normal_termination=False,
        moves=d["moves-uci"].split()[:halfmove],
        moves_seconds=d["moves-seconds"][:halfmove],
        next_move_seconds=(
            d["moves-seconds"][halfmove] if halfmove < len(d["moves-seconds"]) else None
        ),
    )


BLITZ_TIME_CONTROLS_TOKENS = [
    "180+0",
    "300+0",
    "180+2",
    "300+3",
    "300+2",
    "420+0",
    "240+0",
    "180+1",
    "180+3",
    "300+1",
    "360+0",
    "300+4",
    "420+1",
    "240+2",
    "180+5",
    "120+2",
    "120+3",
    "240+3",
    "240+1",
    "360+2",
    "60+3",
    "60+5",
    "240+4",
    "240+5",
]

NORMAL_TERMINATION_TOKEN = "<RESIGNED-OR-CHECKMATED>"

WHITE_WIN_MASK = 0x0000
DRAW_MASK = 0x4000
BLACK_WIN_MASK = 0x8000
OUTCOME_MASKS = {
    "1-0": WHITE_WIN_MASK,
    "1/2-1/2": DRAW_MASK,
    "0-1": BLACK_WIN_MASK,
}

# empirical stats computed on training data
TIME_MEAN = 4.64001
TIME_STDEV = 6.16533


def apply_time_normalization(t: torch.FloatTensor | int) -> torch.FloatTensor | int:
    if isinstance(t, torch.Tensor):
        t = t.clamp(0, 60)
    else:
        if t < 0:
            t = 0
        elif t > 60:
            t = 60
    return (t - TIME_MEAN) / TIME_STDEV


def undo_time_normalization(t_normed: torch.FloatTensor) -> torch.FloatTensor:
    return t_normed * TIME_STDEV + TIME_MEAN


class UCITokenizer:
    def __init__(
        self,
        max_length: int = 128,
        vocab_size_multiple_of: int = 1,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        **kwargs: dict,
    ):
        if kwargs:
            logger.warning(f"tokenizer got unrecognized arguments {kwargs}")

        self.tokens = (
            CHESS_MOVES + BLITZ_TIME_CONTROLS_TOKENS + [NORMAL_TERMINATION_TOKEN]
        )
        self.max_length = max_length
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.resigned_token = NORMAL_TERMINATION_TOKEN
        self.special_tokens = []

        self.special_tokens.extend([bos_token, eos_token, unk_token, pad_token])
        self.tokens.extend(self.special_tokens)
        undefined_idx = 0
        while len(self.tokens) % vocab_size_multiple_of != 0:
            self.tokens.append(f"<UNUSED-{undefined_idx}")
            undefined_idx += 1
        assert len(self.tokens) % vocab_size_multiple_of == 0

        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.special_token_ids = [self.token_to_id[t] for t in self.special_tokens]

        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.resigned_token_id = self.token_to_id[self.resigned_token]

    def tokenize(
        self,
        game: Game,
        add_elo: bool = True,
        add_time_control: bool = True,
        add_termination: bool = True,
        add_value: bool = True,
    ) -> np.ndarray:
        if add_elo:
            assert isinstance(game.white_elo, int)
            assert isinstance(game.black_elo, int)
            seq = [game.white_elo + len(self), game.black_elo + len(self)]

        else:
            seq = [self.bos_token_id]

        moves = game.moves
        assert isinstance(moves, list)

        if add_time_control:
            time_control = game.time_control
            assert isinstance(time_control, str)
        else:
            time_control = self.unk_token

        moves_seconds = game.moves_seconds
        assert isinstance(moves_seconds, list) and len(moves) == len(moves_seconds)

        tokens = [time_control] + moves
        for move, time in zip(tokens, moves_seconds + [-1]):
            if time < 0:
                time = -1
            elif time >= 0xFFFF:
                # this should not happen in blitz
                assert False, f"move time {time}s overflows 16-bit repr"
            idx = self.token_to_id.get(move, self.unk_token_id)
            idx += (time + 1) << 16
            seq.append(idx)

        if add_termination and game.normal_termination:
            seq.append(self.token_to_id[NORMAL_TERMINATION_TOKEN])

        if game.outcome is not None:
            # game is terminal
            seq.append(self.eos_token_id)
        iids = np.array(seq, dtype=np.uint32)

        if add_value and game.outcome is not None:
            outcome = game.outcome
            assert outcome in OUTCOME_MASKS
            mask = OUTCOME_MASKS[outcome]
            iids = iids | mask

        return iids

    def pad(self, seq: list[int]) -> list[int]:
        seq = seq + [self.pad_token_id] * (self.max_length - len(seq))
        assert len(seq) == self.max_length

        return seq

    def __len__(self) -> int:
        return len(self.tokens)

    def pad_and_collate(
        self,
        games: list[Game | np.ndarray],
        return_labels: bool = True,
    ) -> Batch:
        if isinstance(games[0], np.ndarray):
            data = torch.from_numpy(np.array(games).astype(np.int64))
        else:
            seqs = [
                self.tokenize(game)[: self.max_length].astype(np.int64)
                for game in games
            ]
            assert isinstance(games[0], Game)
            seq_length = max(len(seq) for seq in seqs)

            data = torch.stack(
                [
                    torch.from_numpy(
                        np.pad(
                            seq,
                            (0, seq_length - len(seq)),
                            "constant",
                            constant_values=self.pad_token_id,
                        )
                    )
                    for seq in seqs
                ]
            )

        move_time_encoding = data >> 16
        move_seconds = move_time_encoding - 1
        move_seconds_normalized = apply_time_normalization(move_seconds)
        time_labels = torch.where(
            move_time_encoding == 0, -100, move_seconds_normalized
        )

        outcomes = (data & 0xFFFF) >> 14
        assert outcomes.max() < 3

        input_ids = data & 0x3FFF
        attention_mask = input_ids != self.pad_token_id
        labels = torch.where(attention_mask, input_ids, -100)
        labels[labels >= len(self)] = -100
        value_labels = -outcomes.to(torch.float32) + 1
        value_labels = torch.where(attention_mask, value_labels, -100)

        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(0)
        encoding = {
            "input_ids": input_ids,
            "attention_mask": attention_mask.to(torch.float),
            "position_ids": position_ids,
            "value_labels": value_labels,
            "time_labels": time_labels,
        }
        if return_labels:
            encoding["labels"] = labels
        return encoding


def load_data(
    tokenizer_config: DictConfig,
    train_file: str = "data/lichess-2022-blitz-train/2022.bin",
) -> tuple[UCITokenizer, tuple[np.memmap, Dataset, Dataset]]:
    tokenizer = UCITokenizer(**tokenizer_config)
    data_files = {
        "validation": "*val.jsonl",
        "test": "*test.jsonl",
    }
    train_data = np.memmap(
        train_file,
        dtype=np.uint32,
    )

    dataset = load_dataset(
        "data/lichess-2022-blitz-sampled",
        data_files=data_files,
    )
    return tokenizer, (
        train_data,
        dataset["validation"],
        dataset["test"],
    )
