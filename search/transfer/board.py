"""Board-state input (model track): causal board features per position, from the token rows.

The C++ encoder is Codex's (results/recipe10x/board-lr-v1/source): for every position it gives the
64 squares (0 empty, 1-6 white PNBRQK, 7-12 black), side to move, castling rights, en-passant file
and an in-game flag, for the position after that token. Branches follow Codex's findings: one-hot
matmul (not square lookups, whose backward was pathological under deterministic BF16), zero-init
output, FP32 parameters and Adam state with a BF16 forward, LR multiplier 0.1, and the direct map's
~67 active features scaled by 1/8 (board-input-v1).
"""

import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
BOARD_IN = 864
from torch import nn
from torch.nn import functional as F

SOURCE = Path(__file__).resolve().parent
CPP_SHA = "2c5b1a2432f93ecd057000d1aaf5acd789ff848519657c026c3243d22c7b5589"
TABLE_SHA = "d51d604209cc7afb0437906a3bb046aa50bb855608a414abbadbbd545f12e1c5"
_encoder = None


def _load():
    global _encoder
    if _encoder is None:
        src = SOURCE / "board_encode.cpp"
        assert hashlib.sha256(src.read_bytes()).hexdigest() == CPP_SHA
        table_file = SOURCE / "move-table.json"
        assert hashlib.sha256(table_file.read_bytes()).hexdigest() == TABLE_SHA
        cache = Path(
            os.environ.get("ALLIE_BOARD_CACHE", "/scratch/yimingz3/allie/board-encoder")
        )
        cache = cache / CPP_SHA
        cache.mkdir(parents=True, exist_ok=True)
        lib = cache / "encoder.so"
        if not lib.exists():
            tmp = cache / f"encoder-{os.getpid()}.so"
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O3",
                    "-fPIC",
                    "-shared",
                    str(src),
                    "-o",
                    str(tmp),
                ],
                check=True,
            )
            tmp.replace(lib)
        fn = ctypes.CDLL(str(lib)).encode_boards
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64] + [
            ctypes.c_void_p
        ] * 3
        fn.restype = ctypes.c_int
        sq = lambda f, r: ord(f) - 97 + (int(r) - 1) * 8
        table = np.full((2350, 3), -1, np.int32)
        for t, u in enumerate(json.loads(table_file.read_text()), 378):
            table[t] = [
                sq(u[0], u[1]),
                sq(u[2], u[3]),
                dict(n=2, b=3, r=4, q=5)[u[4]] if len(u) == 5 else 0,
            ]
        _encoder = fn, table
    return _encoder


def encode(inputs):
    """(rows, cols) token ids -> (rows, cols, 68) uint8 board states."""
    fn, table = _load()
    x = np.ascontiguousarray(inputs, np.int64)
    out, err = np.empty((*x.shape, 68), np.uint8), np.zeros(3, np.int64)
    code = fn(
        x.ctypes.data, *x.shape, table.ctypes.data, out.ctypes.data, err.ctypes.data
    )
    if code:
        raise ValueError(
            f"invalid token rows: code {code}, row/column/token {err.tolist()}"
        )
    return out


def meta(states):
    """Side to move, castling rights and en-passant file, one-hot, padded to 32 columns."""
    side = F.one_hot(states[:, 64].long(), 2)
    rights = F.one_hot(states[:, 65].long(), 16)
    ep = F.one_hot(states[:, 66].long(), 9)
    return F.pad(torch.cat((side, rights, ep), -1), (0, 5))


def features(states, dtype):
    """(T, 68) states -> (T, BOARD_IN) one-hot features, zero outside games."""
    pieces = F.one_hot(states[:, :64].long(), 13).flatten(1)
    f = torch.cat((pieces, meta(states)), -1).to(dtype)
    return f * states[:, 67:68].to(dtype)


class BoardDirect(nn.Module):
    """Linear map of the one-hot board to the model width (an embedding of piece-on-square)."""

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(BOARD_IN, width))

    def forward(self, states, dtype):
        return features(states, dtype) @ self.weight.type(dtype) / 8


class BoardConv(nn.Module):
    """Codex's BoardConv: 13 planes -> 3x3 conv 32 -> 2 residual 3x3 convs -> 1x1 squeeze 8 ->
    concat side / castling / en-passant embedding -> layer norm -> zero-init linear."""

    def __init__(self, width):
        super().__init__()
        self.first = nn.Parameter(torch.empty(32, 13, 3, 3))
        self.residual = nn.ParameterList(
            nn.Parameter(torch.empty(32, 32, 3, 3)) for _ in range(2)
        )
        self.squeeze = nn.Parameter(torch.empty(8, 32, 1, 1))
        for w in (self.first, *self.residual, self.squeeze):
            nn.init.kaiming_uniform_(w, a=5**0.5)  # nn.Conv2d's default init
        self.meta = nn.Parameter(
            torch.randn(32, 32) * 0.02
        )  # side, castling, ep one-hot rows, padded to 32
        self.output = nn.Parameter(torch.zeros(544, width))

    def forward(self, states, dtype):
        x = (
            F.one_hot(states[:, :64].long(), 13)
            .to(dtype)
            .reshape(-1, 8, 8, 13)
            .permute(0, 3, 1, 2)
        )
        x = F.gelu(F.conv2d(x, self.first.type(dtype), padding=1), approximate="tanh")
        for w in self.residual:
            x = x + F.gelu(F.conv2d(x, w.type(dtype), padding=1), approximate="tanh")
        x = F.conv2d(x, self.squeeze.type(dtype)).flatten(1)
        m = meta(states).to(dtype) @ self.meta.type(dtype)
        x = F.layer_norm(torch.cat((x, m), -1), (544,))
        return (x @ self.output.type(dtype)) * states[:, 67:68].to(dtype)


def build(kind, width):
    net = BoardDirect(width) if kind == "direct" else BoardConv(width)
    for p in net.parameters():
        p.label, p.fp32_state = "board", True
        p.lr_mul = 0.1
        p.wd_mul = 1.0 if p.ndim >= 2 else 0.0
    return net
