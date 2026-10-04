"""Allie 2.0 move prediction: load once, then ask for move probabilities at any rating and clock.

    from allie.lichess.api import Allie
    allie = Allie.from_pretrained()  # yimingzhang/allie-2.0 from the Hub, or an export directory
    allie.predict("1. e4 e5 2. Nf3", white_elo=1800, black_elo=1750, time_control="180+2")

Self-contained (torch, numpy, python-chess and this directory's model.py and tokens.py), so the
Hugging Face release ships it as is.
"""

import io
import re
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from .model import Cache, Model, step
from .tokens import HEADER, MOVE_ID, START, advance, features, fill, header

REPO = "yimingzhang/allie-2.0"
TIME, WDL = slice(2350, 2413), slice(2413, 2416)
# seconds at the centre of each think-time bin: 0..15 exact, then log-spaced
CENTRES = np.r_[np.arange(16), 16 * np.exp(np.arange(47) / 7.06)]


def parse_moves(moves):
    """A move list (UCI or SAN strings) or PGN text -> (UCI moves, clocks after each move or
    None, from PGN %clk comments)."""
    board, ucis, clocks = chess.Board(), [], []
    if isinstance(moves, str) and re.search(r"[.\[{]", moves):
        game = chess.pgn.read_game(io.StringIO(moves))
        if game is None or game.errors:
            raise ValueError(f"cannot parse the PGN: {game and game.errors}")
        if game.board().fen() != chess.STARTING_FEN or game.headers.get("Variant", "Standard") not in (
            "Standard", "Chess", ""):  # fmt: skip
            raise ValueError("Allie plays standard chess from the starting position only")
        for node in game.mainline():
            ucis.append(node.move.uci())
            clocks.append(node.clock())
        return ucis, clocks
    for m in moves.split() if isinstance(moves, str) else moves:
        try:
            move = chess.Move.from_uci(m)
            if move not in board.legal_moves:
                raise ValueError
        except ValueError:
            move = board.parse_san(m)
        board.push(move)
        ucis.append(move.uci())
        clocks.append(None)
    return ucis, clocks


def parse_time_control(tc):
    """ "180+2", (180, 2) or None -> (base seconds, increment seconds) or (None, None)."""
    if tc is None or tc == "-":
        return None, None
    if isinstance(tc, str):
        base, _, inc = tc.partition("+")
        return int(base), int(inc or 0)
    return int(tc[0]), int(tc[1])


def resolve(name, **hub):
    """A local directory with config.json and model.safetensors, downloading a Hugging Face
    repo's into the Hub cache first."""
    if Path(name).is_dir():
        return Path(name)
    from huggingface_hub import snapshot_download

    files = ["config.json", "model.safetensors"]
    return Path(snapshot_download(name, allow_patterns=files, **hub))


class Allie:
    """Allie 2.0 on CPU or GPU. predict() and play() reuse the key-value cache of a recent call
    whose game they extend, so asking move after move in one game costs one or two new tokens."""

    def __init__(self, model, sessions=4):
        self.model, self.sessions, self.keep = model, [], sessions

    @classmethod
    def from_pretrained(cls, name=REPO, device=None, dtype=None, int8=None, active_experts=None,
                        **hub):  # fmt: skip
        """name: a Hugging Face repo or a local directory with config.json and
        model.safetensors. device: default CUDA if available. int8: int8 weights, the default
        on CPU (half the memory, twice the speed). active_experts: route each token through only this many
        of its 16 experts (faster, slightly less accurate). hub: revision, cache_dir, token, ...
        for huggingface_hub.snapshot_download."""
        path = resolve(name, **hub)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = dtype or torch.bfloat16
        if int8 is None:
            int8 = device.type == "cpu" and dtype == torch.bfloat16
        return cls(Model(path, device, dtype, active_experts, int8))

    def analyze(self, moves=(), white_elo=1500, black_elo=1500, time_control=None,
                clocks=None):  # fmt: skip
        """The position after `moves`, as a player of the mover's rating would see it:
        moves: probabilities of each legal move (UCI, most likely first); wdl: the mover's
        win / draw / loss probabilities; think_time: expected seconds on this move.
        time_control: "180+2" or (base, increment) in seconds, None if unknown. clocks: each
        mover's time left after each move in seconds (None entries unknown); PGN %clk comments
        fill it. Without clocks the model reads them as unknown, as in its clockless games."""
        ucis, pgn_clocks = parse_moves(moves)
        clocks = list(clocks) if clocks is not None else pgn_clocks
        if len(clocks) != len(ucis):
            raise ValueError(f"{len(clocks)} clocks for {len(ucis)} moves")
        clocks = [None if c is None else int(c) for c in clocks]
        base, inc = parse_time_control(time_control)
        # no clock at all reads as the model's clockless games (the header keeps the time
        # control); gaps between known clocks are filled
        known = base if any(c is not None for c in clocks) or not ucis else None
        clocks = fill(clocks, known)
        board = chess.Board()
        for m in ucis:
            board.push_uci(m)
        legal = [m.uci() for m in board.legal_moves]
        if not legal:
            raise ValueError("the game is over")
        tokens = header(base, inc, white_elo, black_elo) + [MOVE_ID[m] for m in ucis]
        feats = [
            features(p - HEADER + 1, known, inc, clocks) if p >= HEADER - 1 else [-1] * 3
            for p in range(len(tokens))
        ]
        z = self._logits(tokens, feats).double()
        p = torch.softmax(z[[MOVE_ID[m] for m in legal]], 0).tolist()
        order = sorted(range(len(legal)), key=lambda i: -p[i])
        t = torch.softmax(z[TIME], 0).numpy()
        return dict(
            moves={legal[i]: p[i] for i in order},
            wdl=tuple(torch.softmax(z[WDL], 0).tolist()),
            think_time=float(t @ CENTRES) if len(ucis) >= 2 else None,
        )

    def predict(self, moves=(), white_elo=1500, black_elo=1500, time_control=None,
                clocks=None):  # fmt: skip
        """{legal move (UCI): probability}, most likely first. Arguments as analyze()."""
        return self.analyze(moves, white_elo, black_elo, time_control, clocks)["moves"]

    def play(self, moves=(), elo=1500, opponent_elo=None, time_control=None, clocks=None,
             temperature=1.0, rng=None):  # fmt: skip
        """A move (UCI) for the side to move, sampled as a player rated `elo` would play it
        (temperature 0: the most likely move)."""
        n = len(parse_moves(moves)[0])
        them = elo if opponent_elo is None else opponent_elo
        white, black = (elo, them) if n % 2 == 0 else (them, elo)
        p = self.predict(moves, white, black, time_control, clocks)
        if not temperature >= 0:
            raise ValueError("temperature must be a number >= 0")
        if temperature == 0:
            return next(iter(p))
        logp = np.log(np.maximum(np.array(list(p.values())), 1e-300))
        with np.errstate(over="ignore"):  # a tiny temperature sends all but the best to -inf
            w = np.exp((logp - logp.max()) / temperature)
        rng = rng or np.random.default_rng()
        return list(p)[rng.choice(len(w), p=w / w.sum())]

    def _logits(self, tokens, feats):
        """Logits at the last token, extending the cached game that shares the longest prefix."""
        best, n = None, 0
        for s in self.sessions:
            k = 0
            for a, b, fa, fb in zip(s["tokens"], tokens, s["feats"], feats):
                if a != b or fa != fb:
                    break
                k += 1
            if k > n:
                best, n = s, k
        if best is None:
            best = dict(cache=Cache(self.model), tokens=[], feats=[], boards=[])
            self.sessions = [best, *self.sessions][: self.keep]
        # at least the last token, for its logits; never past what the cache really holds
        n = min(n, len(tokens) - 1, best["cache"].n)
        best["cache"].truncate(n)
        boards = best["boards"][:n] or [START]
        while len(boards) < len(tokens):
            p = len(boards)
            boards.append(advance(boards[-1], tokens[p]) if p >= HEADER else START)
        b = np.frombuffer(b"".join(boards[n:]), np.uint8).reshape(-1, 68)
        x = torch.tensor(tokens[n:]), torch.tensor(feats[n:], dtype=torch.float32), torch.tensor(b)
        try:
            z = step(self.model, [(best["cache"], *x)])[0].cpu()
        except BaseException:
            self.sessions.remove(best)  # its cache may be half written
            raise
        best |= dict(tokens=tokens, feats=feats, boards=boards)
        return z


def main(argv=None):
    """allie-predict: the most likely moves in a position, for a player of a given rating."""
    import argparse

    p = argparse.ArgumentParser(prog="allie-predict", description=main.__doc__)
    p.add_argument(
        "moves", nargs="?", default="", help='PGN or moves, e.g. "1. e4 e5 2. Nf3"'
    )
    p.add_argument("--elo", type=int, default=1500, help="both players' rating")
    p.add_argument("--white-elo", type=int)
    p.add_argument("--black-elo", type=int)
    p.add_argument("--tc", help='time control, e.g. "180+2"')
    p.add_argument(
        "--clocks", help="each mover's seconds left after each move, comma-separated"
    )
    p.add_argument(
        "--model", default=REPO, help="Hugging Face repo or export directory"
    )
    p.add_argument("--device")
    p.add_argument("--bf16", action="store_true", help="BF16 weights on CPU (default int8)")
    p.add_argument("--active-experts", type=int, help="routed experts per token (default 16)")
    p.add_argument("--top", type=int, default=5)
    a = p.parse_args(argv)
    clocks = (
        [float(c) if c else None for c in a.clocks.split(",")] if a.clocks else None
    )
    allie = Allie.from_pretrained(a.model, a.device, int8=False if a.bf16 else None, active_experts=a.active_experts)
    white, black = a.white_elo or a.elo, a.black_elo or a.elo
    out = allie.analyze(a.moves, white, black, a.tc, clocks)
    board = chess.Board()
    for m in parse_moves(a.moves)[0]:
        board.push_uci(m)
    side = "white" if board.turn else "black"
    print(
        f"{side} to move, Elo {white} (white) vs {black} (black), time control {a.tc or '?'}"
    )
    for m, q in list(out["moves"].items())[: a.top]:
        print(f"  {board.san(chess.Move.from_uci(m)):<7} {m:<6} {100 * q:5.1f}%")
    w, d, loss = out["wdl"]
    print(
        f"{side} wins / draws / loses: {100 * w:.0f}% / {100 * d:.0f}% / {100 * loss:.0f}%"
    )
    if out["think_time"] is not None:
        print(f"expected think time: {out['think_time']:.1f} s")


if __name__ == "__main__":
    main()
