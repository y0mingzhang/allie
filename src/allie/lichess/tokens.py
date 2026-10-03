"""Allie's tokens, board states and clock features, built as the training data builds them.

A game is BOS, base time, increment, white Elo (4 digits), black Elo (4 digits), then one token
per move. The token at position p carries the clock features and board state of the position
that predicts move p - 10 (allie.data.mix Games.feats, allie/model/board_encode.cpp).
"""

from allie.data.vocab import BOS, INCREMENTS_ID, MOVES, SECONDS_ID, UNK

MOVE_START = 378
HEADER = 11
CONTEXT = 1025

_sq = lambda f, r: ord(f) - 97 + (int(r) - 1) * 8
TABLE = [
    (_sq(m[0], m[1]), _sq(m[2], m[3]), " nbrq".index(m[4]) + 1 if len(m) == 5 else 0)
    for m in MOVES
]
# squares 0..63 (1-6 white PNBRQK, 7-12 black), white to move, castling rights (1 K, 2 Q, 4 k, 8 q),
# en-passant file + 1, inside a game
START = bytes(
    [4, 2, 3, 5, 6, 3, 2, 4, *[1] * 8, *[0] * 32, *[7] * 8, 10, 8, 9, 11, 12, 9, 8, 10]
) + bytes([1, 15, 0, 1])
CORNERS = ((0, 2), (7, 1), (56, 8), (63, 4))


def header(base, increment, white_elo, black_elo):
    """Header tokens. base, increment: seconds, None if the game has no clock."""
    digits = lambda e: [int(c) for c in f"{min(max(int(e), 0), 9999):04d}"]
    return [
        BOS,
        SECONDS_ID.get("*" if base is None else str(base), UNK),
        INCREMENTS_ID.get("*" if increment is None else str(increment), UNK),
        *digits(white_elo),
        *digits(black_elo),
    ]


def advance(state, token):
    """The board state after a move token (board_encode.cpp). The move must be legal."""
    b = bytearray(state)
    fr, to, promotion = TABLE[token - MOVE_START]
    white, piece = b[64], b[fr]
    kind = (piece - 1) % 6 + 1
    ep = b[66] - 1 + (40 if white else 16) if b[66] else -1
    if kind == 1 and to == ep and not b[to] and fr % 8 != to % 8:
        b[to - 8 if white else to + 8] = 0
    b[fr] = 0
    b[to] = promotion + (0 if white else 6) if promotion else piece
    if kind == 6:
        b[65] &= 12 if white else 3
        if abs(to - fr) == 2:
            rook = to + 1 if to > fr else to - 2
            b[(fr + to) // 2], b[rook] = b[rook], 0
    for square, right in CORNERS:
        if square in (fr, to):
            b[65] &= 15 ^ right
    b[66] = to % 8 + 1 if kind == 1 and abs(to - fr) == 16 else 0
    b[64] = 1 - white
    return bytes(b)


def features(k, base, increment, clocks):
    """Clock features of the position predicting move k (0-based): the mover's time left, the
    opponent's, and the mover's think time on their previous move, in whole seconds, -1 = unknown.
    clocks[j]: the clock of move j's mover after move j (seconds or None). Each side's first move
    does not tick the clock."""
    if base is None:
        return [-1, -1, -1]
    c = lambda j: base if j < 0 else clocks[j] if clocks[j] is not None else -1
    mover, other = c(k - 2), c(k - 1)
    prev = -1
    if k >= 4 and c(k - 4) >= 0 and mover >= 0:
        prev = c(k - 4) - mover + (increment or 0)
    return [mover, other, prev if prev >= 0 else -1]
