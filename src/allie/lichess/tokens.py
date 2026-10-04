"""Allie's tokens, board states and clock features, built as the training data builds them.

A game is BOS, base time, increment, white Elo (4 digits), black Elo (4 digits), then one token
per move. The token at position p carries the clock features and board state of the position
that predicts move p - 10 (allie.data.mix Games.feats, allie/model/board_encode.cpp).
Self-contained (no allie imports), so the Hugging Face release can ship it as is.
"""

HEADER, CONTEXT = 11, 1025
BOS, UNK, MOVE_START = 2348, 2349, 378
SECONDS = [0, 15, 30, 45, 60, 90, *range(120, 10801, 60)]
SECONDS_ID = {s: 192 + i for i, s in enumerate(SECONDS)} | {None: 192 + len(SECONDS)}
INCREMENTS_ID = {i: 10 + i for i in range(181)} | {None: 191}


def _moves():
    """Every queen or knight move between squares, and every promotion, as sorted UCI."""
    name = lambda s: "abcdefgh"[s % 8] + str(s // 8 + 1)
    out = set()
    for a in range(64):
        for b in range(64):
            df, dr = abs(a % 8 - b % 8), abs(a // 8 - b // 8)
            if a != b and (not df or not dr or df == dr or {df, dr} == {1, 2}):
                out.add(name(a) + name(b))
        if a // 8 in (1, 6):  # pawns one step from promotion
            last = 0 if a // 8 == 1 else 56
            for f in range(max(a % 8 - 1, 0), min(a % 8 + 2, 8)):
                out |= {name(a) + name(last + f) + p for p in "bnqr"}
    return sorted(out)


MOVES = _moves()
MOVE_ID = {m: MOVE_START + i for i, m in enumerate(MOVES)}
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
        SECONDS_ID.get(base, UNK),
        INCREMENTS_ID.get(increment, UNK),
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


def fill(clocks, base):
    """Each side's unknown clocks (None) linearly interpolated between its known ones, with the
    base time before its first move; after its last known clock, that clock. The model saw
    games with every clock or with none, never with gaps, so gaps are filled, not left -1.
    A side with no known clock and no base time stays unknown."""
    out = list(clocks)
    for side in (0, 1):
        moves = range(side, len(out), 2)
        known = [(j, out[j]) for j in moves if out[j] is not None]
        if base is not None:
            known.insert(0, (side - 2, base))
        if not known:
            continue
        i = 0
        for j in moves:
            while i + 1 < len(known) and known[i + 1][0] <= j:
                i += 1
            (a, x), nxt = known[i], known[i + 1] if i + 1 < len(known) else None
            if out[j] is None:
                if j < a or nxt is None:  # before the first known clock or after the last
                    out[j] = x
                else:
                    b, y = nxt
                    out[j] = round(x + (y - x) * (j - a) / (b - a))
    return out
