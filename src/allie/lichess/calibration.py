"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control. One rule for every time control:

    x = (R - 1700) / 1000
    T = exp(TAU0 + TAU x)
    N = min(CAP, K t exp(GAMMA x), HW t, HW clock / 10)
    pi ~ s_N^(1/T)          s_0: the policy; s_N: allie.search coverage's calibrated human-move
                            distribution after N simulations; t: the predicted human think time (s)

The time control enters through t: humans think longer in slower games and on harder moves, and the
search follows; it is negligible below about 2000 and grows with the rating. HW is what a 6-CPU bot
playing two games searches per second (about 75 ms a simulation), so the search costs at most about
three quarters of the think time the bot waits anyway, and never a tenth of its clock. N goes onto
LADDER by random rounding in log2(1 + N), as the fit interpolated.

Fitted on 220,000 positions of the golden evaluation's July 2026 human games, up to 5,000 per time
control and 200-point rating band from 800 to 2600: the model sees what the human saw, Stockfish
scores every legal move, and the rule minimizes the paired gaps in move accuracy and blunder rate
between its full move distribution and the human's moves. Cross-validated over games, it cuts the
current sampling's chi-square from 5,744 to 969 (temperature alone: 1,830; search alone: 1,851).
"""

import numpy as np

TAU0, TAU, K, GAMMA, CAP, HW = -0.1, -0.2, 1.414, 3.0, 256, 10.0
LADDER = (0, 8, 32, 128, 256)
_U = np.log2(1 + np.array(LADDER))
_BINS = np.arange(63)
SECONDS = np.where(
    _BINS < 16, _BINS + 0.5, 16 * np.exp((_BINS - 16) / 7.06)
)  # think-time head bins


def temperature(rating):
    return float(np.exp(TAU0 + TAU * (rating - 1700) / 1000))


def budget(rating, time, clock=None):
    """Coverage simulations (continuous) for a player of `rating`; time: the think-time head's
    63-bin distribution for the position; clock: the bot's seconds left (None: unknown)."""
    t = float(np.asarray(time, float) @ SECONDS / np.sum(time))
    n = min(CAP, K * t * np.exp(GAMMA * (rating - 1700) / 1000), HW * t)
    return float(n if clock is None else min(n, HW * clock / 10))


def pick(n, rng):
    """n onto LADDER: one of its two neighbours, at random, linearly in log2(1 + n)."""
    i = float(np.interp(np.log2(1 + n), _U, np.arange(len(_U))))
    lo = int(i)
    return (
        LADDER[lo + 1] if lo + 1 < len(LADDER) and rng.random() < i - lo else LADDER[lo]
    )


def sharpen(p, t):
    """p^(1/t), normalized."""
    z = np.log(np.maximum(np.asarray(p, float), 1e-300)) / t
    z = np.exp(z - z.max())
    return z / z.sum()
