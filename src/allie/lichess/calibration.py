"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, without giving up the policy's
diversity. The move is sampled at temperature 1 from the policy, or from allie.search coverage's
calibrated human-move distribution after

    N = min(CAP, K t^ALPHA exp(GAMMA (R - 1700) / 1000), HW t)

simulations, t being the predicted human think time (s) for the position. The time control enters
through t: humans think longer in slower games and on harder moves, and the search follows. N goes
onto LADDER by random rounding in log2(1 + N), as the fit interpolated, and the rung it lands on
never exceeds a tenth of the clock left above the reserve at HW simulations a second; a search
still running after that tenth of the clock (a loaded machine) stops, and the move comes from the
policy. No search in bullet: there it raises the human-move cross-entropy above the policy's.

HW is what a 6-CPU bot playing two games searches per second on the fast backend (about 17 ms a
simulation), so the search costs about three quarters of the think time the bot waits anyway at
most.

Fitted on 220,000 positions of the golden evaluation's July 2026 human games (up to 5,000 per time
control and 200-point rating band from 800 to 2600): the model sees what the human saw, Stockfish
scores every legal move, and the rule minimizes the paired gaps in move accuracy and blunder rate
between its move distribution and the human's moves, with the human-move cross-entropy held at or
below the policy's in every time control and rating band.
"""

import numpy as np

K, ALPHA, GAMMA, CAP, HW = 2.0, 1.0, 3.75, 256, 40.0  # provisional: refit on the annealed model
LADDER = (0, 32, 128, 256)
NO_SEARCH = ("ultraBullet", "bullet")
_U = np.log2(1 + np.array(LADDER))
_BINS = np.arange(63)
SECONDS = np.where(_BINS < 16, _BINS + 0.5, 16 * np.exp((_BINS - 16) / 7.06))  # think-time head bins


def budget(rating, time, speed="blitz"):
    """Coverage simulations (continuous) for a player of `rating`; time: the think-time head's
    63-bin distribution for the position."""
    if speed in NO_SEARCH:
        return 0.0
    t = float(np.asarray(time, float) @ SECONDS / np.sum(time))
    return float(min(CAP, K * t**ALPHA * np.exp(GAMMA * (rating - 1700) / 1000), HW * t))


def limit(clock, reserve=1.0):
    """Seconds a search may take: a tenth of the clock left above the reserve (None: no clock)."""
    return np.inf if clock is None else max(clock - reserve, 0.0) / 10


def ceiling(clock, reserve=1.0):
    """The most simulations the clock allows: limit()'s seconds at HW simulations a second."""
    return HW * limit(clock, reserve)


def pick(n, rng, most=np.inf):
    """n onto LADDER: one of its two neighbours, at random, linearly in log2(1 + n) (as the fit
    interpolated; this keeps the mean of log2(1 + N), not of N), then the largest rung within
    `most`, the hard limit."""
    i = float(np.interp(np.log2(1 + n), _U, np.arange(len(_U))))
    lo = int(i)
    rung = LADDER[lo + 1] if lo + 1 < len(LADDER) and rng.random() < i - lo else LADDER[lo]
    return max(b for b in LADDER if b <= max(min(rung, most), 0))
