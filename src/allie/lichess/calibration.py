"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, predicting human moves no worse than
its policy does on average. One rule for every rating and time control, no thresholds: the move's
think time is drawn first (behaviour.think); coverage search runs the largest rung of LADDER that
fits both it less MARGIN and a tenth of the clock above the reserve, a simulation priced at COST or
at the bot's recent measured cost when higher (up to 4 COST: a busy or slower machine searches
less), and the move is sampled at temperature 1 from the policy tilted by the searched values'
log-odds, pi ~ prior exp(beta arctanh(0.99 Q)), with

    beta = BETA0 exp(GAMMA (R - 1700) / 1000) (t / 10 s)^DELTA,

t the position's expected human think time (the think-time head's mean). Stronger players and harder
positions get a sharper tilt, fast ones (bullet) little. A search stops at the think time less
MARGIN (or a fifth of the clock), and the policy plays: no search runs past the drawn think time.

Fitted for the annealed Allie 2.0 on the golden evaluation's July 2026 human games (up to 5,000
positions per time control and 200-point bin, 800-2600; analysis/elo_strength/calib_unified.py),
scoring each position as the bot plays it (its rungs mixed by the think-time draws): the debiased
squared Elo error of move accuracy and blunder rate (Stockfish on every legal move) summed over the
blitz, rapid and classical bins, with the human-move cross-entropy at or below the policy's on
average over all bins, DELTA from 0.5, 0.625, 0.75 and 1 (each game half's fit picks these same
values), MARGIN included. RMS Elo error, all games / each half scored by the other half's fit:
bullet 107 / 0 (its flat human curve leaves bullet Elo mostly noise), blitz 0 / 0, rapid 64 / 42,
classical 64 / 49; the per-cell table this replaces, held out: 141 / 80 / 72 / 76. Cross-entropy
against the policy's: bullet +0.0014, blitz -0.0043, rapid -0.0071, classical -0.0069 nats; bullet
1800 above it at 95% (+0.0012 +- 0.0011). Bullet 2600 plays +95 / +188 Elo strong, classical 2400
and 2600 weak (-68 / -121, -134 / -241): LADDER stops at 256, the largest search scored on every bin.
"""

import numpy as np

BETA0, GAMMA, DELTA = 0.297, 3.0, 0.625
LADDER = (8, 32, 128, 256)
COST = 0.008  # seconds a coverage simulation (16 cpus, 12 threads, 10 games)
MARGIN = 0.2  # seconds a search ends before the think time (an engine step under load is <= 0.17 s)
_BINS = np.arange(63)
SECONDS = np.where(_BINS < 16, _BINS + 0.5, 16 * np.exp((_BINS - 16) / 7.06))  # think-time head bins


def beta(rating, time):
    """The tilt for a player of `rating` at a position whose think-time head is `time` (63 bins)."""
    t = float(np.asarray(time, float) @ SECONDS / np.sum(time))
    return BETA0 * np.exp(GAMMA * (rating - 1700) / 1000) * (t / 10) ** DELTA


def limit(clock, reserve=1.0):
    """Seconds a search may take: a fifth of the clock left above the reserve (None: no clock)."""
    return np.inf if clock is None else max(clock - reserve, 0.0) / 5


def affordable(clock, reserve=1.0, think=np.inf, cost=COST):
    """The largest rung of LADDER that fits both a tenth of the clock left above the reserve (half
    of limit()) and `think` seconds at `cost` seconds a simulation; 0: none."""
    fits = min(limit(clock, reserve) / 2, think)
    return max((b for b in LADDER if b * cost <= fits), default=0)


def tilt(prior, q, beta):
    """The searched distribution: pi ~ prior exp(beta arctanh(0.99 Q)), on the W - L value's log-odds
    (Q saturates near +-1, so a tilt on Q itself under-tilts clear positions)."""
    q = np.arctanh(0.99 * np.clip(np.asarray(q, float), -1, 1))
    z = np.log(np.maximum(prior, 1e-300)) + beta * q
    z = np.exp(z - z.max())
    return z / z.sum()
