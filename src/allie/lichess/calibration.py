"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, predicting human moves no worse than
its policy does. Per time control and 200-point rating bin (CELLS), the move is sampled at
temperature 1 from the policy; from allie.search coverage's calibrated human-move distribution after
a fixed number of simulations; or from the policy tilted by the searched values, pi ~ prior
exp(beta Q), after a fixed number of coverage simulations or lookahead calls.

Chosen on 220,000 positions of the golden evaluation's July 2026 human games (up to 5,000 per time
control and bin, 800-2600): per cell, the searcher and budget whose expected move accuracy and
blunder rate (Stockfish scoring every legal move) are closest in Elo to the humans', among those
whose human-move cross-entropy is at least a standard error below the policy's; the policy where
none is.

A search is sized for a tenth of the clock left above the reserve at COST seconds a simulation or
call: the budget steps down its searcher's ladder until it fits. A search still running at a fifth
of the clock stops, and the move comes from the policy.
"""

import numpy as np

# (speed, bin) -> (searcher, budget, beta): beta None is coverage's calibrated distribution; a cell
# not listed plays the policy
CELLS = {}
LADDER = dict(coverage=(8, 32, 128, 256), lookahead=(1, 2, 4, 8, 16))
COST = dict(coverage=1 / 40, lookahead=0.35)  # seconds a simulation or call (bench, 4 threads)


def cell(rating, speed):
    """(searcher, budget, beta) for a player of `rating` at `speed`, or None: the policy."""
    return CELLS.get((speed, min(max(int(rating) // 200 * 200, 800), 2600)))


def limit(clock, reserve=1.0):
    """Seconds a search may take: a fifth of the clock left above the reserve (None: no clock)."""
    return np.inf if clock is None else max(clock - reserve, 0.0) / 5


def affordable(searcher, budget, clock, reserve=1.0):
    """The largest rung of the searcher's ladder up to `budget` that fits a tenth of the clock left
    above the reserve (half of limit()); 0: none."""
    fits = limit(clock, reserve) / 2
    return max((b for b in LADDER[searcher] if b <= budget and b * COST[searcher] <= fits), default=0)


def tilt(prior, q, beta):
    """The searched distribution: pi ~ prior exp(beta Q)."""
    z = np.log(np.maximum(prior, 1e-300)) + beta * np.asarray(q, float)
    z = np.exp(z - z.max())
    return z / z.sum()
