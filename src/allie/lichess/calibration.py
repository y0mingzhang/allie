"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, predicting human moves no worse than
its policy does. Per time control and 200-point rating bin (CELLS), the move is sampled at
temperature 1 from the policy; from allie.search coverage's calibrated human-move distribution after
a fixed number of simulations; or from the policy tilted by the searched values, pi ~ prior
exp(beta Q), after a fixed number of coverage simulations or lookahead calls.

Chosen for the annealed Allie 2.0 on the golden evaluation's July 2026 human games (up to 5,000
positions per time control and bin, 800-2600; analysis/elo_strength/calib_cells.py): per cell, among
the searches whose human-move cross-entropy is below the policy's at 95% confidence (standard errors
clustered by game), the cheapest whose expected move accuracy and blunder rate (Stockfish scoring
every legal move) are within 20 Elo of the closest to the humans'; the policy where none qualifies,
and in bullet, whose think times a search outlasts. The searching cells' errors (Elo, + stronger than
the humans: accuracy / blunder rate) and cross-entropy gaps are in the comments. Classical 2400 and
2600 search 1,024 simulations (about 25 s alone on 6 CPUs), which a tenth of the clock allows with
about 260 s left; with less they step down to 256.

A search is sized for a tenth of the clock left above the reserve at COST seconds a simulation or
call: the budget steps down its searcher's ladder until it fits. A search still running at a fifth
of the clock stops, and the move comes from the policy.
"""

import numpy as np

# (speed, bin) -> (searcher, budget, beta): beta None is coverage's calibrated distribution; a cell
# not listed plays the policy
CELLS = {
    ("blitz", 1800): ("coverage", 128, 0.5),  # -67 / -87, -0.0019
    ("blitz", 2000): ("coverage", 128, None),  # -34 / -63, -0.0073
    ("blitz", 2200): ("coverage", 128, 2.0),  # -40 / -7, -0.0074
    ("blitz", 2400): ("coverage", 128, 3.0),  # +12 / -27, -0.0103
    ("blitz", 2600): ("coverage", 256, 3.0),  # -16 / -60, -0.0155
    ("rapid", 1600): ("coverage", 128, 0.5),  # -53 / -36, -0.0017
    ("rapid", 1800): ("coverage", 32, 1.0),  # -53 / +29, -0.0034
    ("rapid", 2000): ("coverage", 128, 1.0),  # -60 / -66, -0.0049
    ("rapid", 2200): ("coverage", 32, 3.0),  # +8 / +14, -0.0082
    ("rapid", 2400): ("coverage", 128, 6.0),  # +16 / +62, -0.0192
    ("rapid", 2600): ("coverage", 256, 8.0),  # -40 / +24, -0.0307
    ("classical", 1600): ("coverage", 128, 0.5),  # -45 / -78, -0.0026
    ("classical", 1800): ("coverage", 32, 1.0),  # -22 / -44, -0.0042
    ("classical", 2000): ("coverage", 32, 4.0),  # +10 / +5, -0.0113
    ("classical", 2200): ("coverage", 128, 6.0),  # -49 / +2, -0.0189
    ("classical", 2400): ("coverage", 1024, 8.0),  # -73 / -81, -0.0441
    ("classical", 2600): ("coverage", 1024, 16.0),  # -131 / -197, -0.0322
}
LADDER = dict(coverage=(8, 32, 128, 256, 1024), lookahead=(1, 2, 4, 8, 16))
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
