"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, predicting human moves no worse than
its policy does. Per time control and 200-point rating bin (CELLS), the move is sampled at
temperature 1 from the policy; from allie.search coverage's calibrated human-move distribution after
a fixed number of simulations; or from the policy tilted by the searched values, pi ~ prior
exp(beta Q), after a fixed number of coverage simulations or lookahead calls.

The move's think time is drawn first (behaviour.think) and caps the search with a tenth of the clock
left above the reserve, at COST seconds a simulation or call: the largest of the cell's rungs that
fits is searched, with its beta, and the policy plays when none fits. So the bot keeps a human pace,
searching most on the moves a human would think longest about (COST: 15 ms a simulation, about the
median of 1,024-simulation searches in a 6-CPU load test of 10 games, 14.3-14.6 s). A search still
running at a fifth of the clock stops, and the policy plays.

Chosen for the annealed Allie 2.0 on the golden evaluation's July 2026 human games (up to 5,000
positions per time control and bin, 800-2600; analysis/elo_strength/calib_cells.py --think), scoring
each cell as the bot plays it: per position, the rungs mixed by the think-time draws. Per cell,
among the rung sets and betas whose human-move cross-entropy is below the policy's at 95% confidence
(standard errors clustered by game), the cheapest whose expected move accuracy and blunder rate
(Stockfish scoring every legal move) are within 20 Elo of the closest to the humans'; the policy
where none qualifies, and in bullet, whose think times a search outlasts.
"""

import numpy as np

# (speed, bin) -> (searcher, {rung: beta}): the largest rung that the move's think time and the clock
# allow is searched, beta None for coverage's calibrated distribution; a cell not listed plays the
# policy. Comments: the cell's errors (Elo, accuracy / blunder rate) and cross-entropy gap as the bot
# plays it, its rungs mixed by the think times it draws
# fmt: off
CELLS = {
    ("blitz", 1800): ("coverage", {32: 0.5, 8: 0.5}),  # -87 / -107, -0.0020
    ("blitz", 2000): ("coverage", {128: None, 32: None, 8: None}),  # -49 / -79, -0.0073
    ("blitz", 2200): ("coverage", {256: 2.0, 128: 2.0, 32: 2.0, 8: 2.0}),  # -62 / -30, -0.0088
    ("blitz", 2400): ("coverage", {32: 4.0, 8: 4.0}),  # -16 / -73, -0.0117
    ("blitz", 2600): ("coverage", {256: 4.0, 128: 4.0, 32: 4.0, 8: 4.0}),  # -26 / -90, -0.0209
    ("rapid", 1600): ("coverage", {128: 0.5, 32: 0.5, 8: 0.5}),  # -57 / -40, -0.0018
    ("rapid", 1800): ("coverage", {32: 1.0, 8: 1.0}),  # -55 / +27, -0.0033
    ("rapid", 2000): ("coverage", {128: 1.0, 32: 1.0, 8: 1.0}),  # -65 / -71, -0.0053
    ("rapid", 2200): ("coverage", {32: 3.0, 8: 3.0}),  # +2 / +9, -0.0099
    ("rapid", 2400): ("coverage", {128: 6.0, 32: 6.0, 8: 6.0}),  # -11 / +15, -0.0275
    ("rapid", 2600): ("coverage", {256: 8.0, 128: 8.0, 32: 8.0, 8: 8.0}),  # -73 / -45, -0.0423
    ("classical", 1600): ("coverage", {32: 0.5, 8: 0.5}),  # -62 / -91, -0.0025
    ("classical", 1800): ("coverage", {32: 1.0, 8: 1.0}),  # -23 / -45, -0.0042
    ("classical", 2000): ("coverage", {32: 4.0, 8: 4.0}),  # +6 / +1, -0.0121
    ("classical", 2200): ("coverage", {128: 6.0, 32: 6.0, 8: 6.0}),  # -57 / -12, -0.0234
    ("classical", 2400): ("coverage", {1024: 12.0, 256: 12.0, 128: 12.0, 32: 12.0, 8: 12.0}),  # -41 / -56, -0.0247
    ("classical", 2600): ("coverage", {1024: 16.0, 256: 16.0, 128: 16.0, 32: 16.0, 8: 16.0}),  # -188 / -304, -0.0466
}
# fmt: on
LADDER = dict(coverage=(8, 32, 128, 256, 1024), lookahead=(1, 2, 4, 8, 16), kl=(8, 32, 128, 256, 512, 1024))
# seconds a simulation or call (6-cpu load test, 10 games); kl a leaf, coverage's until a load test (alone on 4
# cpus 9.5 ms)
COST = dict(coverage=0.015, lookahead=0.35, kl=0.015)


def cell(rating, speed):
    """(searcher, {rung: beta}) for a player of `rating` at `speed`, or None: the policy."""
    return CELLS.get((speed, min(max(int(rating) // 200 * 200, 800), 2600)))


def limit(clock, reserve=1.0):
    """Seconds a search may take: a fifth of the clock left above the reserve (None: no clock)."""
    return np.inf if clock is None else max(clock - reserve, 0.0) / 5


def affordable(searcher, rungs, clock, reserve=1.0, think=None):
    """The largest of `rungs` that fits both a tenth of the clock left above the reserve (half of
    limit()) and the move's think time at COST seconds a simulation or call; 0: none."""
    fits = min(limit(clock, reserve) / 2, np.inf if think is None else think)
    return max((b for b in rungs if b * COST[searcher] <= fits), default=0)


def tilt(prior, q, beta):
    """The searched distribution: pi ~ prior exp(beta Q)."""
    z = np.log(np.maximum(prior, 1e-300)) + beta * np.asarray(q, float)
    z = np.exp(z - z.max())
    return z / z.sum()
