"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, predicting human moves no worse than
its policy does. Per time control and 200-point rating bin (CELLS), the move is sampled at
temperature 1 from the policy; from allie.search coverage's calibrated human-move distribution after
a fixed number of simulations; or from the policy tilted by the searched values, pi ~ prior
exp(beta Q) or, on their log-odds, prior exp(beta arctanh(0.99 Q)), after a fixed number of coverage
simulations or lookahead calls.

The move's think time is drawn first (behaviour.think) and caps the search with a tenth of the clock
left above the reserve, at COST seconds a simulation or call: the largest of the cell's rungs that
fits is searched, with its beta, and the policy plays when none fits. So the bot keeps a human pace,
searching most on the moves a human would think longest about (COST: 15 ms a simulation, about the
median of 1,024-simulation searches in a 6-CPU load test of 10 games, 14.3-14.6 s). A search still
running at a fifth of the clock stops, and the policy plays.

Chosen for the annealed Allie 2.0 on the golden evaluation's July 2026 human games (up to 5,000
positions per time control and bin, 800-2600; analysis/elo_strength/calib_cells.py --think --atanh
0.99), scoring each cell as the bot plays it: per position, the rungs mixed by the think-time draws.
Per cell, among the rung sets, tilts (on Q or on its log-odds) and betas whose human-move
cross-entropy is below the policy's at 95% confidence (standard errors clustered by game), the
cheapest whose expected move accuracy and blunder rate (Stockfish scoring every legal move) are
within 20 Elo of the closest to the humans'; the policy where none qualifies, in bullet, whose
think times a search outlasts, and where a pick qualifies only at the gate's edge and neither half
of the games, picking on its own, confirms it on the other (blitz 1600, classical 1000).
"""

import numpy as np

# (speed, bin) -> (searcher, {rung: beta}[, "atanh"]): the largest rung that the move's think time
# and the clock allow is searched, beta None for coverage's calibrated distribution, "atanh" for a
# tilt on the values' log-odds; a cell not listed plays the policy. Comments: the cell's errors (Elo,
# accuracy / blunder rate) and cross-entropy gap as the bot plays it, its rungs mixed by the think
# times it draws
# fmt: off
CELLS = {
    ("blitz", 1800): ("coverage", {32: 0.75, 8: 0.75}),  # -41 / -63, -0.0022
    ("blitz", 2000): ("coverage", {32: 1.5, 8: 1.5}),  # +0 / -34, -0.0041
    ("blitz", 2200): ("coverage", {128: 1.5, 32: 1.5, 8: 1.5}, "atanh"),  # -50 / -25, -0.0078
    ("blitz", 2400): ("coverage", {32: 4.0, 8: 4.0}),  # -16 / -73, -0.0117
    ("blitz", 2600): ("coverage", {128: 3.0, 32: 3.0, 8: 3.0}, "atanh"),  # -12 / -79, -0.0179
    ("rapid", 1600): ("coverage", {128: 0.5, 32: 0.5, 8: 0.5}),  # -57 / -40, -0.0018
    ("rapid", 1800): ("coverage", {32: 1.0, 8: 1.0}),  # -55 / +27, -0.0033
    ("rapid", 2000): ("coverage", {32: 1.5, 8: 1.5}),  # -25 / -33, -0.0049
    ("rapid", 2200): ("coverage", {32: 2.0, 8: 2.0}, "atanh"),  # -2 / +0, -0.0146
    ("rapid", 2400): ("coverage", {128: 4.0, 32: 4.0, 8: 4.0}, "atanh"),  # -10 / +15, -0.0276
    ("rapid", 2600): ("coverage", {128: 6.0, 32: 6.0, 8: 6.0}, "atanh"),  # -37 / +9, -0.0216
    ("classical", 1600): ("coverage", {128: 0.75, 32: 0.75, 8: 0.75}),  # +6 / -41, -0.0028
    ("classical", 1800): ("coverage", {32: 0.75, 8: 0.75}, "atanh"),  # -18 / -44, -0.0047
    ("classical", 2000): ("coverage", {32: 4.0, 8: 4.0}),  # +6 / +1, -0.0121
    ("classical", 2200): ("coverage", {128: 4.0, 32: 4.0, 8: 4.0}, "atanh"),  # -30 / +20, -0.0276
    ("classical", 2400): ("coverage", {1024: 6.0, 256: 6.0, 128: 6.0, 32: 6.0, 8: 6.0}, "atanh"),  # -26 / -35, -0.0395
    ("classical", 2600): ("coverage", {1024: 6.0, 256: 6.0, 128: 6.0, 32: 6.0, 8: 6.0}, "atanh"),  # -171 / -263, -0.0394
}
# fmt: on
LADDER = dict(coverage=(8, 32, 128, 256, 1024), lookahead=(1, 2, 4, 8, 16), kl=(8, 32, 128, 256, 512, 1024))
# seconds a simulation or call (6-cpu load test, 10 games); kl a leaf, coverage's until a load test (alone on 4
# cpus 9.5 ms)
COST = dict(coverage=0.015, lookahead=0.35, kl=0.015)


def cell(rating, speed):
    """(searcher, {rung: beta}[, "atanh"]) for a player of `rating` at `speed`, or None: the policy."""
    return CELLS.get((speed, min(max(int(rating) // 200 * 200, 800), 2600)))


def limit(clock, reserve=1.0):
    """Seconds a search may take: a fifth of the clock left above the reserve (None: no clock)."""
    return np.inf if clock is None else max(clock - reserve, 0.0) / 5


def affordable(searcher, rungs, clock, reserve=1.0, think=None):
    """The largest of `rungs` that fits both a tenth of the clock left above the reserve (half of
    limit()) and the move's think time at COST seconds a simulation or call; 0: none."""
    fits = min(limit(clock, reserve) / 2, np.inf if think is None else think)
    return max((b for b in rungs if b * COST[searcher] <= fits), default=0)


def tilt(prior, q, beta, atanh=False):
    """The searched distribution: pi ~ prior exp(beta Q), or with atanh prior exp(beta arctanh(0.99 Q)),
    on the W - L value's log-odds (Q saturates near +-1, so a fixed beta under-tilts clear positions)."""
    q = np.asarray(q, float)
    q = np.arctanh(0.99 * np.clip(q, -1, 1)) if atanh else q
    z = np.log(np.maximum(prior, 1e-300)) + beta * q
    z = np.exp(z - z.max())
    return z / z.sum()
