"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, predicting human moves no worse than
its policy does in any time control and rating. One rule for every rating and time control, no
thresholds: the move's think time is drawn first (behaviour.think); a KL search (treers.KL, GROW and
READ) evaluates the largest rung of LADDER that fits both that time less MARGIN and a tenth of the
clock above the reserve, a leaf priced at COST or at the bot's recent measured cost when higher (up to
4 COST), and the move is sampled at temperature 1 from the policy tilted by the searched values,
pi ~ prior exp(beta v), with

    beta = BETA0 exp(GAMMA (R - 1700) / 1000) (t / 10 s)^DELTA,

t the position's expected human think time (the think-time head's mean). Stronger players and harder
positions get a sharper tilt, fast ones (bullet) little. The search reads every node under one fixed
view, VIEWS: both players rated 3000 at a classical header with no clocks (the root's prior stays the
game's own); it grows best-first on log-odds values, its root steered toward the moves the tilted output
is unsure of, and its values v are log odds, arctanh(0.98 (W - L)). A search ends by the think time less
MARGIN (or a fifth of the clock) with the tree it has, its last call cut to the leaves that fit: no search
runs past the drawn think time.
"""

import numpy as np

BETA0, GAMMA, DELTA = 0.297, 3.0, 0.625
GROW = dict(own=8.0, opp=8.0, soft=True, squash=0.95, root=8.0)
READ = dict(own=8.0, opp=8.0, soft=True, squash=0.98)
VIEWS = ("r3000/tc1800+20/noclock",)
LADDER = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
COST = 0.004  # seconds a KL leaf (16 cpus, 12 threads, a lone search of 128 leaves; larger ones cost less)
MEASURED = 128  # searches of fewer leaves cost more a leaf (each call's fixed costs) and leave the price be
MARGIN = 0.2  # seconds a search ends before the think time (an engine step under load is <= 0.17 s)
_BINS = np.arange(63)
SECONDS = np.where(
    _BINS < 16, _BINS + 0.5, 16 * np.exp((_BINS - 16) / 7.06)
)  # think-time head bins


def beta(rating, time):
    """The tilt for a player of `rating` at a position whose think-time head is `time` (63 bins)."""
    t = float(np.asarray(time, float) @ SECONDS / np.sum(time))
    return BETA0 * np.exp(GAMMA * (rating - 1700) / 1000) * (t / 10) ** DELTA


def limit(clock, reserve=1.0):
    """Seconds a search may take: a fifth of the clock left above the reserve (None: no clock)."""
    return np.inf if clock is None else max(clock - reserve, 0.0) / 5


def affordable(clock, reserve=1.0, think=np.inf, cost=COST):
    """The largest rung of LADDER that fits both a tenth of the clock left above the reserve (half
    of limit()) and `think` seconds at `cost` seconds a leaf; 0: none."""
    fits = min(limit(clock, reserve) / 2, think)
    return max((b for b in LADDER if b * cost <= fits), default=0)


def tilt(prior, v, beta):
    """The searched distribution: pi ~ prior exp(beta v), v the searched values' log odds."""
    z = np.log(np.maximum(prior, 1e-300)) + beta * np.asarray(v, float)
    z = np.exp(z - z.max())
    return z / z.sum()
