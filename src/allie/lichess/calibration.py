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

Fitted for the annealed Allie 2.0 on the golden evaluation's July 2026 human games (198,794 positions,
up to 5,000 per time control and 200-point bin, 800-2600; analysis/elo_strength/calib_unified.py on the
GPU export of this search at every rung up to each position's largest affordable one), scoring each
position as the bot plays it (its rungs mixed by the think-time draws, MARGIN included): the debiased
squared Elo error of move accuracy and blunder rate (Stockfish on every legal move) summed over the
blitz, rapid and classical bins, with every bin's human-move cross-entropy at or below the policy's at
95%, DELTA from 0, 0.25, 0.5, 0.75, 1 and 1.5. Each game half's fit picks these same values. Held out
(each half scored by the other's fit), RMS Elo error bullet 0, blitz 0, rapid 55, classical 0 (debiased:
within the noise); cross-entropy against the policy's bullet -0.0002, blitz -0.0038, rapid -0.0071,
classical -0.0101 nats, no bin above it at 95%. 8 of the 80 strength tests (accuracy and blunder rate
per bin) fall outside the noise at 95%, about 4 expected by chance, in 6 bins (Elo on accuracy / blunder
rate): bullet 2600 +115 / +219, blitz 2600 +112 / +91, rapid 2200 +87 / +123, rapid 2600 +39 / +158,
classical 1800 +76 / +38, and bullet 1200 -431 / -334, where the human curve is flat. Classical 2600 plays -34 / -50 (the coverage rule this replaces:
-135 / -235).
"""

import numpy as np

BETA0, GAMMA, DELTA = 0.206, 3.0, 0.75
GROW = dict(own=8.0, opp=8.0, soft=True, squash=0.95, root=8.0)
READ = dict(own=8.0, opp=8.0, soft=True, squash=0.98)
VIEWS = ("r3000/tc1800+20/noclock",)
LADDER = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
COST = 0.004  # seconds a KL leaf (16 cpus, 12 threads, a lone search of 128 leaves; larger ones cost less)
MEASURED = 128  # searches of fewer leaves evaluated cost more a leaf (each call's fixed costs) and leave the price be
MARGIN = 0.3  # seconds a search ends before the think time (10-game load tests: a late request and the move's own work)
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
