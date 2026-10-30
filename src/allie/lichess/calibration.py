"""Calibrated strength (`play.mode = "calibrated"`): Allie asked to play at rating R makes moves of
the quality an R-rated human makes at the same time control, predicting human moves no worse than
its policy does in any time control and rating. One rule per time control, no rating thresholds: the
move's think time is drawn first (behaviour.think); a KL search (treers.KL, GROW and READ) evaluates
the largest rung of LADDER that fits both that time less MARGIN and a tenth of the clock above the
reserve, a leaf priced at COST or at the bot's recent measured cost when higher (up to 4 COST), and
the move is sampled at temperature 1 from the policy tilted by the searched values,
pi ~ prior exp(beta v), with

    beta = BETA0 exp(GAMMA x) (t / 10 s)^(DELTA + ETA x) + FLOOR,  x = (R - 1700) / 1000,

the constants per time control (RULES), t the position's expected human think time (the think-time
head's mean). Stronger players and harder positions get a sharper tilt, the tilt grows with rating
more steeply where humans think longer (not in classical), and FLOOR is the tilt every rating gets:
Allie 2.1's policy plays 0.5-0.9 accuracy points below the humans at 800-1400, where the exponential
vanishes. The search reads every node under one fixed view, VIEWS: both players rated 3000 at a
classical header with no clocks (the root's prior stays the game's own); it grows best-first on
log-odds values, its root steered toward the moves the tilted output is unsure of, and its values v
are log odds, arctanh(0.98 (W - L)). A search ends by the think time less MARGIN (or a fifth of the
clock) with the tree it has, its last call cut to the leaves that fit: no search runs past the drawn
think time.

Fitted for Allie 2.1 on the golden evaluation's July 2026 human games (198,794 positions, up to
5,000 per time control and 200-point bin, 800-2600; analysis/elo_strength/calib_unified.py
--policy extra --by-format --loss z --bullet-from 800 --ce cells --ce-z 1.96 --sim 0.006 on the GPU
export of this search at every rung up to 1,024 leaves), scoring each position as the bot plays it
(its rungs mixed by the think-time draws at COST, MARGIN included), each time control on its own
bins: the debiased squared error of move accuracy and blunder rate (Stockfish on every legal move)
in standard errors over every bin, with every bin's human-move cross-entropy at or below the
policy's at 95%, on a grid of BETA0 0.1-0.6 (9 points), GAMMA 2-3, DELTA 0.5-1, ETA 0-0.5 and FLOOR
0-0.6. One rule for every time control misses 10-21 of the 80 strength tests (accuracy and blunder
rate per bin) held out: Allie 2.1's raw strength rises with rating more steeply than Allie 2.0's,
and one exponential in rating fits only one end. Held out (each game half scored by the other's
fit): 5 of 80 tests outside the noise at 95%, about 4 expected by chance (bullet 800 -908 / -728 Elo
on accuracy / blunder rate, bullet 2000 +180 on accuracy, rapid 1400 +121 on blunder rate, classical
1800 +62 on accuracy); RMS Elo error bullet 292, blitz 41, rapid 92, classical 0 (debiased; bullet's
flat human curve inflates its Elo); cross-entropy against the policy's 0.0000, -0.0014, -0.0058 and
-0.0026 nats, no bin above it by each half's errors (bullet 2000 is above it with the halves pooled,
in every rule tried: the policy already plays 0.2 points strong there). A bullet-only refit on a
wider grid (BETA0 from 0.02, GAMMA to 4, FLOOR 0-0.1) picks the same tilt, a near constant 0.1;
bullet without a tilt passes bullet 2000 on September but misses 4 more tests on July (800-1200,
2200, 2600). On September 2026 games (197,191 positions, none in the fit): 11 of 80 tests outside
the noise, in 8 bins: bullet 800 -497 / -633, bullet 2000 +245 on blunder rate, blitz 800 -417 /
-254, blitz 2200 +90 / +167, rapid 2400 -42 and rapid 2600 -38 on accuracy, classical 1200 -93 on
accuracy, classical 1800 +67 on blunder rate; RMS Elo error 157, 95, 49 and 19; cross-entropy
against the policy's -0.0002, -0.0018, -0.0074 and -0.0080 nats, no bin above it at 95%. Allie 2.0
with its own rule, 0.245 exp(2.5 x) (t / 10 s)^(0.5 + 0.5 x) and a ladder to 4,096 leaves at 4 ms:
2 of 80 held out on July, 8 on September.
"""

import numpy as np

# BETA0, GAMMA, DELTA, ETA, FLOOR by format (behaviour.FORMATS: bullet, blitz, rapid, classical)
RULES = (
    (0.1, 2.0, 1.0, 0.5, 0.1),
    (0.196, 2.5, 1.0, 0.5, 0.2),
    (0.125, 3.0, 0.5, 0.25, 0.1),
    (0.196, 3.0, 0.5, 0.0, 0.0),
)
GROW = dict(own=8.0, opp=8.0, soft=True, squash=0.95, root=8.0)
READ = dict(own=8.0, opp=8.0, soft=True, squash=0.98)
VIEWS = ("r3000/tc1800+20/noclock",)
LADDER = (8, 16, 32, 64, 128, 256, 512, 1024)
COST = 0.006  # seconds a KL leaf (16 cpus, 12 threads, a lone search of 128 leaves; larger ones cost less)
MEASURED = 128  # searches of fewer leaves evaluated cost more a leaf (each call's fixed costs) and leave the price be
MARGIN = 0.3  # seconds a search ends before the think time (10-game load tests: a late request and the move's own work)
_BINS = np.arange(63)
SECONDS = np.where(_BINS < 16, _BINS + 0.5, 16 * np.exp((_BINS - 16) / 7.06))  # think-time head bins


def beta(rating, time, fmt):
    """The tilt for a player of `rating` in format `fmt` (behaviour.FORMATS' index) at a position whose
    think-time head is `time` (63 bins)."""
    b0, g, d, e, c = RULES[fmt]
    t = float(np.asarray(time, float) @ SECONDS / np.sum(time))
    x = (rating - 1700) / 1000
    return b0 * np.exp(g * x) * (t / 10) ** (d + e * x) + c


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
