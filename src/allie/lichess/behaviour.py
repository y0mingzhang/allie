"""When the bot thinks, resigns and agrees to draws, calibrated on held-out human games.

analysis/lichess/human.py fits behaviour.json on half of the main evaluation's July 2026 games
(every clock known) and checks it on the other half:
- resignation: a per-position hazard for each player, on its turn (instead of moving) or right
  after its own move, logistic in the model's win / draw / loss probabilities for that player,
  the ply, its rating, the format and its clock; never below the P(loss) under which humans
  almost never resign (the 5th percentile of their resignations), nor before ply 20 (6% of human
  resignations come earlier, at a median P(loss) of 0.74: opening blunders that could be played on;
  live, the bot resigned at move 3);
- draws: a hazard of the same form for ending a game by agreement on one's turn (the data show
  agreements, not offers, so this is a heuristic policy), and the expected score up to which
  humans agree, from ply 30 (83% of human agreements come later; live, a bot that accepted any
  offer in a level opening was offered draws at move 3);
- think time: a soft guard from human time use, under a hard cap that keeps a reserve.
"""

import json
import math
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("behaviour.json")
PARAMETERS = json.loads(PATH.read_text()) if PATH.exists() else {}  # empty only while fitting
# allie.search's formats: bullet, blitz, rapid, classical
FORMATS = dict(ultraBullet=0, bullet=0, blitz=1, rapid=2, classical=3, correspondence=3)
KEYS = ("resign", "draw", "guard")


def check():
    missing = [k for k in KEYS if k not in PARAMETERS]
    if missing:
        raise RuntimeError(f"{PATH} lacks {missing}: run analysis/lichess/human.py fit")


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def features(wdl, ply, elo, fmt, clock, base, on_turn=True):
    """The hazards' inputs. wdl: the player's own win / draw / loss probabilities; ply: the
    position's (moves played); fmt: 0..3; clock (that player's), base: seconds or None."""
    w, d, loss = wdl
    left = 1.0 if not base or clock is None else min(max(clock / base, 0.0), 1.5)
    x = [
        1.0,
        logit(loss),
        logit(w),
        logit(d),
        min(ply, 200) / 100,
        (elo - 1500) / 500,
        left,
    ]
    return x + [float(fmt == f) for f in (0, 2, 3)] + [float(on_turn)]


def resign_features(wdl, ply, elo, fmt, clock, base, on_turn, knots):
    """features() plus a piecewise-linear P(loss) (hinges in logit P(loss) at the knots, on
    and off turn): human resignation rises steeply between P(loss) 0.85 and 0.99 and falls
    again when mate is imminent, which a single logit cannot follow."""
    x = features(wdl, ply, elo, fmt, clock, base, on_turn)
    z = logit(wdl[2])
    hinge = [max(0.0, z - logit(k)) for k in knots]
    return x + hinge + [float(on_turn) * v for v in hinge]


def hazard(name, x):
    z = float(np.dot(PARAMETERS[name]["coef"], x))
    return 1 / (1 + math.exp(-min(max(z, -50), 50)))


def think(time_probs, rng, clock, increment, ply, lag=0.0):
    """Seconds to spend on this move, compute included: a draw from the think-time head (bin,
    then uniform within it), less the lag the network and server add (human times are server
    side, their lag included), under a soft guard fitted on human time use (a share of the
    clock plus the increment) and a hard cap that always leaves the reserve on the clock."""
    if ply < 2:  # no think time is recorded for each side's first move; humans move fast
        t = float(rng.uniform(0.5, 2.0))
    else:
        p = np.asarray(time_probs, np.float64)
        b = int(rng.choice(len(p), p=p / p.sum()))
        u = rng.uniform(-0.5, 0.5)
        t = max(b + u, 0.0) if b < 16 else 16 * math.exp((b - 16 + u) / 7.06)
        t = max(t - lag, 0.0)
    if clock is None:
        return t
    return min(t, float(guard(clock, increment or 0, **PARAMETERS["guard"])))


def guard(clock, increment, reserve, share):
    """The longest a move may take: `share` of the clock above the reserve plus the increment (half the share
    under a minute: live, single long thinks late in bullet games and the lag of the moves after them flagged
    the bot), and never the clock above the reserve (the increment comes after the move). Arrays too."""
    hard = np.maximum(clock - reserve, 0.0)
    return np.minimum(hard * share * np.clip(clock / 120, 0.5, 1.0) + increment, hard)


def resign(wdl, ply, elo, fmt, clock, base, rng, on_turn=True):
    """Resign now, as a human of this rating would here? on_turn: instead of moving; otherwise
    right after one's own move, with wdl that player's view of the new position."""
    r = PARAMETERS["resign"]
    if ply < r["min_ply"] or wdl[2] < r["floor"]:
        return False
    x = resign_features(wdl, ply, elo, fmt, clock, base, on_turn, r["knots"])
    return rng.random() < hazard("resign", x)


def offer_draw(wdl, ply, elo, fmt, clock, base, rng):
    """Offer a draw with this move, as often as humans end games by agreement here."""
    if ply < PARAMETERS["draw"]["min_ply"]:
        return False
    return rng.random() < hazard("draw", features(wdl, ply, elo, fmt, clock, base))


def accept_draw(wdl, ply):
    """Accept an offer from accept_min_ply on, unless clearly better (three quarters of human agreements
    happen at or below this expected score for the better side)."""
    w, d, _ = wdl
    p = PARAMETERS["draw"]
    return ply >= p["accept_min_ply"] and w + d / 2 <= p["accept"]
