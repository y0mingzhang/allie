"""When the bot thinks, resigns and agrees to draws, calibrated on held-out human games.

analysis/lichess/human.py fits behaviour.json on July 2026 games with every clock known: a
per-turn resignation hazard (logistic in the model's win / draw / loss probabilities for the side
to move, the ply, its rating, the format and its clock), a draw-offer hazard of the same form, the
expected score at which humans agree draws, and a think-time guard against flagging.
"""

import json
import math
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("behaviour.json")
PARAMETERS = json.loads(PATH.read_text()) if PATH.exists() else {}  # absent only while fitting
# allie.search's formats: bullet, blitz, rapid, classical
FORMATS = dict(ultraBullet=0, bullet=0, blitz=1, rapid=2, classical=3, correspondence=3)
CENTRES = np.r_[np.arange(16), 16 * np.exp(np.arange(47) / 7.06)]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def features(wdl, ply, elo, fmt, clock, base):
    """The hazards' inputs. wdl: the side to move's win / draw / loss probabilities; fmt: 0..3;
    clock, base: seconds (None if the game has no clock)."""
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
    return x + [float(fmt == f) for f in (0, 2, 3)]


def hazard(name, x):
    z = float(np.dot(PARAMETERS[name]["coef"], x))
    return 1 / (1 + math.exp(-min(max(z, -50), 50)))


def think(time_probs, rng, clock, increment, ply):
    """Seconds to wait before moving: a draw from the think-time head (bin, then uniform
    within it), held under the guard so the clock is never spent below its reserve."""
    if (
        ply < 2
    ):  # no think time is recorded for each side's first move; humans move fast
        return float(rng.uniform(0.5, 2.0))
    p = np.asarray(time_probs, np.float64)
    b = int(rng.choice(len(p), p=p / p.sum()))
    u = rng.uniform(-0.5, 0.5)
    t = max(b + u, 0.0) if b < 16 else 16 * math.exp((b - 16 + u) / 7.06)
    if clock is None:
        return t
    g = PARAMETERS["guard"]
    budget = (
        max(0.0, clock - g["reserve"]) * g["share"] + (increment or 0) * g["increment"]
    )
    return min(t, budget)


def resign(wdl, ply, elo, fmt, clock, base, rng):
    """Resign now, as a human of this rating would at this turn? Never while the model gives
    the side to move a real chance (P(loss) under the floor)."""
    if wdl[2] < PARAMETERS["resign"]["floor"]:
        return False
    return rng.random() < hazard("resign", features(wdl, ply, elo, fmt, clock, base))


def offer_draw(wdl, ply, elo, fmt, clock, base, rng):
    """Offer a draw with this move, as often as humans end games by agreement here."""
    if ply < PARAMETERS["draw"]["min_ply"]:
        return False
    return rng.random() < hazard("draw", features(wdl, ply, elo, fmt, clock, base))


def accept_draw(wdl):
    """Accept an offer unless clearly better: humans agree at expected scores up to this."""
    w, d, _ = wdl
    return w + d / 2 <= PARAMETERS["draw"]["accept"]
