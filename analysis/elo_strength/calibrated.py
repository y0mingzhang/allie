"""Calibrated strength: Allie asked to play at rating R makes moves of the quality an R-rated human
makes in the same time control. One rule for every time control (CALIBRATION.md, calib_rule.py):

    x = (R - 1700) / 1000
    T = exp(TAU x)
    N = min(CAP, K * t * exp(GAMMA x))     t: the model's predicted human think time (s)
    pi ~ s_N^(1/T)                         s_0: the policy; s_N: allie.search coverage's calibrated
                                           human-move distribution after N simulations

The time control enters through t: humans think longer in slower games and on harder moves, so the
search does too. N is rounded onto LADDER at random, in log2(1 + N), so the expected move quality is
exactly the interpolation the fit used.
"""

import numpy as np

TAU, K, GAMMA, CAP = (
    -0.1,
    22.6,
    0.0,
    128,
)  # provisional (early fit); calib_rule.py sets them
LADDER = (0, 8, 32, 128, 256)
_U = np.log2(1 + np.array(LADDER))
_BINS = np.arange(63)
SECONDS = np.where(
    _BINS < 16, _BINS + 0.5, 16 * np.exp((_BINS - 16) / 7.06)
)  # think-time head bins


def temperature(rating):
    return float(np.exp(TAU * (rating - 1700) / 1000))


def budget(rating, think):
    """Simulations (continuous) for a move a human of `rating` is expected to think `think` s on."""
    return float(min(CAP, K * think * np.exp(GAMMA * (rating - 1700) / 1000)))


def ladder(n):
    """[(budget, weight)]: n's two neighbours on LADDER, weighted linearly in log2(1 + n)."""
    i = float(np.interp(np.log2(1 + n), _U, np.arange(len(_U))))
    lo = int(np.floor(i))
    w = i - lo
    return [(LADDER[lo], 1 - w)] + ([(LADDER[lo + 1], w)] if w > 0 else [])


def pick(n, rng):
    """A LADDER budget for continuous n, at random so its expectation is ladder(n)'s."""
    (a, _), *rest = ladder(n)
    return rest[0][0] if rest and rng.random() < rest[0][1] else a


def think_seconds(probabilities):
    """Expected think time (s) under the think-time head's 63-bin distribution."""
    p = np.asarray(probabilities, float)
    return float(p @ SECONDS / p.sum())


def distribution(p, t):
    """p^(1/t), normalized (p: the policy or the search's distribution, any scale)."""
    z = np.log(np.maximum(np.asarray(p, float), 1e-300)) / t
    z = np.exp(z - z.max())
    return z / z.sum()


# --- the mode on allie.lichess's Game (pinned 7e39e50 here; the bot's MODES interface in the port) ---


def coverage(game, n, legal):
    """The search's calibrated distribution over `legal` (UCI strings) after n simulations:
    allie.search coverage on the game's cache with the bot's calibration, as tree.Coverage."""
    import json

    from allie.data.vocab import MOVES
    from allie.lichess.tree import CALIBRATION, Tree
    from allie.search import Search
    from allie.search.native import from_prefix

    par = json.loads(CALIBRATION.read_text())
    par["budget_policies"].setdefault(str(n), par["budget_policies"]["128"])
    z = game.sync()
    feats = np.array(game.features(), np.float32)
    row = dict(prefix=list(game.tokens), cell=game.cell(game.elo[len(game.moves) % 2]),
               legal=from_prefix(np.asarray(game.tokens)).legal())  # fmt: skip

    def run():
        tree = Tree(game, z, capacity=4 * n + 256)
        s = Search(tree, threads=4, calibration=par)
        return s._batch(
            [row], [feats], "coverage", n, "predicted", False, 0.9, 2.0, 1.25
        )[0]

    out = game.engine.run(run)
    by = {MOVES[t - 378]: v for t, v in zip(out["tokens"], out["probabilities"])}
    return np.array([by[m] for m in legal])


def decide(game, rating, clock):
    """(move, think seconds, search budget) for the calibrated mode on an allie.lichess Game."""
    import torch
    from allie.data.vocab import MOVE_ID
    from allie.lichess.engine import TIME, Play

    z = game.sync().double()
    legal = [m.uci() for m in game.board.legal_moves]
    p = torch.softmax(z[torch.tensor([MOVE_ID[m] for m in legal])], 0).numpy()
    think = think_seconds(torch.softmax(z[TIME], 0).numpy())
    n = pick(budget(rating, think), game.rng) if len(legal) > 1 else 0
    pi = distribution(coverage(game, n, legal) if n else p, temperature(rating))
    return legal[int(game.rng.choice(len(pi), p=pi))], game.think(z, Play(), clock), n
