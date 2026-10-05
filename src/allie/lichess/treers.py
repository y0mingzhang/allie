"""allie.search's coverage search on the Rust tree (allie_fast.Coverage: tree.cpp's forest, value.cpp's backup,
tree.py Nodes' boards and clocks), with tree.Coverage's interface and outputs. The nodes are evaluated through
tree.Nodes on any backend, so a search's steps join the engine's batches as before; the output policy is
allie.search.policy's with the same calibration.
"""

import json
from time import monotonic

import allie_fast
import numpy as np
from scipy.special import softmax

from allie.search.policy import policy

from . import tree
from .tokens import MOVE_START, MOVES
from .tree import CALIBRATION, POLICY, Late, Mixed, Tree, View


class Coverage:
    """search(game, simulations, deadline) -> (legal moves, searched human-move probabilities, their prior,
    their searched values for the mover), or None if the search runs past deadline (time.monotonic()).
    views as tree.Coverage's: every node read under each, its W/D/L their mean."""

    def __init__(self, threads=4, views=("true",)):
        self.parameters = json.loads(CALIBRATION.read_text())
        bp = self.parameters["budget_policies"]
        for b, key in POLICY.items():
            bp.setdefault(str(b), bp[key])
        self.threads, self.views = threads, tuple(views)

    def __call__(self, game, simulations, deadline=np.inf):
        own = game.__dict__.setdefault("views", {})
        games = [
            game if v == "true" else own.get(v) or own.setdefault(v, View(game, v))
            for v in self.views
        ]
        zs = [g.sync() for g in games]
        if (
            self.views != ("true",) and monotonic() > deadline
        ):  # a view's first sync prefills the whole game
            return None
        z = game.sync()
        feats = np.array(game.features(), np.float32)
        concurrent = game.engine.model.fast is not None

        def run():
            trees = [
                Tree(g, x, capacity=4 * simulations + 256) for g, x in zip(games, zs)
            ]
            for t in trees:
                t.concurrent, t.deadline = concurrent, deadline
            oracle = (
                Mixed(trees, z.double().numpy())
                if self.views != ("true",)
                else trees[0]
            )
            return self.search(game, oracle, feats, simulations)

        try:
            return run() if concurrent else game.engine.run(run)
        except Late:
            return None

    def search(self, game, oracle, feats, simulations):
        """Search._batch for one coverage root on the Rust tree: the nodes through the oracle's handles, the
        values by value.cpp's backup, the output distribution by the calibrated policy."""
        par = self.parameters
        oracle.reset()
        nodes = oracle.handles([game.tokens], [feats], "predicted")
        root = np.asarray(nodes.root_logits)
        legal = allie_fast.Position.from_tokens(game.tokens).legal()
        budget = 0 if len(legal) == 1 else simulations  # a forced move is not searched
        inc = game.inc if game.inc is not None else -1
        cov = allie_fast.Coverage(
            list(game.tokens), root[0], feats, inc, budget, par["cpuct"], "predicted"
        )
        while not cov.done:
            h = cov.select()
            if len(h):
                cov.update(nodes(h))
        if cov.evals[0] != nodes.queries:
            raise RuntimeError("node accounting mismatch")
        ids = np.array([legal], np.int64) - MOVE_START
        b = par["backup"]
        q = cov.reduce(np.log(b["tau"]), b["exponent"], b["count_scale"], budget, ids)[
            0
        ]
        mask = np.ones_like(ids, bool)
        logits = root[:, 378:2346][np.arange(1)[:, None], ids].astype(float)
        prior = softmax(np.where(mask, logits, -np.inf), axis=1)
        params = (
            par["old_parameters"]["unchanged"]
            if simulations in (64, 1000)
            else par["budget_policies"][str(budget or 128)]
        )
        row = dict(
            prefix=list(game.tokens), cell=game.cell(game.elo[len(game.moves) % 2])
        )
        p = policy([row], root, q, ids, mask, np.array([feats[-1, 0]]), params)
        return (
            [MOVES[t - MOVE_START] for t in legal],
            p[0].copy(),
            prior[0].copy(),
            q[0].copy(),
        )


class KL:
    """search(game, leaves, deadline) -> (legal moves by falling prior, their prior, their searched values for the
    mover), or None past the deadline: tree.KL on the Rust forest (allie_fast.KL: kl.py's Forest, grow, backup and
    root_q), the nodes evaluated through tree.Nodes under each view."""

    GROW, READ = tree.KL.GROW, tree.KL.READ

    def __init__(self, grow=None, read=None, clock_rule="zero", views=("true",)):
        self.grow, self.read, self.clock_rule = (
            grow or self.GROW,
            read or self.READ,
            clock_rule,
        )
        self.views = tuple(views)
        assert self.views == ("true",) or clock_rule == "zero", (
            "views keep their own clocks only on the zero rule"
        )

    def __call__(self, game, leaves, deadline=np.inf):
        own = game.__dict__.setdefault("views", {})
        games = [
            game if v == "true" else own.get(v) or own.setdefault(v, View(game, v))
            for v in self.views
        ]
        zs = [g.sync() for g in games]
        if (
            self.views != ("true",) and monotonic() > deadline
        ):  # a view's first sync prefills the whole game
            return None
        z = game.sync()
        feats = [np.array(g.features(), np.float32) for g in games]
        concurrent = game.engine.model.fast is not None

        def run():
            trees = [Tree(g, x, capacity=2 * leaves + 256) for g, x in zip(games, zs)]
            bridges = []
            for t, g, f in zip(trees, games, feats):
                t.concurrent, t.deadline = concurrent, deadline
                bridges.append(t.handles([g.tokens], [f], self.clock_rule))
            prior = (
                None if self.views == ("true",) else z.double().numpy()
            )  # the game's own prior
            inc = games[0].inc if games[0].inc is not None else -1
            forest = allie_fast.KL(list(game.tokens), [b.root_logits[0] for b in bridges], prior, feats[0], inc,
                                   trees[0].capacity, self.clock_rule)  # fmt: skip
            while (h := forest.select(leaves, **self.grow)) is not None:
                if len(h):
                    forest.update([np.asarray(b(h), float) for b in bridges])
            return forest.root_q(**self.read)

        try:
            moves, prior, q = run() if concurrent else game.engine.run(run)
        except Late:
            return None
        return [MOVES[t - MOVE_START] for t in moves], prior, q
