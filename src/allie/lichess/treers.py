"""allie.search's coverage and KL searches on the Rust trees (allie_fast.Coverage: tree.cpp's forest, value.cpp's
backup, tree.py Nodes' boards and clocks; allie_fast.KL: kl.py's forest), with tree.Coverage's and tree.KL's
interfaces and outputs. On the Rust backend the whole loop is native: the tree evaluates its leaves through the
model's Server as path items on the game's cache and its own slot buffers, no Python per leaf, its steps merged
with the other games'. On the torch backend the leaves go through tree.Nodes as before. The output
policy is allie.search.policy's with the same calibration.

Tree reuse (reuse=True, native only): a game keeps each searcher's last tree (`game.trees`); at its next turn,
if the two plies since form a path in it, the search starts from that grandchild's subtree (allie_fast's
reroot: nodes, visits, values and slot rows kept) and evaluates only the budget's remaining leaves. The kept
leaves were evaluated under the clocks the search assumed for the two plies (predicted think times under
"predicted", none under KL's "zero") while the cache now holds the real ones: an approximation inherent to
reuse under either rule, which the calibration refits for. There is no evaluation cache keyed by position: the
model's output depends on the move sequence and the clocks, not on the position alone, so one would change the
outputs. Every search leaves its counts in `game.last_search` (reused, evaluated, ...).
"""

import json
import logging
from time import monotonic

import allie_fast
import numpy as np
from scipy.special import softmax

from allie.search.policy import policy

from . import tree
from .fastrs import RustFast
from .tokens import CONTEXT, MOVE_START, MOVES
from .tree import CALIBRATION, POLICY, WDL, Late, Mixed, Tree, View

log = logging.getLogger(__name__)


def server(game):
    """The model's Server when it runs on the Rust engine, else None."""
    fast = game.engine.model.fast
    return fast.server if isinstance(fast, RustFast) else None


def ref(cache):
    """A cache as the native loop reads it: pointers, capacity, rows."""
    return (
        cache.k.data_ptr(),
        cache.v.data_ptr(),
        cache.e.data_ptr(),
        cache.capacity,
        cache.n,
    )


def increment(game):
    return game.inc if game.inc is not None else -1


def games_of(searcher, game):
    """The game under each of the searcher's views, and their synced logits."""
    own = game.__dict__.setdefault("views", {})
    games = [
        game if v == "true" else own.get(v) or own.setdefault(v, View(game, v))
        for v in searcher.views
    ]
    return games, [g.sync() for g in games]


def reusable(game, games, key, reuse):
    """The game's kept tree for `key` when every cache (the game's and its views') holds exactly the rows the tree
    was searched on plus two plies (a clock learnt later that changed an earlier row starts fresh); the game's slot
    for it is cleared, to be set again by keep()."""
    trees = game.__dict__.setdefault("trees", {})
    old = trees.pop(key, None) if reuse else None
    if old is not None and all(
        len(g.used) == len(u) + 2 and g.used[: len(u)] == u
        for g, u in zip(games, old[1])
    ):
        return old[0]
    return None


def keep(game, games, key, tree):
    """Keep a finished tree with the cache rows it was searched on, for reusable()."""
    game.trees[key] = (tree, [list(g.used) for g in games])


class Coverage:
    """search(game, simulations, deadline) -> (legal moves, searched human-move probabilities, their prior,
    their searched values for the mover), or None if the search runs past deadline (time.monotonic()).
    views as tree.Coverage's: every node read under each, its W/D/L their mean. reuse: start from the last
    search's tree when the game followed it (native only). merged: all views' leaves of a call in one step."""

    def __init__(self, threads=4, views=("true",), reuse=False, merged=True):
        self.parameters = json.loads(CALIBRATION.read_text())
        bp = self.parameters["budget_policies"]
        for b, key in POLICY.items():
            bp.setdefault(str(b), bp[key])
        self.threads, self.views = threads, tuple(views)
        self.reuse, self.merged = reuse, merged

    def __call__(self, game, simulations, deadline=np.inf):
        if len(game.tokens) >= CONTEXT:  # no room for a node: the policy plays
            return None
        games, zs = games_of(self, game)
        if (
            self.views != ("true",) and monotonic() > deadline
        ):  # a view's first sync prefills the whole game
            return None
        z = game.sync()
        feats = [np.array(g.features(), np.float32) for g in games]
        own = np.array(
            game.features(), np.float32
        )  # the game's own clocks: the root's think time, the policy's seconds
        legal = allie_fast.Position.from_tokens(game.tokens).legal()
        budget = 0 if len(legal) == 1 else simulations  # a forced move is not searched
        srv = server(game)
        try:
            if srv is None:
                cov, root = self.handles(game, games, zs, z, own, budget, deadline)
            else:
                root = z.double().numpy() if self.views == ("true",) else mixed_root(zs, z)
                cov = self.native(srv, game, games, root, feats, budget, deadline)
        except Late:
            return None
        if cov is None:
            return None
        return self.outputs(game, cov, root, legal, own, budget, simulations)

    def handles(self, game, games, zs, z, feats, budget, deadline):
        """The tree driven by its handles, the leaves evaluated through tree.Nodes (the torch
        backend); feats: the game's own: (the finished tree, its root logits)."""
        concurrent = game.engine.model.fast is not None
        trees = [Tree(g, x, capacity=4 * budget + 256) for g, x in zip(games, zs)]
        for t in trees:
            t.concurrent, t.deadline = concurrent, deadline
        oracle = (
            Mixed(trees, z.double().numpy()) if self.views != ("true",) else trees[0]
        )

        def run():
            oracle.reset()
            nodes = oracle.handles([game.tokens], [feats], "predicted")
            root = np.asarray(nodes.root_logits)[0]
            cov = allie_fast.Coverage(
                list(game.tokens),
                root,
                feats,
                increment(game),
                budget,
                self.parameters["cpuct"],
                "predicted",
            )
            while not cov.done:
                h = cov.select()
                if len(h):
                    cov.update(nodes(h))
            if cov.evals[0] != nodes.queries:
                raise RuntimeError("node accounting mismatch")
            game.last_search = dict(
                searcher="coverage",
                budget=budget,
                reused=0,
                kept_pulls=0,
                evaluated=nodes.queries,
                nodes=cov.stats()["nodes"],
                late=False,
            )
            return cov, root

        return run() if concurrent else game.engine.run(run)

    def native(self, srv, game, games, root, feats, budget, deadline):
        """The whole search in Rust through the Server, from the kept tree when reuse allows; None past the
        deadline."""
        key = ("coverage", self.views)
        cov = reusable(game, games, key, self.reuse)
        if (
            cov is not None
            and cov.reroot(game.tokens[-2:], root, feats, budget) is None
        ):
            cov = None
        if cov is None:
            cov = allie_fast.Coverage(
                list(game.tokens),
                root,
                feats[0],
                increment(games[0]),
                budget,
                self.parameters["cpuct"],
                "predicted",
                self.views != ("true",),
            )
            for g, f in zip(games[1:], feats[1:]):
                cov.add_view(f, increment(g))
        done = cov.run(srv, [ref(g.cache) for g in games], deadline, self.merged)
        s = cov.stats()
        game.last_search = dict(searcher="coverage", budget=budget, reused=s["reused_leaves"], kept_pulls=s["kept_pulls"],
                                evaluated=s["evaluated_leaves"], nodes=s["nodes"], late=not done)  # fmt: skip
        if done and self.reuse:  # an unfinished (late) tree cannot be re-rooted
            keep(game, games, key, cov)
        return cov if done else None

    def outputs(self, game, cov, root, legal, feats, budget, simulations):
        """Search._batch's outputs for the finished tree: the values by value.cpp's backup, the distribution
        by the calibrated policy."""
        par = self.parameters
        ids = np.array([legal], np.int64) - MOVE_START
        b = par["backup"]
        q = cov.reduce(np.log(b["tau"]), b["exponent"], b["count_scale"], budget, ids)[
            0
        ]
        mask = np.ones_like(ids, bool)
        logits = root[None, 378:2346][np.arange(1)[:, None], ids].astype(float)
        prior = softmax(np.where(mask, logits, -np.inf), axis=1)
        params = (
            par["old_parameters"]["unchanged"]
            if simulations in (64, 1000)
            else par["budget_policies"][str(budget or 128)]
        )
        row = dict(
            prefix=list(game.tokens), cell=game.cell(game.elo[len(game.moves) % 2])
        )
        p = policy([row], root[None], q, ids, mask, np.array([feats[-1, 0]]), params)
        return (
            [MOVES[t - MOVE_START] for t in legal],
            p[0].copy(),
            prior[0].copy(),
            q[0].copy(),
        )


def mixed_root(zs, own):
    """MixedNodes' root logits: the views' W/D/L mixed, everything else the game's own."""
    r = tree.mixed([x.double().numpy()[None] for x in zs])
    own = own.double().numpy()
    r[0, : WDL.start], r[0, WDL.stop :] = own[: WDL.start], own[WDL.stop :]
    return r[0]


class KL:
    """search(game, leaves, deadline) -> (legal moves by falling prior, their prior, their searched values for the
    mover), or None past the deadline: tree.KL on the Rust forest (allie_fast.KL: kl.py's Forest, grow, backup and
    root_q), the nodes evaluated under each view. Natively it evaluates in small requests, each sent only if one as
    slow as the last request (of any search by this searcher: the engine's load; its estimate halves every LAG
    seconds without a request) ends by the deadline; a search cut short gives the values of the smaller tree it
    has (`game.last_search["late"]`), None if it evaluated nothing.
    reuse and merged as Coverage's; every search, reused or not, has room for 2 leaves + 256 nodes (a cut is
    logged and left in `game.last_search["full"]`)."""

    GROW, READ = tree.KL.GROW, tree.KL.READ
    LAG = 10.0

    def __init__(
        self,
        grow=None,
        read=None,
        clock_rule="zero",
        views=("true",),
        reuse=False,
        merged=True,
    ):
        self.grow, self.read, self.clock_rule = (
            grow or self.GROW,
            read or self.READ,
            clock_rule,
        )
        self.views, self.reuse, self.merged = tuple(views), reuse, merged
        assert self.views == ("true",) or clock_rule == "zero", (
            "views keep their own clocks only on the zero rule"
        )
        self.lag, self.timed = 0.0, 0.0  # seconds the last native request took (any game's), and when

    def __call__(self, game, leaves, deadline=np.inf):
        if len(game.tokens) >= CONTEXT:
            return None
        games, zs = games_of(self, game)
        if (
            self.views != ("true",) and monotonic() > deadline
        ):  # a view's first sync prefills the whole game
            game.last_search = dict(searcher="kl", budget=leaves, reused=0, evaluated=0, nodes=0, late=True, full=False)
            return None
        z = game.sync()
        feats = [np.array(g.features(), np.float32) for g in games]
        roots = [x.double().numpy() for x in zs]
        prior = (
            None if self.views == ("true",) else z.double().numpy()
        )  # the game's own prior
        cap = 2 * leaves + 256
        srv = server(game)
        try:
            if srv is None:
                forest = self.handles(
                    game, games, zs, feats, roots, prior, cap, leaves, deadline
                )
            else:
                forest = self.native(
                    srv, game, games, feats, roots, prior, cap, leaves, deadline
                )
        except Late:
            return None
        if forest is None:
            return None
        if forest.full:
            log.warning("a kl search of %d leaves filled its %d nodes after %d", leaves, cap, forest.spent)
        moves, p, q = forest.root_q(**self.read)
        return [MOVES[t - MOVE_START] for t in moves], p, q

    def handles(self, game, games, zs, feats, roots, prior, cap, leaves, deadline):
        concurrent = game.engine.model.fast is not None
        trees = [Tree(g, x, capacity=cap) for g, x in zip(games, zs)]
        for t in trees:
            t.concurrent, t.deadline = concurrent, deadline

        def run():
            bridges = [
                t.handles([g.tokens], [f], self.clock_rule)
                for t, g, f in zip(trees, games, feats)
            ]
            forest = allie_fast.KL(
                list(game.tokens),
                roots,
                prior,
                feats[0],
                increment(games[0]),
                cap,
                self.clock_rule,
            )
            while (h := forest.select(leaves, **self.grow)) is not None:
                if len(h):
                    forest.update([np.asarray(b(h), float) for b in bridges])
            game.last_search = dict(
                searcher="kl",
                budget=leaves,
                reused=0,
                evaluated=forest.spent,
                nodes=forest.size,
                late=False,
                full=forest.full,
            )
            return forest

        return run() if concurrent else game.engine.run(run)

    def native(self, srv, game, games, feats, roots, prior, cap, leaves, deadline):
        key = ("kl", self.views, self.clock_rule)
        forest = reusable(game, games, key, self.reuse)
        if (
            forest is not None
            and forest.reroot(game.tokens[-2:], roots, prior, feats, cap) is None
        ):
            forest = None
        if forest is None:
            forest = allie_fast.KL(
                list(game.tokens),
                roots,
                prior,
                feats[0],
                increment(games[0]),
                cap,
                self.clock_rule,
            )
            for g, f in zip(games[1:], feats[1:]):
                forest.add_view(f, increment(g))
        done = forest.run(
            srv,
            [ref(g.cache) for g in games],
            leaves,
            deadline,
            self.merged,
            self.lag * 0.5 ** ((monotonic() - self.timed) / self.LAG),
            **self.grow,
        )
        if forest.spent > forest.reused:
            self.lag, self.timed = forest.lag, monotonic()
        game.last_search = dict(searcher="kl", budget=leaves, reused=forest.reused, evaluated=forest.spent - forest.reused,
                                nodes=forest.size, late=not done, full=forest.full)  # fmt: skip
        if done and self.reuse:
            keep(game, games, key, forest)
        return forest if done or forest.spent > forest.reused else None
