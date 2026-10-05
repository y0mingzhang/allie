"""allie.search's coverage search over a live game, on this package's model.

Search nodes extend the game's cache: each node adds one token whose attention reads the game's
keys and values plus its own path's. Clock bookkeeping is allie.search's (as MoEHandles in
allie.search.moe_oracle). The output calibration is refit for Allie 2.0: allie.search's own
calibration.json hurts this model at every budget.
"""

import json
import weakref
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from allie.search import Search, kl, lookahead
from allie.search.board import advance_clocks, predicted_seconds, root_other_previous
from allie.search.native import from_prefix, load

from .engine import Game
from .model import Cache
from .tokens import CONTEXT, HEADER, MOVE_START, MOVES, advance, header

CALIBRATION = Path(__file__).with_name("calibration-allie-2.0.json")
WDL = slice(2413, 2416)
# simulations -> the output policy fitted for them (8, 25 and 32 reuse 128's, 1024 256's)
POLICY = {5: "5", 8: "128", 25: "128", 32: "128", 128: "128", 256: "256", 1024: "256"}


class Late(Exception):
    """A search ran past its deadline."""


class Tree:
    """allie.search's oracle interface (reset, handles, new_tokens) for one root: the game's
    current position, whose keys and values are the game's cache."""

    def __init__(self, game, logits, capacity):
        self.game, self.model, self.cache = game, game.engine.model, game.cache
        self.root_logits = logits[None].double().numpy()
        m = self.model
        kw = dict(dtype=m.dtype, device=m.device)
        self.k = torch.empty(m.layers, m.heads, capacity, m.head_dim, **kw)
        self.v, self.e = torch.empty_like(self.k), torch.empty(capacity, m.width, **kw)
        self.k[:, :, 0] = self.v[:, :, 0] = 0  # slot 0 (the root's id) pads shorter paths
        self.capacity, self.new_tokens = capacity, 0
        self.concurrent = False  # nodes through the engine's queue (a search on the game's thread)
        self.deadline = np.inf  # time.monotonic() past which no further nodes run (Late)
        self.caches = []  # the fast backend's pool

    def reset(self):
        self.new_tokens = 0

    def pool(self, n):
        """n caches holding the game's prefix (the fast backend's leaves), copied once per tree."""
        caches = self.caches
        g, n0 = self.cache, self.cache.n
        while len(caches) < n:
            c = Cache(self.model, n0 + 8)
            c.k[:, :, :n0], c.v[:, :, :n0], c.e[:n0] = g.k[:, :, :n0], g.v[:, :, :n0], g.e[:n0]
            c.n = n0
            caches.append(c)
        return caches[:n]

    def handles(self, prefixes, feats, clock_rule="predicted"):
        assert len(prefixes) == 1 and list(prefixes[0]) == self.game.tokens
        return Nodes(self, np.asarray(feats[0], np.float32), clock_rule)


class Nodes:
    """MoEHandles' causal board and clock bookkeeping over Tree's per-node keys and values."""

    def __init__(self, tree, feats, clock_rule):
        self.tree, self.clock_rule = tree, clock_rule
        game = tree.game
        self.root_logits = tree.root_logits
        n, cap = len(game.tokens), tree.capacity
        inc = game.inc if game.inc is not None else -1
        self.parent = np.full(cap, -1, np.int64)
        self.length, self.token = np.zeros(cap, np.int64), np.zeros(cap, np.int64)
        self.board = [None] * cap
        self.feats = np.full((cap, 3), -1.0)
        self.other = np.full(cap, -1.0)
        self.elapsed = np.zeros(cap)
        self.length[0], self.board[0], self.feats[0] = n, game.boards[-1], feats[-1]
        self.other[0] = root_other_previous(game.tokens, feats, inc)
        self.inc = inc
        if clock_rule == "predicted":
            self.elapsed[0] = predicted_seconds(self.root_logits)[0]
        self.queries, self.per_root_queries = 0, np.zeros(1, np.int64)

    def __call__(self, handles):
        if monotonic() > self.tree.deadline:
            raise Late
        h = np.asarray(handles, np.int64)
        ids, parents, tokens, lengths = h.T
        assert (self.length[ids] == 0).all() and (
            lengths == self.length[parents] + 1
        ).all()
        assert lengths.max() <= CONTEXT
        self.parent[ids], self.token[ids] = parents, tokens
        self.length[ids] = lengths
        for i, p, t in zip(ids, parents, tokens):
            self.board[i] = advance(self.board[p], int(t))
        n = len(ids)
        f, o = advance_clocks(
            self.feats[parents], self.other[parents], self.length[parents],
            np.full(n, self.inc), self.elapsed[parents],
        )  # fmt: skip
        self.feats[ids], self.other[ids] = f, o
        z = self.forward(ids, tokens, lengths)
        if self.clock_rule == "predicted":
            self.elapsed[ids] = predicted_seconds(z)
        self.queries += n
        self.per_root_queries[0] += n
        self.tree.new_tokens += n
        return z

    def forward(self, ids, tokens, lengths):
        if self.tree.model.fast is not None:
            return self.fast(ids)
        t, m = self.tree, self.tree.model
        dev, dt = m.device, m.dtype
        cache, n0 = t.cache, t.cache.n
        assert ids.max() < t.capacity
        paths = []
        for i in ids:  # the node's ancestors below the root, then itself
            p, j = [], int(i)
            while j:
                p.append(j)
                j = int(self.parent[j])
            paths.append(p[::-1])
        width = max(map(len, paths))
        path = torch.tensor([p + [0] * (width - len(p)) for p in paths], device=dev)
        live = torch.tensor(
            [[k < len(p) for k in range(width)] for p in paths], device=dev
        )
        slots = torch.as_tensor(ids, device=dev)
        before = torch.as_tensor(self.parent[ids], device=dev)

        def previous(e):
            t.e[slots] = e
            p = t.e[before]
            p[before == 0] = cache.e[n0 - 1]
            return p

        def attend(i, q, k, v):
            t.k[i][:, slots], t.v[i][:, slots] = k.transpose(0, 1), v.transpose(0, 1)
            kp, vp = cache.k[i, :, :n0], cache.v[i, :, :n0]  # [H, L, D]
            kn, vn = t.k[i][:, path], t.v[i][:, path]  # [H, N, P, D]
            s = (
                torch.cat(
                    (
                        torch.einsum("nhd,hld->nhl", q, kp),
                        torch.einsum("nhd,hnpd->nhp", q, kn),
                    ),
                    -1,
                ).float()
                * m.scale
            )
            s[..., n0:] = s[..., n0:].masked_fill(~live[:, None], float("-inf"))
            a = F.softmax(s, -1).to(dt)
            return torch.einsum("nhl,hld->nhd", a[..., :n0], vp) + torch.einsum(
                "nhp,hnpd->nhd", a[..., n0:], vn
            )

        feats = torch.as_tensor(self.feats[ids], dtype=torch.float32, device=dev)
        boards = torch.tensor(
            np.frombuffer(b"".join(self.board[i] for i in ids), np.uint8).reshape(
                -1, 68
            ),
            device=dev,
        )
        pos = torch.as_tensor(lengths - 1, device=dev)
        tok = torch.as_tensor(tokens, device=dev)
        z = m.forward(tok, pos, feats, boards, previous, attend)
        return z.double().cpu().numpy()

    def path(self, i):
        """The node's ancestors below the root, then itself."""
        p, j = [], int(i)
        while j:
            p.append(j)
            j = int(self.parent[j])
        return p[::-1]

    def fast(self, ids, chunk=8):
        """The nodes' logits from the fast backend. Each node runs on a copy of the game's cache
        (pooled: the kernel writes only past the root, so the copied prefix stays) holding its
        ancestors' keys, values and embeddings at their own positions (kept in the tree's slots when
        they ran); only the node's own token is new. Up to `chunk` nodes run in one step: the pool holds
        that many copies (about 13 MB each at move 40, 1 024 tokens at most: 150 MB)."""
        t, m = self.tree, self.tree.model
        n0 = t.cache.n
        out = []
        for lo in range(0, len(ids), chunk):
            if monotonic() > t.deadline:
                raise Late
            batch, items = ids[lo : lo + chunk], []
            caches = t.pool(len(batch))
            for c, i in zip(caches, batch):
                p = self.path(i)
                c.truncate(n0)
                c.reserve(n0 + len(p))
                for d, j in enumerate(p[:-1]):
                    r = n0 + d
                    c.k[:, :, r], c.v[:, :, r], c.e[r] = t.k[:, :, j], t.v[:, :, j], t.e[j]
                c.n = n0 + len(p) - 1
                board = np.frombuffer(self.board[i], np.uint8).reshape(1, 68).copy()
                feats = torch.as_tensor(self.feats[[i]], dtype=torch.float32)
                items.append((c, torch.as_tensor(self.token[[i]]), feats, torch.as_tensor(board)))
            z = t.game.engine.steps(items) if t.concurrent else m.fast.step(items)
            out.append(z.double().numpy())
            for c, i in zip(caches, batch):
                r = c.n - 1
                t.k[:, :, i], t.v[:, :, i], t.e[i] = c.k[:, :, r], c.v[:, :, r], c.e[r]
        return np.concatenate(out)


class Coverage:
    """search(game, simulations, deadline) -> (legal moves, searched human-move probabilities, their
    prior, their searched values for the mover), or None if the search runs past deadline
    (time.monotonic()). views: the readings every node is evaluated under ("true": the game's own; else a
    View spec); its W/D/L is their mean (Mixed), the tree's other outputs the first view's, the root's
    prior and think time always the game's own."""

    def __init__(self, threads=4, views=("true",)):
        self.parameters = json.loads(CALIBRATION.read_text())
        bp = self.parameters["budget_policies"]
        for b, key in POLICY.items():
            bp.setdefault(str(b), bp[key])
        self.threads, self.views = threads, tuple(views)
        load(), load("value")  # built or loaded now, not in a game's first search

    def __call__(self, game, simulations, deadline=np.inf):
        own = game.__dict__.setdefault("views", {})
        games = [game if v == "true" else own.get(v) or own.setdefault(v, View(game, v)) for v in self.views]
        zs = [g.sync() for g in games]
        if self.views != ("true",) and monotonic() > deadline:  # a view's first sync prefills the whole game
            return None
        z = game.sync()
        feats = np.array(game.features(), np.float32)
        elo = game.elo[len(game.moves) % 2]
        row = dict(
            prefix=list(game.tokens),
            cell=game.cell(elo),
            legal=from_prefix(np.asarray(game.tokens)).legal(),
        )

        concurrent = game.engine.model.fast is not None

        def run():
            trees = [Tree(g, x, capacity=4 * simulations + 256) for g, x in zip(games, zs)]
            for t in trees:
                t.concurrent, t.deadline = concurrent, deadline
            oracle = Mixed(trees, z.double().numpy()) if self.views != ("true",) else trees[0]
            s = Search(oracle, threads=1 if concurrent else self.threads, calibration=self.parameters)
            return s._batch([row], [feats], "coverage", simulations, "predicted",
                            False, 0.9, 2.0, 1.25)[0]  # fmt: skip

        # on the fast backend, the search runs on this game's thread and its nodes join the
        # engine's batches with other games' moves and searches; otherwise it has the engine alone
        try:
            out = run() if concurrent else game.engine.run(run)
        except Late:
            return None
        moves = [MOVES[t - MOVE_START] for t in out["tokens"]]
        return moves, out["probabilities"], out["legal_prior"], out["values"]


class View:
    """The game read under another header or with no clocks: its own cache, the game's moves. spec: parts
    joined by '/': rE (both ratings E), swap (the ratings exchanged), tcB+I (base and increment seconds),
    noclock (every clock feature unknown)."""

    def __init__(self, game, spec):
        self.game, self.engine = weakref.proxy(game), game.engine  # the game owns its views
        head, self.clockless, self.inc = list(game.tokens[1:11]), False, game.inc
        for part in spec.split("/"):
            if part == "swap":
                head = head[:2] + head[6:] + head[2:6]
            elif part.startswith("tc") and part[2:].replace("+", "", 1).isdigit() and "+" in part:
                base, inc = map(int, part[2:].split("+"))
                head[:2] = header(base, inc, 0, 0)[1:3]
                self.inc = inc if inc <= 180 else None  # the clocks advance by the header's increment
            elif part == "noclock":
                self.clockless = True
            elif part.startswith("r") and part[1:].isdigit():
                head[2:] = [int(c) for c in f"{min(int(part[1:]), 9999):04d}" * 2]
            else:
                raise ValueError(f"unknown view part {part!r}")
        self.head = head
        self.cache, self.logits, self.used = Cache(game.engine.model), None, []

    tokens = property(lambda self: self.game.tokens[:1] + self.head + self.game.tokens[11:])
    boards = property(lambda self: self.game.boards)

    def features(self):
        return [[-1] * 3] * len(self.game.tokens) if self.clockless else self.game.features()

    def sync(self):
        n = len(self.game.tokens)
        if self.cache.n > n:  # the game was taken back past the view's cache
            self.cache.truncate(n)
            self.used, self.logits = self.used[:n], None
        return Game.sync(self)


def mixed(zs):
    """The first logits with the W/D/L rows set to the log of the mean W/D/L probabilities of all."""
    z = np.array(zs[0], np.float64)
    p = [np.exp(x - x.max(-1, keepdims=True)) for x in (np.asarray(y, np.float64)[..., WDL] for y in zs)]
    z[..., WDL] = np.log(np.mean([q / q.sum(-1, keepdims=True) for q in p], 0))
    return z


class Mixed:
    """allie.search's oracle over Trees of the game's views: every node is evaluated under each; its W/D/L
    is their mean, every other output the first view's; the root's policy and think time are `own`'s
    (the game's own logits)."""

    def __init__(self, trees, own):
        self.trees, self.own = trees, own

    new_tokens = property(lambda self: sum(t.new_tokens for t in self.trees))

    def reset(self):
        for t in self.trees:
            t.reset()

    def handles(self, prefixes, feats, clock_rule="predicted"):
        assert list(prefixes[0][HEADER:]) == list(self.trees[0].game.tokens[HEADER:])
        return MixedNodes([t.handles([t.game.tokens], [np.array(t.game.features(), np.float32)], clock_rule)
                           for t in self.trees], self.own, clock_rule)  # fmt: skip


class MixedNodes:
    """Every view's nodes; the root's policy and think time the game's own, and inside the tree every view's
    clocks follow the first view's think times."""

    def __init__(self, nodes, own, clock_rule):
        self.nodes, self.root_logits = nodes, mixed([n.root_logits for n in nodes])
        self.root_logits[0, : WDL.start] = own[: WDL.start]
        self.root_logits[0, WDL.stop :] = own[WDL.stop :]
        if clock_rule == "predicted":
            for n in nodes:
                n.elapsed[0] = predicted_seconds(self.root_logits)[0]

    queries = property(lambda self: self.nodes[0].queries)
    per_root_queries = property(lambda self: self.nodes[0].per_root_queries)

    def __call__(self, handles):
        ids = np.asarray(handles, np.int64)[:, 0]
        z = [self.nodes[0](handles)]
        for n in self.nodes[1:]:
            z.append(n(handles))
            n.elapsed[ids] = self.nodes[0].elapsed[ids]
        return mixed(z)


class Lookahead:
    """search(game, calls, deadline) -> (legal moves, their prior, their searched values for the
    mover), or None if the search runs past deadline (time.monotonic()): allie.search.lookahead on
    the game's cache, m root moves grown by k children a call after the first. On the fast backend
    it runs on the game's thread, as Coverage does."""

    def __init__(self, m=8, k=2, beta=4.0):
        self.m, self.k, self.beta = m, k, beta
        load()

    def __call__(self, game, calls, deadline=np.inf):
        z = game.sync()
        feats = np.array(game.features(), np.float32)
        concurrent = game.engine.model.fast is not None

        def run():
            tree = Tree(game, z, capacity=256 + calls * self.m * self.k)
            tree.concurrent, tree.deadline = concurrent, deadline
            bridge = tree.handles([game.tokens], [feats])
            return lookahead.search(bridge, game.tokens, calls, self.m, self.k, self.beta)

        try:
            moves, prior, q = run() if concurrent else game.engine.run(run)
        except Late:
            return None
        return [MOVES[t - MOVE_START] for t in moves], prior, q


class KL:
    """search(game, leaves, deadline) -> (legal moves, their prior, their searched values for the mover),
    or None if the search runs past deadline (time.monotonic()): allie.search.kl on the game's cache, grown
    best-first by reach to `leaves` network evaluations (kl.grow keywords `grow`: each side's policy tilted
    by its values) and read by kl.backup keywords `read`. With read's squash the values are log odds,
    arctanh(squash (W - L)), and so is Q; moves left unexpanded at the root take the root's own value. The
    tree's clocks do not advance (clock_rule "zero"). views: as Coverage's, the readings every node is
    evaluated under, its W/D/L their mean, the policy below the root the first view's, the root's prior the
    game's own; each view costs a network read a leaf. On the fast backend it runs on the game's thread, as
    Coverage does."""

    GROW = dict(own=5.0, opp=5.0, soft=True, kappa=0.5)
    READ = dict(own=12.0, opp=12.0, soft=True, squash=0.95)

    def __init__(self, grow=None, read=None, clock_rule="zero", views=("true",)):
        self.grow, self.read, self.clock_rule = grow or self.GROW, read or self.READ, clock_rule
        self.views = tuple(views)
        assert self.views == ("true",) or clock_rule == "zero", "views keep their own clocks only on the zero rule"
        load()

    def __call__(self, game, leaves, deadline=np.inf):
        own = game.__dict__.setdefault("views", {})
        games = [game if v == "true" else own.get(v) or own.setdefault(v, View(game, v)) for v in self.views]
        zs = [g.sync() for g in games]
        if self.views != ("true",) and monotonic() > deadline:  # a view's first sync prefills the whole game
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
            root = None if self.views == ("true",) else z[None].double().numpy()  # the game's own prior
            F = kl.Forest(bridges, [from_prefix(np.asarray(game.tokens))], [len(game.tokens)], trees[0].capacity, prior=root)
            kl.grow(F, leaves, **self.grow)
            moves, prior, q = kl.root_q(F, kl.backup(F.view(), **self.read)[0], 0)
            s = self.read.get("squash", 0.0)
            return moves, prior, np.where(np.isnan(q), np.arctanh(s * np.clip(F.value[0], -1, 1)) if s else F.value[0], q)

        try:
            moves, prior, q = run() if concurrent else game.engine.run(run)
        except Late:
            return None
        return [MOVES[t - MOVE_START] for t in moves], prior, q
