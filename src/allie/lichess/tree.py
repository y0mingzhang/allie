"""allie.search's coverage search over a live game, on this package's model.

Search nodes extend the game's cache: each node adds one token whose attention reads the game's
keys and values plus its own path's. Clock bookkeeping is allie.search's (as MoEHandles in
allie.search.moe_oracle). The output calibration is refit for Allie 2.0: allie.search's own
calibration.json hurts this model at every budget.
"""

import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from allie.search import Search
from allie.search.board import advance_clocks, predicted_seconds, root_other_previous
from allie.search.native import from_prefix, load

from .model import Cache
from .tokens import CONTEXT, MOVE_START, MOVES, advance

CALIBRATION = Path(__file__).with_name("calibration-allie-2.0.json")
# simulations -> the output policy fitted for them (8, 25 and 32 reuse 128's)
POLICY = {5: "5", 8: "128", 25: "128", 32: "128", 128: "128", 256: "256"}


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
    """search(game, simulations, deadline) -> (legal moves, searched human-move probabilities), or
    None if the search runs past deadline (time.monotonic())."""

    def __init__(self, threads=4):
        self.parameters = json.loads(CALIBRATION.read_text())
        bp = self.parameters["budget_policies"]
        for b, key in POLICY.items():
            bp.setdefault(str(b), bp[key])
        self.threads = threads
        load(), load("value")  # built or loaded now, not in a game's first search

    def __call__(self, game, simulations, deadline=np.inf):
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
            tree = Tree(game, z, capacity=4 * simulations + 256)
            tree.concurrent, tree.deadline = concurrent, deadline
            s = Search(tree, threads=1 if concurrent else self.threads, calibration=self.parameters)
            return s._batch([row], [feats], "coverage", simulations, "predicted",
                            False, 0.9, 2.0, 1.25)[0]  # fmt: skip

        # on the fast backend, the search runs on this game's thread and its nodes join the
        # engine's batches with other games' moves and searches; otherwise it has the engine alone
        try:
            out = run() if concurrent else game.engine.run(run)
        except Late:
            return None
        return [MOVES[t - MOVE_START] for t in out["tokens"]], out["probabilities"]
