"""Lookahead search for playing strength in a few network calls.

The first call evaluates the position after every legal move; each later call takes the `m` root moves a
player is most likely to choose, tilted by their values (log p + beta Q), and grows each one's soft
principal variation by `k` children. Values are the model's win minus loss for the side to move, backed
up softly: V = T log(sum_c p_c exp(-V_c / T) + rest exp(v / T)), unexpanded moves at the node's own
value, T = tau / sqrt(1 + (subtree - 1) / 16) as in allie.search's coverage.

A bridge is an oracle's node evaluator (allie.search.moe_oracle.MoEHandles, allie.lichess.tree.Nodes): it
has root_logits and maps (node, parent, token, length) rows to their logits, nodes evaluated once each,
the root being node 0.
"""

import numpy as np

from .native import from_prefix

WDL = slice(2413, 2416)


def softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


class Tree:
    """One root's search tree, grown in network calls."""

    def __init__(self, bridge, prefix):
        self.bridge, self.calls = bridge, 0
        self.parent, self.prior, self.depth, self.value, self.terminal, self.length = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.pos, self.moves, self.probs, self.kids = [], [], [], []
        self._node(-1, 1.0, from_prefix(prefix), len(prefix))
        self._evaluate([0], np.asarray(bridge.root_logits, float)[:1])

    def _node(self, parent, prior, pos, length):
        out = pos.outcome()
        self.parent.append(parent)
        self.prior.append(prior)
        self.depth.append(0 if parent < 0 else self.depth[parent] + 1)
        self.value.append(0.0 if out < 0 or out == 0.5 else -1.0)
        self.terminal.append(out >= 0)
        self.length.append(length)
        self.pos.append(pos)
        self.moves.append(np.zeros(0, np.int64))
        self.probs.append(np.zeros(0))
        self.kids.append(np.zeros(0, np.int64))
        return len(self.parent) - 1

    def _evaluate(self, ids, z):
        for i, zi in zip(ids, z):
            legal = np.array(self.pos[i].legal(), np.int64)
            p = softmax(zi[legal])
            o = np.argsort(-p, kind="stable")
            self.moves[i], self.probs[i] = legal[o], p[o]
            self.kids[i] = np.full(len(legal), -1, np.int64)
            w = softmax(zi[WDL])
            self.value[i] = w[0] - w[2]

    def expand(self, requests):
        """(node, j) pairs, j indexing the node's moves by falling prior: one network call."""
        rows = []
        for i, j in requests:
            if self.kids[i][j] >= 0:
                continue
            t = int(self.moves[i][j])
            c = self._node(
                i, self.probs[i][j], self.pos[i].child(t), self.length[i] + 1
            )
            self.kids[i][j] = c
            if not self.terminal[c]:
                rows.append((c, i, t, self.length[c]))
        if rows:
            self.calls += 1
            self._evaluate(
                [r[0] for r in rows],
                np.asarray(self.bridge(np.array(rows, np.int64)), float),
            )

    def backup(self, tau=0.2):
        """Soft values, side to move at each node."""
        V = np.array(self.value)
        sub = np.ones(len(V))
        for i in range(len(V) - 1, 0, -1):
            sub[self.parent[i]] += sub[i]
        for i in range(len(V) - 1, -1, -1):
            k = self.kids[i]
            live = k >= 0
            if self.terminal[i] or not live.any():
                continue
            q, p = -V[k[live]], self.probs[i][live]
            rest = max(1 - p.sum(), 0.0)
            t = tau / np.sqrt(1 + (sub[i] - 1) / 16)
            hi = max(q.max(), self.value[i]) if rest > 1e-12 else q.max()
            V[i] = hi + t * np.log(
                (p * np.exp((q - hi) / t)).sum()
                + rest * np.exp((self.value[i] - hi) / t)
            )
        return V

    def frontier(self, V, i, k, tau):
        """The first node on node i's soft principal variation to grow, as (node, j) requests."""
        while not self.terminal[i]:
            kids = self.kids[i]
            live = kids >= 0
            if not live.any():
                return [(i, j) for j in range(min(k, len(kids)))]
            q, p = -V[kids[live]], self.probs[i][live]
            hi = max(q.max(), self.value[i])
            w = p * np.exp((q - hi) / tau)
            if max(1 - p.sum(), 0.0) * np.exp((self.value[i] - hi) / tau) > w.max():
                return [(i, int(j)) for j in np.flatnonzero(~live)[:k]]
            i = int(kids[live][np.argmax(w)])
        return []


def search(bridge, prefix, calls, m=4, k=4, beta=4.0, tau=0.2):
    """(legal move tokens by falling prior, their prior, their searched values Q for the mover)."""
    T = Tree(bridge, prefix)
    if len(T.moves[0]) > 1 and calls > 0:
        T.expand([(0, j) for j in range(len(T.moves[0]))])
        for _ in range(calls - 1):
            V = T.backup(tau)
            q = -V[T.kids[0]]
            top = np.argsort(
                -(np.log(np.maximum(T.probs[0], 1e-30)) + beta * q), kind="stable"
            )[:m]
            req = [r for a in T.kids[0][top] for r in T.frontier(V, int(a), k, tau)]
            if not req:
                break
            T.expand(req)
    V = T.backup(tau)
    q = np.where(T.kids[0] >= 0, -V[np.maximum(T.kids[0], 0)], T.value[0])
    return T.moves[0], T.probs[0], q


def tilt(prior, q, alpha=1.0, beta=0.0):
    """pi ~ prior^alpha exp(beta Q)."""
    return softmax(
        alpha * np.log(np.maximum(prior, 1e-300)) + beta * np.asarray(q, float)
    )
