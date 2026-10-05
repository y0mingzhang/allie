"""KL-regularized search against the human reply model (piKL: Jacob et al. 2022, "Modeling strong and
human-like gameplay with KL-regularized search").

Each side plays its human policy p tilted by its values, pi ~ p exp(beta Q), the maximizer of
E_pi Q - KL(pi || p) / beta: the root's mover with beta `own` at even depths, the opponent with `opp`
at odd depths (0: the human policy itself, expectimax). A node's value, win minus loss for its side to
move, is the expectation of its children's under that policy, the moves not yet expanded at the node's
own value (soft=True: the regularized objective log(sum p exp(beta Q)) / beta instead). Trees grow
best-first in network calls: each call evaluates every root's unexpanded moves of largest reach, the
product of the policies along their path, the root's own moves weighted by sqrt(p (1 - p)) (how
allie.search's coverage spreads its simulations).

A bridge is an oracle's node evaluator (allie.search.moe_oracle.MoEHandles, allie.lichess.tree.Nodes):
root_logits for roots 0..n-1 and a call mapping (node, parent, token, length) rows to their logits, each
node evaluated once. Positions are allie.search.native boards (legal(), child(token), outcome()).
"""

import numpy as np

WDL = slice(2413, 2416)
CONTEXT = 1025  # the model's longest game, header included
TINY = 1e-5  # unexpanded prior mass below this is rounding (float32 dumps leave ~1e-7): the node is fully expanded


def softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


class Forest:
    """Search trees over a batch of roots (node r < n is root r), grown in network calls. Node i's legal
    moves sit at edges start[i]..start[i] + count[i] by falling prior; below the roots the first done[i] are expanded.
    cost[i]: network evaluations of its root's tree up to node i, so cost <= b is the tree of budget b (up to a
    terminal node or two: a budget-b grow re-ranks once more after a last call that reached terminals)."""

    def __init__(self, bridge, positions, lengths, cap, values=None, policy=(0, 0), prior=None):
        """bridge: one bridge, or views of the same roots that each evaluate every node (e.g. other header
        ratings): the W/D/L is the mean of the views in `values` (default all), the policy the root's first
        view's and below it view policy[0] at the mover's nodes, policy[1] at the opponent's. prior: the
        roots' logits under the game's own header, its policy and heads the roots' when no view reads it."""
        n = len(positions)
        self.prior_logits = None if prior is None else np.asarray(prior, float)
        self.bridges = bridge if isinstance(bridge, list) else [bridge]
        self.values = list(range(len(self.bridges))) if values is None else values
        self.policy = policy
        self.n, self.size, self.cap, self.calls = n, n, cap, 0
        i64 = lambda fill: np.full(cap, fill, np.int64)  # noqa: E731
        self.parent, self.owner, self.token, self.depth = (
            i64(-1),
            i64(-1),
            i64(0),
            i64(0),
        )
        self.length, self.cost, self.born, self.edge = i64(0), i64(0), i64(0), i64(-1)
        self.start, self.count, self.done = i64(0), i64(0), i64(0)
        self.owner[:n], self.length[:n] = np.arange(n), lengths
        self.prior, self.value = np.ones(cap), np.zeros(cap)
        self.wdl, self.wdlv = np.zeros((cap, 3)), np.zeros((cap, len(self.bridges), 3))
        self.terminal = np.zeros(cap, bool)
        self.pos = list(positions) + [None] * (cap - n)
        self.spent = np.zeros(n, np.int64)
        self.etok, self.ep, self.ekid = (
            np.zeros(0, np.int64),
            np.zeros(0),
            np.zeros(0, np.int64),
        )
        self.edges = 0
        z = [np.asarray(b.root_logits, float) for b in self.bridges]
        self.heads = (z[0] if prior is None else self.prior_logits)[:, 2350:2416]
        self._set(np.arange(n), z)
        self.rootw = self.rootw0 = np.concatenate([root_weights(self.ep[self.start[r] : self.start[r] + self.count[r]]) for r in range(n)])
        root = [self.etok[self.start[r] : self.start[r] + self.count[r]] for r in range(n)]
        self.rootpv = np.stack([np.concatenate([softmax(x[r][m]) for r, m in enumerate(root)]) for x in z], 1)

    def _set(self, ids, z):
        legal = [np.array(self.pos[i].legal(), np.int64) for i in ids]
        need = self.edges + sum(map(len, legal))
        if need > len(self.ep):
            grow = max(need, 2 * len(self.ep), 1024)
            self.etok, self.ep = np.resize(self.etok, grow), np.resize(self.ep, grow)
            self.ekid = np.resize(self.ekid, grow)
        for k, (i, m) in enumerate(zip(ids, legal)):
            pz = self.prior_logits if i < self.n and self.prior_logits is not None else z[0 if i < self.n else self.policy[self.depth[i] % 2]]
            p = softmax(pz[k][m])
            o = np.argsort(-p, kind="stable")
            e = slice(self.edges, self.edges + len(m))
            self.etok[e], self.ep[e], self.ekid[e] = m[o], p[o], -1
            self.start[i], self.count[i] = self.edges, len(m)
            self.edges += len(m)
            self.wdlv[i] = [softmax(x[k][WDL]) for x in z]
            w = self.wdlv[i, self.values].mean(0)
            self.wdl[i], self.value[i] = w, w[0] - w[2]

    def expand(self, requests):
        """(node, j) pairs, j an unexpanded move of the node: one network call for the new nonterminal
        children, in request order (a request's cost counts the evaluations before it)."""
        rows, new = [], []
        for i, j in requests:
            c, e = self.size, self.start[i] + j
            assert j < self.count[i] and self.ekid[e] < 0 and c < self.cap
            self.size += 1
            self.done[i] += 1
            t, r = int(self.etok[e]), self.owner[i]
            self.ekid[e], self.edge[c] = c, e
            self.parent[c], self.owner[c], self.token[c] = i, r, t
            self.prior[c], self.depth[c], self.length[c] = (
                self.ep[e],
                self.depth[i] + 1,
                self.length[i] + 1,
            )
            self.born[c] = self.calls + 1
            self.pos[c] = self.pos[i].child(t)
            out = self.pos[c].outcome()
            if out >= 0:
                self.terminal[c] = True
                self.value[c] = 0.0 if out == 0.5 else -1.0
                self.wdl[c] = self.wdlv[c] = (0, 1, 0) if out == 0.5 else (0, 0, 1)
            else:
                self.spent[r] += 1
                rows.append((c, i, t, self.length[c]))
                new.append(c)
            self.cost[c] = self.spent[r]
        if rows:
            self.calls += 1
            h = np.array(rows, np.int64)
            self._set(np.array(new), [np.asarray(b(h), float) for b in self.bridges])

    def view(self):
        """The arrays backup() reads, over the nodes so far."""
        s = slice(0, self.size)
        return dict(parent=self.parent[s], depth=self.depth[s], prior=self.prior[s], wdl=self.wdl[s],
                    terminal=self.terminal[s], owner=self.owner[s])  # fmt: skip

    def dump(self):
        s = slice(0, self.size)
        roots = range(self.n)
        return dict(parent=self.parent[s], owner=self.owner[s], token=self.token[s].astype(np.int16),
                    prior=self.prior[s].astype(np.float32), depth=self.depth[s].astype(np.int16),
                    born=self.born[s].astype(np.int16), cost=self.cost[s].astype(np.int32),
                    wdl=self.wdl[s].astype(np.float32), terminal=self.terminal[s], calls=np.full(self.n, self.calls),
                    leaves=self.spent.copy(), heads=self.heads.astype(np.float32),
                    root_moves=np.concatenate([self.etok[self.start[r]:self.start[r] + self.count[r]] for r in roots]).astype(np.int16),
                    root_probs=np.concatenate([self.ep[self.start[r]:self.start[r] + self.count[r]] for r in roots]).astype(np.float32),
                    root_len=self.count[:self.n].copy()) | ({} if len(self.bridges) == 1 else dict(wdlv=self.wdlv[s].astype(np.float32))) | ({} if len(self.bridges) == 1 and self.prior_logits is None else dict(root_probs_v=self.rootpv.astype(np.float32)))  # fmt: skip


def backup(F, keep=None, own=0.0, opp=0.0, soft=False, kappa=0.0, squash=0.0):
    """Values (side to move) of the nodes in `keep` (default all) of a forest's view() or dump: each
    node's policy p tilted by beta (own at even depths, opp at odd), times (1 + (subtree - 1) / 16)^kappa
    (kappa 0.5: KL weight falling as 1 / sqrt(subtree), as allie.search's soft backup sharpens), its
    unexpanded moves at its own value. 0 < squash < 1: values on the log-odds scale, arctanh(squash (W - L)).
    Returns (V, sigma, rest): each child's probability under its parent's tilted policy and each
    node's tilted probability per unit prior of an unexpanded move."""
    par, d = F["parent"], np.asarray(F["depth"], np.int64)
    w = np.asarray(F["wdl"], float)
    v, prior = w[:, 0] - w[:, 2], np.asarray(F["prior"], float)
    if squash:
        v = np.arctanh(squash * np.clip(v, -1, 1))
    n = len(par)
    keep = np.ones(n, bool) if keep is None else keep
    levels = [np.flatnonzero(keep & (d == k)) for k in range(d.max() + 1)]
    seen, sub = np.zeros(n), np.ones(n)
    for c in levels[:0:-1]:
        np.add.at(seen, par[c], prior[c])
        np.add.at(sub, par[c], sub[c])
    beta = np.where(d % 2 == 0, own, opp) * (1 + (sub - 1) / 16) ** kappa
    left = np.where(seen < 1 - TINY, 1 - seen, 0.0)
    V, hi, Z = v.copy(), v.copy(), np.ones(n)
    for c in levels[:0:-1]:
        p, q = par[c], -V[c]
        hi[p] = np.where(left[p] > 0, v[p], -np.inf)
        np.maximum.at(hi, p, q)
        e = prior[c] * np.exp(beta[p] * (q - hi[p]))
        u = np.unique(p)
        Z[u] = left[u] * np.exp(beta[u] * np.minimum(v[u] - hi[u], 0))
        num = Z * v
        np.add.at(Z, p, e)
        np.add.at(num, p, e * q)
        if soft:
            b = np.where(beta[u] > 0, beta[u], 1.0)
            V[u] = np.where(beta[u] > 0, hi[u] + np.log(Z[u]) / b, num[u] / Z[u])
        else:
            V[u] = num[u] / Z[u]
    sigma = np.zeros(n)
    for c in levels[1:]:
        p = par[c]
        sigma[c] = prior[c] * np.exp(beta[p] * (-V[c] - hi[p])) / Z[p]
    return V, sigma, np.exp(beta * np.minimum(v - hi, 0)) / Z


def root_weights(p):
    w = np.sqrt(p * (1 - p))
    return w / max(w.sum(), 1e-300)


def grow(F, budget, own=0.0, opp=0.0, soft=False, kappa=0.0, k=8, g=0.125, width=4, root=0.0, full=False, floor=0.0, squash=0.0):
    """Best-first by reach until each root has `budget` network evaluations (or nothing to expand): each
    call takes per root its max(k, g * evaluations so far) unexpanded moves of largest reach, below the
    root at most `width` of them from one node. root > 0: the root's move weights are the mean of
    sqrt(p (1 - p)) and sqrt(pi (1 - pi)), pi ~ p exp(root Q) the output tilt (unexpanded moves at the
    root's value), each normalized: the search follows where the tilted output is uncertain. full: every
    root move is evaluated before any deeper node. floor: below the root, reach follows the tilted policy
    mixed with the share `floor` of the human policy itself (breadth where the tilt is sure). squash: as
    backup's."""
    n, E = F.n, len(F.rootw)
    top = np.repeat(np.arange(n), F.count[:n])
    cov = F.rootw0
    while True:
        need = np.where(F.count[:n] > 1, budget - F.spent, 0)
        size = F.size
        V, sigma, rest = backup(F.view(), own=own, opp=opp, soft=soft, kappa=kappa, squash=squash)
        sigma, rest = (1 - floor) * sigma + floor * F.prior[:size], (1 - floor) * rest + floor
        if root:
            kid = F.ekid[:E]
            v0 = np.arctanh(squash * np.clip(F.value[top], -1, 1)) if squash else F.value[top]
            z = np.log(np.maximum(F.ep[:E], 1e-300)) + root * np.where(kid >= 0, -V[np.maximum(kid, 0)], v0)
            z = np.exp(z - np.maximum.reduceat(z, F.start[:n])[top])
            pi = z / np.add.reduceat(z, F.start[:n])[top]
            w = np.sqrt(pi * (1 - pi))
            F.rootw = 0.5 * cov + 0.5 * w / np.maximum(np.add.reduceat(w, F.start[:n])[top], 1e-300)
        d = F.depth[:size]
        reach = np.ones(size)
        for k_ in range(1, d.max() + 1):
            c = np.flatnonzero(d == k_)
            reach[c] = F.rootw[F.edge[c]] if k_ == 1 else reach[F.parent[c]] * sigma[c]
        i = n + np.flatnonzero(~F.terminal[n:size] & (F.done[n:size] < F.count[n:size])
                               & (F.length[n:size] < CONTEXT) & (need[F.owner[n:size]] > 0))  # fmt: skip
        reps = np.minimum(F.count[i] - F.done[i], width)
        i = np.repeat(i, reps)
        j = F.done[i] + np.arange(len(i)) - np.repeat(np.cumsum(reps) - reps, reps)
        e = F.start[i] + j
        score = reach[i] * rest[i] * F.ep[e]
        free = np.flatnonzero((F.ekid[:E] < 0) & (need[top] > 0) & (F.length[top] < CONTEXT))
        i, j = np.r_[top[free], i], np.r_[free - F.start[top[free]], j]
        score = np.r_[F.rootw[free] + full, score]
        r = F.owner[i]
        o = np.lexsort((j, -score, r))
        r, i, j = r[o], i[o], j[o]
        quota = np.minimum(np.maximum(k, np.ceil(g * F.spent)).astype(np.int64), need)
        take = np.arange(len(r)) - np.searchsorted(r, r) < quota[r]
        take &= np.cumsum(take) <= F.cap - F.size  # a full forest stops growing
        if not take.any():
            return
        F.expand(list(zip(i[take].tolist(), j[take].tolist())))


def root_q(F, V, r):
    """(moves by falling prior, prior, Q for the mover; nan where unexpanded) of root r."""
    e = slice(F.start[r], F.start[r] + F.count[r])
    kid = F.ekid[e]
    return (
        F.etok[e].copy(),
        F.ep[e].copy(),
        np.where(kid >= 0, -V[np.maximum(kid, 0)], np.nan),
    )
