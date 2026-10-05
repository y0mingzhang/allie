import numpy as np
import pytest

from allie.search import kl
from allie.search.native import MOVE_ID, from_prefix

START = [2348, 198, 11, 2, 5, 0, 0, 2, 5, 0, 0]
PREFIXES = [
    START + [MOVE_ID[u] for u in g]
    for g in (("e2e4", "e7e5"), ("d2d4", "g8f6", "c2c4"), ("f2f3", "e7e5", "g2g4"))
]


class Bridge:
    """Stand-in model over several roots: logits drawn from a seed of the node's moves; checks the protocol."""

    def __init__(self, prefixes):
        self.paths, self.calls = {r: tuple(p) for r, p in enumerate(prefixes)}, 0
        self.root_logits = np.array(
            [self.logits(self.paths[r]) for r in range(len(prefixes))]
        )

    @staticmethod
    def logits(path):
        return np.random.default_rng(abs(hash(path)) % 2**32).normal(0, 2, 2432)

    def __call__(self, rows):
        self.calls += 1
        out = []
        for node, parent, token, length in rows:
            assert node not in self.paths and parent in self.paths
            assert length == len(self.paths[parent]) + 1
            self.paths[node] = self.paths[parent] + (int(token),)
            out.append(self.logits(self.paths[node]))
        return np.array(out)


def size(F, i, keep):
    return 1 + sum(size(F, c, keep) for c in range(F.size) if F.parent[c] == i and keep[c])


def reference(F, i, own, opp, soft, keep, kappa=0.0, squash=0.0):
    """Independent recursion of kl.backup."""
    v = np.arctanh(squash * F.value[i]) if squash else F.value[i]
    kids = [c for c in range(F.size) if F.parent[c] == i and keep[c]]
    if not kids:
        return v
    beta = (own if F.depth[i] % 2 == 0 else opp) * (1 + (size(F, i, keep) - 1) / 16) ** kappa
    left = 1 - sum(F.prior[c] for c in kids)
    p = np.array([F.prior[c] for c in kids] + [left if left > kl.TINY else 0])
    q = np.array([-reference(F, c, own, opp, soft, keep, kappa, squash) for c in kids] + [v])
    if soft and beta > 0:
        return np.log((p * np.exp(beta * q)).sum()) / beta
    w = p * np.exp(beta * (q - q[p > 0].max()))
    return (w * q).sum() / w.sum()


def forest(budget=60, **kw):
    b = Bridge(PREFIXES)
    F = kl.Forest(
        b,
        [from_prefix(np.array(p)) for p in PREFIXES],
        [len(p) for p in PREFIXES],
        4096,
    )
    kl.grow(F, budget, **kw)
    return b, F


@pytest.mark.parametrize(
    "tilt",
    [dict(), dict(own=4), dict(own=3, opp=1, soft=True), dict(own=8, opp=2, kappa=0.5),
     dict(own=5, opp=5, soft=True, kappa=0.5), dict(own=12, opp=12, soft=True, squash=0.95)],
)  # fmt: skip
def test_backup_matches_recursion(tilt):
    _, F = forest(**tilt)
    for budget in (5, 20, 60):
        keep = F.cost[: F.size] <= budget
        V, sigma, rest = kl.backup(F.view(), keep, **tilt)
        for r in range(F.n):
            assert V[r] == pytest.approx(reference(F, r, keep=keep, **(dict(own=0, opp=0, soft=False) | tilt)), abs=1e-12)
        par = F.parent[: F.size]
        for i in range(F.size):  # each kept node's tilted policy sums to one with its unexpanded moves
            kids = np.flatnonzero((par == i) & keep)
            if keep[i] and len(kids):
                left = 1 - F.prior[kids].sum()
                assert sigma[kids].sum() + rest[i] * left * (left > kl.TINY) == pytest.approx(1, abs=1e-12)


def test_rounding_is_not_a_move():
    """A fully expanded node whose priors sum to a hair under one (float32 dumps) is its children's value,
    however far its own static value sits above them."""
    d = dict(parent=np.array([-1, 0, 0]), depth=np.array([0, 1, 1]), prior=np.array([1.0, 0.6, 0.4 - 1e-7]),
             wdl=np.array([[0.995, 0, 0.005], [0.995, 0, 0.005], [0.995, 0, 0.005]]), terminal=np.zeros(3, bool))  # fmt: skip
    for tilt in (dict(own=12), dict(own=12, soft=True), dict(own=12, soft=True, squash=0.95)):
        V = kl.backup(d, **tilt)[0]
        assert V[0] == pytest.approx(-V[1], abs=1e-6)


def test_budget_protocol_and_prefix():
    b, F = forest(budget=60, own=4, k=4, g=0.25)
    assert (F.spent == 60).all()
    assert b.calls == F.calls and len(b.paths) - F.n == F.spent.sum()
    for i in range(
        F.n, F.size
    ):  # cost counts the evaluations so far; parents come first
        assert F.parent[i] < i and (F.cost[i] >= F.cost[i - 1] or F.owner[i] != F.owner[i - 1])
        assert (F.cost[i] >= F.cost[F.parent[i]]) or F.parent[i] < F.n
    for r in range(F.n):
        own = np.flatnonzero(F.owner[: F.size] == r)
        assert F.cost[own].max() == 60 and set(
            F.cost[own[~F.terminal[own]]][1:]
        ) == set(range(1, 61))
    for i in range(F.n, F.size):  # below the roots, moves are expanded by falling prior
        kids = F.ekid[F.start[i] : F.start[i] + F.count[i]]
        assert (kids[: F.done[i]] >= 0).all() and (kids[F.done[i] :] < 0).all()


def test_root_q():
    _, F = forest(budget=40)
    V, _, _ = kl.backup(F.view())
    for r in range(F.n):
        moves, prior, q = kl.root_q(F, V, r)
        assert sorted(moves.tolist()) == sorted(
            from_prefix(np.array(PREFIXES[r])).legal()
        )
        assert np.all(np.diff(prior) <= 0) and prior.sum() == pytest.approx(1)
        kid = F.ekid[F.start[r] : F.start[r] + F.count[r]]
        assert np.isnan(q[kid < 0]).all() and np.allclose(
            q[kid >= 0], -V[kid[kid >= 0]]
        )


class View(Bridge):
    """The stand-in model under another header: other logits for the same nodes."""

    @staticmethod
    def logits(path):
        return np.random.default_rng((abs(hash(path)) + 1) % 2**32).normal(0, 2, 2432)


def test_views():
    b, v = Bridge(PREFIXES), View(PREFIXES)
    F = kl.Forest([b, v], [from_prefix(np.array(p)) for p in PREFIXES], [len(p) for p in PREFIXES], 4096, policy=(0, 1))
    kl.grow(F, 30, own=4)
    d = F.dump()
    for i in range(F.size):
        if F.terminal[i]:
            continue
        path = b.paths[i]
        legal = np.array(F.pos[i].legal())
        z = (b if i < F.n or F.depth[i] % 2 == 0 else v).logits(path)
        e = slice(F.start[i], F.start[i] + F.count[i])
        assert np.allclose(F.ep[e], kl.softmax(z[F.etok[e]])) and set(F.etok[e]) == set(legal)
        w = np.mean([kl.softmax(x.logits(path)[kl.WDL]) for x in (b, v)], 0)
        assert F.value[i] == pytest.approx(w[0] - w[2])
    for r in range(F.n):
        e = slice(F.start[r], F.start[r] + F.count[r])
        for k, x in enumerate((b, v)):
            assert np.allclose(d["root_probs_v"][e, k], kl.softmax(x.logits(b.paths[r])[F.etok[e]]))


def test_prior():
    """A view under another header searches; the roots keep the game's own policy."""
    b, v = Bridge(PREFIXES), View(PREFIXES)
    F = kl.Forest(v, [from_prefix(np.array(p)) for p in PREFIXES], [len(p) for p in PREFIXES], 4096, prior=b.root_logits)
    kl.grow(F, 20, own=4)
    for r in range(F.n):
        e = slice(F.start[r], F.start[r] + F.count[r])
        assert np.allclose(F.ep[e], kl.softmax(b.root_logits[r][F.etok[e]]))
    assert np.allclose(F.heads, b.root_logits[:, 2350:2416])
    i = F.n + 1
    e = slice(F.start[i], F.start[i] + F.count[i])
    assert np.allclose(F.ep[e], kl.softmax(v.logits(v.paths[i])[F.etok[e]]))
