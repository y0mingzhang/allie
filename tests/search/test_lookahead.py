import numpy as np
import pytest

from allie.search import lookahead
from allie.search.native import MOVE_ID, from_prefix

START = [2348, 198, 11, 2, 5, 0, 0, 2, 5, 0, 0]


class Bridge:
    """Stand-in model: logits drawn from a seed of the node's moves; checks the node protocol."""

    def __init__(self, prefix):
        self.paths, self.calls = {0: tuple(prefix)}, 0
        self.root_logits = self.logits(self.paths[0])[None]

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


def value(z):
    w = np.exp(z[2413:2416] - z[2413:2416].max())
    return (w[0] - w[2]) / w.sum()


def soft(T, i, tau):
    """Independent recursion of Tree.backup."""
    k = T.kids[i]
    if T.terminal[i] or not (k >= 0).any():
        return T.value[i], 1
    total, size = 0.0, 1
    vals = []
    for c, p in zip(k, T.probs[i]):
        if c >= 0:
            v, n = soft(T, c, tau)
            size += n
            vals.append((p, -v))
    rest = 1 - sum(p for p, _ in vals)
    t = tau / np.sqrt(1 + (size - 1) / 16)
    total = sum(p * np.exp(q / t) for p, q in vals) + max(rest, 0) * np.exp(
        T.value[i] / t
    )
    return t * np.log(total), size


def test_one_call_is_every_child_value():
    prefix = START + [MOVE_ID["e2e4"], MOVE_ID["e7e5"]]
    b = Bridge(prefix)
    moves, prior, q = lookahead.search(b, prefix, 1)
    assert b.calls == 1
    legal = from_prefix(prefix).legal()
    assert sorted(moves.tolist()) == sorted(legal) and abs(prior.sum() - 1) < 1e-12
    for t, v in zip(moves, q):
        assert v == pytest.approx(-value(b.logits(tuple(prefix) + (int(t),))))


@pytest.mark.parametrize("m,k,calls", [(4, 4, 5), (8, 2, 6), (1, 8, 3)])
def test_backup_and_calls(m, k, calls):
    prefix = START + [MOVE_ID[u] for u in ("d2d4", "g8f6", "c2c4", "e7e6")]
    b = Bridge(prefix)
    T = lookahead.Tree(b, prefix)
    T.expand([(0, j) for j in range(len(T.moves[0]))])
    for _ in range(calls - 1):
        V = T.backup()
        top = np.argsort(-(np.log(T.probs[0]) - 4 * V[T.kids[0]]), kind="stable")[:m]
        T.expand([r for a in T.kids[0][top] for r in T.frontier(V, int(a), k, 0.2)])
    assert b.calls == T.calls == calls
    assert len(T.parent) <= 1 + len(T.moves[0]) + (calls - 1) * m * k
    V = T.backup()
    for i in range(len(V)):
        assert V[i] == pytest.approx(soft(T, i, 0.2)[0], abs=1e-12)
    _, _, q = lookahead.search(Bridge(prefix), prefix, calls, m, k)
    np.testing.assert_allclose(q, -V[T.kids[0]], atol=1e-12)


def test_context_limit(monkeypatch):
    prefix = START + [MOVE_ID["e2e4"], MOVE_ID["e7e5"]]
    monkeypatch.setattr(lookahead, "CONTEXT", len(prefix) + 3)
    b = Bridge(prefix)
    lookahead.search(b, prefix, 8, 2, 2)
    assert max(map(len, b.paths.values())) <= lookahead.CONTEXT and b.calls == 3


def test_mate_in_one_scores_a_win():
    # 1. f3 e5 2. g4: Qh4 mates
    prefix = START + [MOVE_ID[u] for u in ("f2f3", "e7e5", "g2g4")]
    moves, _, q = lookahead.search(Bridge(prefix), prefix, 1)
    assert q[moves.tolist().index(MOVE_ID["d8h4"])] == 1.0


def test_tilt():
    p = np.array([0.5, 0.3, 0.2])
    assert np.allclose(lookahead.tilt(p, [0.0, 0.0, 0.0]), p)
    pi = lookahead.tilt(p, [0.0, 1.0, -1.0], 1.0, 2.0)
    assert np.isclose(pi.sum(), 1) and pi[1] > p[1] and pi[2] < p[2]
