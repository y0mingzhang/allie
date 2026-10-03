import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.special import logsumexp
from allie.search import Search
from allie.search.native import load, from_prefix, MOVES, MOVE_ID

FIXTURE = Path(__file__).with_name("migration.json")


def logits(prefix):
    seed = int(hashlib.sha256(np.array(prefix, np.int16).tobytes()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=2432).astype(np.float32)


class FakeOracle:
    def reset(self):
        self.new_tokens = 0

    def handles(self, prefixes, features, clock_rule):
        self.new_tokens = sum(map(len, prefixes))
        return FakeHandles(prefixes)


class FakeHandles:
    def __init__(self, prefixes):
        self.known = {i: p for i, p in enumerate(prefixes)}
        self.owner = {i: i for i in range(len(prefixes))}
        self.root_logits = np.array([logits(p) for p in prefixes])
        self.queries = 0
        self.per_root_queries = np.zeros(len(prefixes), int)

    def __call__(self, handles):
        result = []
        for node, parent, token, length in handles:
            assert int(node) not in self.known
            p = self.known[int(parent)] + [int(token)]
            assert len(p) == length
            self.known[int(node)] = p
            self.owner[int(node)] = self.owner[int(parent)]
            self.per_root_queries[self.owner[int(node)]] += 1
            self.queries += 1
            result.append(logits(p))
        return np.array(result)


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_precleanup_predictions(self):
        queries = self.fixture["positions"]
        engine = Search(FakeOracle(), threads=1)
        for key, expected in self.fixture["expected"].items():
            with self.subTest(method=key):
                if key.startswith("allie"):
                    got = engine.predict(queries, method="allie", budget=int(key[5:]))
                else:
                    got = engine.predict(queries, budget=key if key == "adaptive" else int(key))
                for actual, reference in zip(got, expected):
                    for name in ("probabilities", "values"):
                        np.testing.assert_array_equal(actual[name], reference[name])
                    if "nodes" in reference:
                        self.assertEqual(actual["nodes"], reference["nodes"])
                        self.assertEqual(actual["simulations"], reference["simulations"])
                    self.assertTrue((actual["probabilities"] > 0).all())
                    self.assertAlmostEqual(actual["probabilities"].sum(), 1., places=13)

    def test_chunking_and_label_independence(self):
        queries = self.fixture["positions"]
        a = Search(FakeOracle(), threads=1).predict(queries)
        altered = [q | {"target": -123, "outcome": "wrong", "future_clock": -999} for q in queries]
        same_batch = Search(FakeOracle(), threads=1).predict(altered)
        for x, y in zip(a, same_batch):
            np.testing.assert_array_equal(x["probabilities"], y["probabilities"])
        b = Search(FakeOracle(), batch_size=1, threads=1).predict(altered)
        for x, y in zip(a, b):
            # Different NumPy reduction widths can round the calibration at FP64 epsilon.
            np.testing.assert_allclose(x["probabilities"], y["probabilities"], atol=2e-16, rtol=0)
            np.testing.assert_array_equal(x["values"], y["values"])
            self.assertEqual(x["nodes"], y["nodes"])

    def test_legal_and_invalid_inputs(self):
        engine = Search(FakeOracle())
        self.assertEqual(engine.predict([]), [])
        q = self.fixture["positions"][0]
        r = engine.predict([q], method="legal")[0]
        self.assertEqual(r["nodes"], 0)
        np.testing.assert_array_equal(r["probabilities"], r["legal_prior"])
        for bad in (q | {"cell": 16}, q | {"prefix": [1.] * 12}, q | {"features": [[0, 0, 0]]}):
            with self.assertRaises(ValueError):
                engine.predict([bad])
        corrupt = q["prefix"].copy()
        corrupt[1] = MOVE_ID["e2e4"]
        with self.assertRaises(ValueError):
            engine.predict([q | {"prefix": corrupt}])
        with self.assertRaises(ValueError):
            engine.predict([q], budget=99)
        with self.assertRaises(ValueError):
            engine.predict([q], method="allie", budget=50, allie_cpuct=float("nan"))

    def test_rules_against_python_chess(self):
        import chess
        rng = np.random.default_rng(402)
        for game in range(8):
            a, b = load().Position(), chess.Board()
            for ply in range(80):
                expected = [MOVE_ID[m.uci()] for m in b.legal_moves]
                self.assertEqual(a.legal(), expected)
                outcome = b.outcome(claim_draw=False)
                value = -1 if outcome is None else (.5 if outcome.winner is None else float(outcome.winner))
                self.assertEqual(a.outcome(), value)
                if outcome:
                    break
                token = expected[int(rng.integers(len(expected)))]
                a.push(token)
                b.push_uci(MOVES[token - 378])
                self.assertEqual(a.fen().split()[:3], b.fen().split()[:3])
        for fen in ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"):
            a, b = load().Position(fen), chess.Board(fen)
            self.assertEqual(a.outcome(), .5 if b.outcome().winner is None else float(b.outcome().winner))

    def test_value_recursion(self):
        rng = np.random.default_rng(281)
        n = 60
        parent = np.array([-1] + [int(rng.integers(i)) for i in range(1, n)], np.int32)
        prior = np.zeros(n)
        degree = np.array([sum(parent == i) + 1 for i in range(n)], np.int32)
        for i in range(n):
            child = np.flatnonzero(parent == i)
            if len(child):
                prior[child] = rng.dirichlet(np.ones(len(child)+1))[:-1]
        d = dict(parent=parent, move=np.arange(n, dtype=np.int32), born=np.arange(n, dtype=np.int32),
                 roots=np.array([0], np.int32), degree=degree, prior=prior, boot=rng.uniform(-1, 1, n),
                 mass=np.ones(n), terminal=np.full(n, -1.))
        d["terminal"][-2:] = [.5, 1.]
        ids = np.arange(n, dtype=np.int32)[None]
        for budget in (0, 17, 60):
            for scale in (4., 16., 64.):
                def recur(i):
                    if d["terminal"][i] >= 0:
                        return (0. if d["terminal"][i] == .5 else -1.), 1
                    child = [j for j in range(n) if parent[j] == i and j <= budget]
                    if not child:
                        return -d["boot"][i], 1
                    vv = [recur(j) for j in child]
                    count = 1 + sum(v[1] for v in vv)
                    tau = .2 * (1 + (count-1)/scale)**-.5
                    p = np.r_[prior[child], max(0, 1-prior[child].sum())]
                    values = np.array([-v[0] for v in vv] + [-d["boot"][i]])
                    return tau * logsumexp(values/tau, b=p), count
                want = [-recur(j)[0] if j > 0 and parent[j] == 0 and j <= budget else -d["boot"][0] for j in range(n)]
                got = load("value").Backup(d, budget, scale).reduce(np.log(.2), -.5, ids)[0, 0]
                np.testing.assert_allclose(got, want, atol=2e-14, rtol=0)
            project = load("value").Projection(d, budget)
            project.project(0.)
            np.testing.assert_array_equal(project.reduce(np.log(.2), -.5, ids),
                load("value").Backup(d, budget, 16.).reduce(np.log(.2), -.5, ids))


if __name__ == "__main__":
    unittest.main()
