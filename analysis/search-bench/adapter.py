"""Maia-3 benchmark and dev positions as search queries, and Bench: Search at any fixed budget, where 5/8/25
use the 128-simulation output policy and 460 the 512 one unless the calibration has its own.
"""

import numpy as np

from allie import paths
from allie.search import Search
from allie.search.native import from_prefix

G = paths.DATA / "strat-eval-v1"
DATA = paths.DATA / "maia3-bench"
POLICY = {5: "128", 8: "128", 25: "128", 128: "128", 460: "512"}


def positions():
    """Benchmark positions: prefix tokens, causal features, cell, target token, game."""
    with np.load(G / "strat.npz") as z:
        rows = z["rows"].astype(np.int64)
    with np.load(G / "feats.npz") as z:
        feats = z["feats"]
    with np.load(DATA / "legal.npz") as z:
        pos, target = z["pos"], z["target"]
    with np.load(DATA / "games.npz") as z:
        s = z["sel"][z["keep"]]
    out = []
    for (r, c), t, (g, ply, cell, *_) in zip(pos, target, s):
        bos = np.flatnonzero(rows[r, : c + 1] == 2348)[-1]
        assert c - bos == 10 + ply
        out.append(
            dict(
                prefix=rows[r, bos : c + 1].tolist(),
                features=feats[r, bos : c + 1].astype(np.float32),
                cell=int(cell),
                target=int(t) + 378,
                game=int(g),
            )
        )
    return out


def dev_positions():
    """build_dev.py's calibration positions, in the same form as positions()."""
    z = dict(np.load(DATA / "search" / "dev-positions.npz"))
    o, tokens, feats = z["offsets"], z["tokens"], z["feats"]
    return [dict(prefix=tokens[a:b].tolist(), features=feats[a:b].copy(), cell=int(c), target=int(t), game=int(g))
            for a, b, c, t, g in zip(o[:-1], o[1:], z["cell"], z["target"], z["game"])]


class Bench(Search):
    def __init__(self, oracle, **kw):
        super().__init__(oracle, **kw)
        bp = self.parameters["budget_policies"]
        for b, key in POLICY.items():
            bp.setdefault(str(b), bp[key])

    def predict(self, queries, budget, clock_rule="predicted"):
        if budget in ("legal", "adaptive") or budget in (64, 1000):
            method = "legal" if budget == "legal" else "coverage"
            return super().predict(
                queries, method=method, budget=budget, clock_rule=clock_rule
            )
        assert budget in POLICY, budget
        rows, feats = [], []
        for q in queries:
            board = from_prefix(np.asarray(q["prefix"]))
            assert board.outcome() < 0
            rows.append(
                dict(prefix=list(q["prefix"]), cell=q["cell"], legal=board.legal())
            )
            feats.append(np.asarray(q["features"], np.float32))
        out = []
        for lo in range(0, len(rows), self.batch_size):
            out += self._batch(
                rows[lo : lo + self.batch_size],
                feats[lo : lo + self.batch_size],
                "coverage",
                budget,
                clock_rule,
                False,
                0.9,
                2.0,
                1.25,
            )
        return out
