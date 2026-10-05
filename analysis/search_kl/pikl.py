"""harness.py plugin: KL-regularized search (allie.search.kl) on the harness forest's bridges and roots.

harness.py OUT --algo pikl:tree --budgets 8,...,4096 --kw '{"own": 4, "opp": 0}' (HARNESS_PATH = this directory)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/allie/search"))
import kl  # noqa: E402


def tree(
    F,
    budgets,
    own=4.0,
    opp=0.0,
    soft=False,
    kappa=0.0,
    backup=None,
    k=8,
    g=0.125,
    width=4,
    policy=(0, 0),
    root=0.0,
):
    """Grown once to max(budgets) leaves under the growth tilts; each budget is its nodes of cost <= budget,
    read by `backup` (kl.backup keywords, default the growth's)."""
    T = kl.Forest(F.bridges, F.pos[: F.n], F.length[: F.n], F.cap, policy=tuple(policy))
    kl.grow(T, max(budgets), own, opp, soft, kappa, k, g, width, root)
    view = T.view()
    cost = T.cost[: T.size]
    evaluated = ~T.terminal[: T.size] & (T.parent[: T.size] >= 0)
    out = {}
    for b in budgets:
        keep = cost <= b
        V, _, _ = kl.backup(
            view, keep, **(backup or dict(own=own, opp=opp, soft=soft, kappa=kappa))
        )
        q = []
        for r in range(F.n):
            moves, _, qr = kl.root_q(T, V, r)
            e = T.ekid[T.start[r] : T.start[r] + T.count[r]]
            qr[(e >= 0) & ~keep[np.maximum(e, 0)]] = np.nan
            at = {int(t): x for t, x in zip(moves, qr)}
            q.append(np.array([at[int(t)] for t in F.moves[r]]))
        out[b] = (
            q,
            len(F.bridges)
            * np.bincount(T.owner[: T.size][keep & evaluated], minlength=F.n),
        )
    return out
