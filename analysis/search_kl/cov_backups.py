"""KL-regularized backups (allie.search.kl.backup) on stored coverage forests (gpu.py cov runs): the same trees,
coverage's soft backup against expectimax-style backups, per budget (nodes of pull tag <= budget) on the scaling
set's cells that the run covers.

python cov_backups.py RUN_DIR [--budgets 8,...,1024] [--backups own:opp[:soft],...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    "/home/yimingz3/src/allie-wt-seff/analysis/search_eff",
    str(Path(__file__).resolve().parents[2] / "src/allie/search"),
]
import common as c  # noqa: E402
import evaluate as ev  # noqa: E402
import frontier as fr  # noqa: E402
import kl  # noqa: E402
from score_kl import CELLS, parse, row  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--budgets", default="8,32,128,256,512,1024")
    p.add_argument(
        "--backups", default="cov,0:0,2:0,4:0,8:0,1000:0,4:2,4:0:soft,1000:1000"
    )
    a = p.parse_args()
    D = c.Data()
    F = ev.load_forest(a.run)
    sel = np.isin(F["index"], np.load(c.OUT / "subset-scale.npy"))
    pos = np.searchsorted(D.index, F["index"])
    S = ev.subset(D, pos[sel])
    roots = np.zeros(len(F["parent"]), bool)
    roots[F["roots"][sel]] = True
    mine = roots[F["roots"][F["owner"]]]
    prior = np.zeros(len(D.prior))
    for r in np.flatnonzero(sel):  # the forest's own root policy
        a0, b0 = F["root_off"][r], F["root_off"][r + 1]
        slot = {
            int(t): D.offsets[pos[r]] + j
            for j, t in enumerate(D.legal[D.offsets[pos[r]] : D.offsets[pos[r] + 1]])
        }
        prior[[slot[int(t)] for t in F["root_moves"][a0:b0]]] = F["root_probs"][a0:b0]
    S.prior = prior[S.flat]
    print(
        f"{a.run}: {sel.sum()} roots; CE - raw at matched accuracy (x: best reachable Elo gap) / Elo gap at CE optimum"
    )
    for b in map(int, a.budgets.split(",")):
        keep = mine & (F["tag"] <= b)
        _, leaves = ev.budget_stats(F, keep)
        for spec in a.backups.split(","):
            V = (
                ev.soft_backup(F, keep)
                if spec == "cov"
                else kl.backup(F, keep, **parse(spec))[0]
            )
            q, v0 = ev.root_q(F, D, V, keep, pos)
            M = fr.match(S, q[S.flat], v0[sel], CELLS, free=False)
            print(f"{b:6d} {leaves[sel].mean():6.0f} {spec:13s} {row(M)}", flush=True)


if __name__ == "__main__":
    main()
