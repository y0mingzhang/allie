"""Strength match and CE per leaf budget for gpu_kl.py runs (or any forest dump with a per-node cost), under
KL-regularized backups (allie.search.kl.backup): per cell, the CE of pi ~ prior exp(beta Q) at the beta that
matches the humans' accuracy, minus the raw policy's, and the Elo gaps at the CE-optimal beta (search_eff's
frontier.match, alpha 1).

python score_kl.py RUN [--budgets 8,16,...] [--backups own:opp[:soft],...]
"""

import argparse
import pickle
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

R = c.OUT / "pikl"
CELLS = [(3, 2400), (3, 2600), (2, 2600), (1, 2400)]


def parse(s):
    """'own:opp[:s][:kKAPPA][:xSQUASH]' -> kl.backup keywords (s: the soft objective; x: log-odds values)."""
    own, opp, *flags = s.split(":")
    kappa = [float(x[1:]) for x in flags if x.startswith("k")]
    squash = [float(x[1:]) for x in flags if x.startswith("x")]
    return dict(own=float(own), opp=float(opp), soft="s" in flags, kappa=kappa[0] if kappa else 0.0,
                squash=squash[0] if squash else 0.0)


def row(R_):
    return "  ".join(
        (
            f"{m['nll'] - m['own']:+.4f}"
            if abs(m["gap_acc"]) < 25
            else f"x{m['ceiling']:+5.0f}"
        )
        + f"/{m['ce_gap_acc']:+5.0f}"
        for m in (R_[k] for k in CELLS if k in R_)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--budgets", default="8,16,32,64,128,256,512,1024,2048,4096")
    p.add_argument("--backups", default="0:0,4:0,8:0,4:2,4:4,8:8,16:16,8:8:s,4:4:k0.5,8:8:s:k0.5,1000:0")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    D = c.Data()
    F = ev.load_forest(R / a.run)
    pos = np.searchsorted(D.index, F["index"])
    assert (D.index[pos] == F["index"]).all()
    S = ev.subset(D, pos)
    ev.annealed_prior(F, S)
    cost = F["cost"]
    res = {}
    print(
        f"{a.run}: CE - raw at matched accuracy (x: best reachable Elo gap) / Elo gap at the CE optimum"
    )
    print(
        "budget leaves backup        "
        + "  ".join(f"{c.FORMATS[f][:5]} {b}".ljust(13) for f, b in CELLS)
    )
    for b in map(int, a.budgets.split(",")):
        if b > cost.max():
            break
        keep = cost <= b
        _, leaves = ev.budget_stats(F, keep)
        for spec in a.backups.split(","):
            V, _, _ = kl.backup(F, keep, **parse(spec))
            q, v0 = ev.root_q(F, D, V, keep, pos)
            M = fr.match(S, q[S.flat], v0, CELLS, free=False)
            res[b, spec] = dict(
                leaves=leaves.mean(), match=M, q=q[S.flat].astype(np.float32)
            )
            print(f"{b:6d} {leaves.mean():6.0f} {spec:13s} {row(M)}", flush=True)
    pickle.dump(res, open(R / f"score-{a.run}{a.tag}.pkl", "wb"))


if __name__ == "__main__":
    main()
