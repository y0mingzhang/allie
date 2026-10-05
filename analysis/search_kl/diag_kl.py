"""Why a run's tilt cannot reach the humans' accuracy: per dev cell and budget, the share of root moves and of prior
mass searched, and the accuracy of the argmax-Q move (the tilt's ceiling) with unsearched moves at the root's value
or at the searched moves' prior-weighted mean, against the humans'.

python diag_kl.py HARNESS_RUN [HARNESS_RUN ...] [--budgets 64,256,1024,4096]
"""

import argparse
import sys

import numpy as np

sys.path[:0] = ["/home/yimingz3/src/allie-wt-seff/analysis/search_eff"]
import common as c  # noqa: E402
import evaluate as ev  # noqa: E402
from scale_score import CELLS, load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--budgets", default="64,256,1024,4096")
    a = p.parse_args()
    D = c.Data()
    for run in a.runs:
        z, budgets = load(run)
        pos = np.searchsorted(D.index, z["index"])
        S = ev.subset(D, pos)
        perm = np.empty(len(S.legal), np.int64)
        for r in range(S.n):
            a0, b0 = z["offsets"][r], z["offsets"][r + 1]
            slot = {
                int(t): S.offsets[r] + j
                for j, t in enumerate(S.legal[S.offsets[r] : S.offsets[r + 1]])
            }
            perm[a0:b0] = [slot[int(t)] for t in z["legal"][a0:b0]]
        S.prior = np.empty(len(S.legal))
        S.prior[perm] = z["prior"]
        w = z["heads"][:, 63:66].astype(float)
        w = np.exp(w - w.max(1, keepdims=True))
        v0 = (w[:, 0] - w[:, 2]) / w.sum(1)
        print(run)
        for b in [int(x) for x in a.budgets.split(",") if int(x) in budgets]:
            q = np.full(len(S.legal), np.nan)
            q[perm] = z[f"q{b}"]
            seen = ~np.isnan(q)
            for f, bin_ in CELLS:
                s = (S.f == f) & (S.bin == bin_)
                if not s.any():
                    continue
                C = ev.subset(S, np.flatnonzero(s))
                qc, sc = q[C.flat], seen[C.flat]
                out = [f"{c.FORMATS[f][:5]} {bin_} b{b:<5d} moves {c.segsum(sc, C.offsets).mean() / np.diff(C.offsets).mean():.2f}"
                       f" mass {c.segsum(sc * C.prior, C.offsets).mean():.3f}"]  # fmt: skip
                for name, qq in (
                    ("root", np.where(sc, qc, v0[C.rows][C.seg])),
                    ("mean", ev.fill(C, qc, v0[C.rows])),
                ):
                    top = np.maximum.reduceat(qq, C.offsets[:-1])[C.seg]
                    arg = (qq >= top - 1e-9).astype(float)
                    arg /= c.segsum(arg, C.offsets)[C.seg]
                    out.append(
                        f"argmax acc ({name}) {ev.expect(C, arg)[:, 0].mean():.3f}"
                    )
                out.append(f"human {C.hmet[:, 0].mean():.3f}")
                print("  " + "  ".join(out), flush=True)


if __name__ == "__main__":
    main()
