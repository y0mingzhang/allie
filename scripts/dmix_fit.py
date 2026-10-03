"""The golden law and compute multipliers against a control curve.

law(metric) is the isoflop-v1 law for ours, L(N, D) with budgets N*D: original-val metrics
(move, expert2400) use its saved shared-floor fit; golden metrics (strat_macro over the 16
cells, strat_expert over the four >= 2400 cells) refit it on the isoflop-v1 ours runs'
strat-eval-v1 scores. A run's multiplier at budget C is C'/C where curve(C') = its loss,
curve being the law shifted through the control runs.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isoflop_fit import additive

ROOT = (
    Path(os.environ.get("ALLIE_PROJECT_ROOT", "/home/yimingz3/src/allie"))
    / "results/recipe10x"
)
EVAL = ROOT.parent / "lm-eval"


def strat(name):
    f = EVAL / name / "strat-v1.json"
    if not f.exists():
        return dict(strat_macro=np.nan, strat_expert=np.nan)
    s = json.loads(f.read_text())
    return dict(strat_macro=s["macro"], strat_expert=s["expert_macro"])


def law(metric):
    if metric in ("move", "expert2400"):
        f = json.loads((ROOT / f"isoflop-v1/fit-{metric}.json").read_text())
        f = f["shared_E"]["fit"]["ours"]
        return np.array(
            [np.log(f["E"]), np.log(f["A"]), f["alpha"], np.log(f["B"]), f["beta"]]
        )
    pts = []
    for p in sorted((ROOT / "isoflop-v1/results").glob("iso-ours-*.json")):
        r = json.loads(p.read_text())
        pts.append((r["n_nonembed"], r["tokens"], r["budget"], strat(p.stem)[metric]))
    pts = np.array([x for x in pts if np.isfinite(x[3])])
    assert len(pts) >= 8, f"only {len(pts)} isoflop-v1 runs scored on strat-eval-v1"
    return additive(dict(ours=pts), shared=False)[:5]


def multiplier(curve, loss, c):
    if not np.isfinite(loss):
        return float("nan")
    g = lambda lc: curve(np.exp(lc)) - loss
    lo, hi = np.log(c) - 9, np.log(c) + 9
    return float(np.exp(brentq(g, lo, hi)) / c) if g(lo) > 0 > g(hi) else float("nan")
