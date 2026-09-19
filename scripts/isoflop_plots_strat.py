"""isoflop-v1 plots on the golden eval: isoflop_plots.py fed strat-eval-v1 macro / expert macro CE.

Writes results/recipe10x/strat-eval-v1/plots/{isoflop-curves,frontier-strat_macro,frontier-strat_expert}.png.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isoflop_fit as f
import isoflop_plots as plots

EVAL = f.STUDY.parents[1] / "lm-eval"
KEY = dict(strat_macro="macro", strat_expert="expert_macro")


def load(metric):
    out = {}
    for recipe in f.RECIPES:
        pts = []
        for p in sorted((f.STUDY / "results").glob(f"iso-{recipe}-*.json")):
            s = EVAL / p.stem / "strat-v1.json"
            if s.exists():
                x = json.loads(p.read_text())
                pts.append(
                    (
                        x["n_nonembed"],
                        x["tokens"],
                        x["budget"],
                        json.loads(s.read_text())[KEY[metric]],
                    )
                )
        out[recipe] = np.array(pts)
    return out


f.load = load
plots.METRIC = dict(
    strat_macro="Golden macro CE (16 cells)",
    strat_expert="Golden expert macro CE (≥2400)",
)
plots.OUT = f.STUDY.parent / "strat-eval-v1" / "plots"
plots.OUT.mkdir(parents=True, exist_ok=True)
plots.curves()
for m in KEY:
    plots.frontier(m, boot=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
