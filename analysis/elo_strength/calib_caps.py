"""Chi-square and per-time-control errors of the chosen rule at lower caps (calib_rule.py's data).

python calib_caps.py CALIB_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np

import calib_fit as cf
import calib_rule as cr

d = Path(sys.argv[1])
cr.HW = 10.0
A = cr.precompute(d)
human = cf.summarize(cf.table(np.load(d / "positions-human.npz")["meta"], cf.load_mpv(d / "mpv-human"), {}, {}), [])
cr.prepare(A, human)
everything = np.ones(len(A["elo"]), bool)
for cap in (0, 8, 16, 32, 64, 128, 256):
    par = (-0.1, -0.2, 1.414 if cap else 0.0, 3.0, max(cap, 1))
    g = cr.gaps(A, cr.values(A, *par), everything)
    print(cap, round(cr.chi2(g)), json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in cr.report(g).items()}), flush=True)
