"""Human-move cross-entropy of calibrated rules against the raw policy, per time control x 200-point
mover-Elo cell, on the paired positions (the human's actual move under each move distribution).

python calib_ce.py CALIB_DIR --dists search-human search-human-256 [--rules rules.json] [--out ce.json]
rules.json: {name: {"params": [tau0, tau, k, gamma, cap], "hw": 40}}
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import calib_fit as cf

DEFAULT = {
    "chosen": {"params": [-0.1, -0.1, 1.0, 4.0, 256], "hw": 40},
    "temperature part": {"params": [-0.1, -0.1, 0, 0, 256]},
    "search part (T = 1)": {"params": [0, 0, 1.0, 4.0, 256], "hw": 40},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--dists", nargs="+", default=["search-human", "search-human-256"])
    p.add_argument("--rules")
    p.add_argument("--out", default="ce.json")
    a = p.parse_args()
    d = Path(a.dir)
    rules = json.loads(Path(a.rules).read_text()) if a.rules else DEFAULT
    meta = np.load(d / "positions-human.npz")["meta"]
    dists = cf.load_dists([d / x for x in a.dists])
    cells = defaultdict(lambda: defaultdict(list))
    for i, dd in dists.items():
        if "prior" not in dd or "cov_p256" not in dd:
            continue
        m = meta[i]
        h = list(dd["legal"]).index(int(m[8]))
        raw = -np.log(max(float(dd["prior"][h]), 1e-12))
        c = cells[int(m[2]), int(m[3])]
        c["raw"].append(raw)
        for name, r in rules.items():
            pi = cf.rule_mode(dd, m, tuple(r["params"]), r.get("hw", np.inf), tuple(r.get("ladder", (0, 8, 32, 128, 256))), r.get("no_bullet", False))
            c[name].append(-np.log(max(float(pi[h]), 1e-12)) - raw)
    out, names = {}, list(rules)
    print(
        "cell                n   raw CE  "
        + "  ".join(f"{n[:18]:>22s}" for n in names)
        + "   (difference vs raw, 95% CI)"
    )
    for (f, b), c in sorted(cells.items()):
        if not cf.solid(f, b):
            continue
        row = dict(n=len(c["raw"]), raw=float(np.mean(c["raw"])))
        cols = []
        for n in names:
            x = np.array(c[n])
            mean, half = float(x.mean()), float(1.96 * x.std() / np.sqrt(len(x)))
            row[n] = (mean, half)
            cols.append(f"{mean:+.4f} [{mean - half:+.4f},{mean + half:+.4f}]")
        out[f"{cf.FORMATS[f]}/{b}"] = row
        print(
            f"{cf.FORMATS[f]:9s} {b:4d} {row['n']:5d}  {row['raw']:.4f}  "
            + "  ".join(f"{s:>22s}" for s in cols)
        )
    (d / a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
