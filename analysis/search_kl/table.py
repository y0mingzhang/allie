"""Scaling rows (scale_score.py jsonl) as a table: per run name and budget, each dev cell's CE at matched accuracy
minus raw (x: unreachable, the best reachable accuracy gap in Elo).

python table.py [--curves a.jsonl,b.jsonl] [--names substr,...]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

R = "/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/search-eff"
CELLS = ["classical/2400", "classical/2600", "rapid/2600", "blitz/2400"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--curves", default=f"{R}/pikl/curves.jsonl,{R}/scaling/curves.jsonl"
    )
    p.add_argument("--names", default="")
    p.add_argument("--elo", action="store_true", help="also the accuracy Elo gap at the CE-optimal tilt")
    a = p.parse_args()
    rows = {}
    for f in [x for x in a.curves.split(",") if Path(x).exists()]:
        for line in open(f):
            r = json.loads(line)
            rows[r["family"], r["name"], r["cell"], r["budget"]] = r
    want = [s for s in a.names.split(",") if s]
    by = defaultdict(dict)
    for (fam, name, cell, b), r in rows.items():
        if not want or any(s in f"{fam}: {name}" for s in want):
            by[f"{fam}: {name}", b][cell] = r
    print(
        f"{'run':44s} {'budget':>6s} {'leaves':>6s}  "
        + "  ".join(f"{c:>15s}" for c in CELLS)
    )
    for (name, b), cs in sorted(by.items()):
        leaves = max(r["leaves"] for r in cs.values())
        cols = []
        for c in CELLS:
            r = cs.get(c)
            cols.append(
                ""
                if r is None
                else (f"{r['ce_match']:+.4f}" if r["ce_match"] is not None else f"x{r['reach']:+.0f}")
                + (f" {r['elo_acc_opt']:+.0f}" if a.elo else "")
            )
        print(
            f"{name[:44]:44s} {b:6d} {leaves:6.0f}  "
            + "  ".join(f"{x:>15s}" for x in cols)
        )


if __name__ == "__main__":
    main()
