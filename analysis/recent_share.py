"""Estimated share of a recipe table's drawn tokens from Lichess months since YYYY-MM, per format x Elo cell and
external store, over the run without a tail and inside a recent tail (data.mix recentNN(YYYY-MM[,p=P])), and the
passes the tail gives each bucket's recent games against the table's cap.

    .venv/bin/python analysis/recent_share.py RECIPE.json INVENTORY.json 2024-01 [T=0.1] [P=1]

Estimates: a bucket draws games x table weight (its passes R) x the inventory's mean tokens per game over all its
months, its months in proportion to their games at that mean length, so its recent token share is f, its recent games'
share of it. A tail over the last T draws a split bucket's (Lichess, 0 < f < 1) recent games at P, giving them
R (1 - T + T P / f) expected passes; other buckets draw as before (f = 0: old only).
"""

import json
import sys
from pathlib import Path

import numpy as np

NAMES = (
    "ultrabullet",
    "bullet",
    "blitz",
    "rapid",
    "classical",
    "correspondence",
    "other",
)
BANDS = ("<1400", "1400-2000", "2000-2400", ">=2400")
OTB = (1, 2, 3)


def cell(code):
    src, fmt, hi = code // 100000, code // 10000 % 10 - 1, code // 100 % 100 * 100
    if src:
        return "otb" if src in OTB else "engine"
    band = int(np.searchsorted((1400, 2000, 2400), hi, side="right"))
    return f"{NAMES[fmt]}/{BANDS[band]}"


def main(recipe, inventory, since, tail=0.1, p=1.0):
    r, inv = json.load(open(recipe)), json.load(open(inventory))
    cap = float(r["args"].get("cap", 8))
    recent = {}
    for m in inv["months"]:
        if Path(m).name >= since and not Path(m).name.startswith("20xx"):
            for b in json.load(open(f"{m}/buckets.json")):
                recent[b["code"]] = recent.get(b["code"], 0) + b["games"]
    buckets = []  # code, cell, est. drawn tokens, supply tokens, recent fraction, in the tail, recent passes, split
    for c, v in inv["codes"].items():
        R, code = r["weights"][c], int(c)
        f = recent.get(code, 0) / v["games"] if v["games"] else 0.0
        split = code // 100000 == 0 and 0 < f < 1
        passes = R * (1 - tail + tail * p / f) if split else R if f else 0.0
        n, s = v["games"] * R * v["tok"], v["games"] * v["tok"]
        buckets.append((code, cell(code), n, s, f, p if split else f, passes, split))
    order = [f"{n}/{b}" for n in NAMES for b in BANDS] + ["otb", "engine"]
    groups = {k: [b for b in buckets if b[1] == k] for k in order}
    groups["lichess"] = [b for b in buckets if "/" in b[1]]
    groups["all"] = buckets
    total = sum(b[2] for b in buckets)
    print(f"| cell | est. token share | passes | {since}+ share | {since}+ share in the tail "
          f"| max recent passes | buckets > {cap:g} |")  # fmt: skip
    print("|---|---|---|---|---|---|---|")
    for k, bs in groups.items():
        n = sum(b[2] for b in bs)
        if not n:
            continue
        share = sum(b[2] * b[4] for b in bs) / n
        tailed = sum(b[2] * b[5] for b in bs) / n
        over = sum(b[6] > cap for b in bs)
        print(f"| {k} | {n / total:.4f} | {n / sum(b[3] for b in bs):.2f} | {share:.3f} | {tailed:.3f} "
              f"| {max(b[6] for b in bs):.2f} | {over} |")  # fmt: skip
    for x in (cap, 6, 4):
        hit = [b for b in buckets if b[6] > x]
        print(f"recent passes > {x:g}: {len(hit)} buckets ({sum(b[7] for b in hit)} split by the tail, the rest "
              f"recent-only at their table passes), {sum(b[2] for b in hit) / total:.2%} of est. drawn tokens")  # fmt: skip
    top = max(buckets, key=lambda b: b[6])
    print(f"max {top[6]:.2f} recent passes: bucket {top[0]} ({top[1]}), R {r['weights'][str(top[0])]:.2f}, "
          f"f {top[4]:.3f}; tail T {tail:g}, P {p:g}, table cap {cap:g}")  # fmt: skip


if __name__ == "__main__":
    main(*sys.argv[1:4], *map(float, sys.argv[4:]))
