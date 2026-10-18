"""The live bot's score against its opponents, from its log: overall, by format and by opponent rating band.

    python analysis/lichess/live_score.py LOG [LOG ...] [--since 2026-10-18]

Performance is 400 log10(s / (1 - s)) Elo above the opponents, with a delta-method standard error. "On the
board" leaves out games won on time or because the opponent left, which measure clock handling and the
opponent's engagement rather than the moves: the number calibrated play (and play.offset) is judged by.
"""

import argparse
import math
import re

START = re.compile(r"^(\S+ \S+) game (\w{8}): \w+ as (white|black), Elo (\d+) vs (\d+)(?: \w+)?, (\d+)\+(\d+)")
OVER = re.compile(r"game (\w{8}) over: (\w+), winner (\w+)")
BANDS = (0, 1000, 1200, 1400, 1600, 1800, 2000, 4000)


def speed(base, inc):
    t = base + 40 * inc
    return "bullet" if t < 179 else "blitz" if t < 479 else "rapid" if t < 1499 else "classical"


def games(paths, since=""):
    g = {}
    for path in paths:
        for line in open(path, errors="replace"):
            if (m := START.search(line)) and m[1] >= since:
                g.setdefault(m[2], dict(white=m[3] == "white", ours=int(m[4]), them=int(m[5]),
                                        speed=speed(int(m[6]), int(m[7]))))  # fmt: skip
            elif (m := OVER.search(line)) and m[1] in g and "status" not in g[m[1]]:
                g[m[1]] |= dict(status=m[2], winner=m[3])
    out = []
    for x in g.values():
        if x.get("status") in (None, "aborted", "noStart"):
            continue
        won = x["winner"] == ("white" if x["white"] else "black")
        x["score"] = 0.5 if x["winner"] == "None" else float(won)
        x["board"] = x["status"] != "timeout" and not (x["status"] == "outoftime" and x["score"] != 0.5)
        out.append(x)
    return out


def perf(rows):
    n = len(rows)
    s = sum(r["score"] for r in rows) / n
    if s in (0, 1):
        return f"n {n:5d}  score {s:.3f}"
    se = math.sqrt(sum((r["score"] - s) ** 2 for r in rows) / max(n - 1, 1) / n) / (s * (1 - s))
    offset = sum(r["ours"] - r["them"] for r in rows) / n
    return (f"n {n:5d}  score {s:.3f}  {400 * math.log10(s / (1 - s)):+4.0f} +- {400 / math.log(10) * se:3.0f} Elo"
            f"  (header offset {offset:+.0f})")  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--since", default="", help="games started from this time on (log timestamps)")
    a = p.parse_args()
    rows = games(a.logs, a.since)
    groups = [("all", rows), ("on the board", [r for r in rows if r["board"]])]
    for f in ("bullet", "blitz", "rapid", "classical"):
        groups.append((f"  {f}, on the board", [r for r in rows if r["board"] and r["speed"] == f]))
    for lo, hi in zip(BANDS, BANDS[1:]):
        groups.append((f"  opponent {lo}-{hi}, on the board", [r for r in rows if r["board"] and lo <= r["them"] < hi]))
    for name, g in groups:
        if g:
            print(f"{name:34s} {perf(g)}")


if __name__ == "__main__":
    main()
