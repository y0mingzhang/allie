"""Projected finish of every unfinished run in the given studies, flagging barrier stragglers.

usage: straggler.py STUDY...   (e.g. data-v1-round2p data-v1-round2s)
A run is a straggler when its projected finish is more than 30% past its study's median projection.
Report only: moving a run (checkpoint + resubmit on a fast free GPU) is a separate, deliberate step.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/yimingz3/src/allie")
SLACK = 1.3


def projection(name, steps):
    """Seconds until training ends, from the recent step rate; None before the second log line."""
    log = ROOT / "results/pretrain" / name / "train.jsonl"
    rows = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    # rows since the last restart (step goes back or tok/s collapses), minus its startup row
    k = len(rows) - 1
    while (
        k > 0
        and rows[k - 1]["step"] < rows[k]["step"]
        and (rows[k]["tokens_per_second"] > 0.2 * rows[k - 1]["tokens_per_second"])
    ):
        k -= 1
    rows = rows[k + 1 :]
    if len(rows) < 2:
        return None
    a, b = rows[max(0, len(rows) - 9)], rows[-1]
    rate = (b["seconds"] - a["seconds"]) / max(1, b["step"] - a["step"])
    age = time.time() - log.stat().st_mtime
    return max(0.0, (steps - b["step"]) * rate - age), b["step"], rate


def main():
    for s in sys.argv[1:]:
        study = ROOT / "results/recipe10x" / s
        plan = json.loads((study / "plan.json").read_text())
        rows = []
        for r in plan["runs"]:
            n = r["name"]
            if not (study / "results" / f"{n}.json").exists():
                rows.append((n, projection(n, plan["sizes"][r["budget"]]["steps"])))
        known = [p[0] for _, p in rows if p]
        median = float(np.median(known)) if known else 0.0
        for n, p in rows:
            if p is None:
                print(f"{s}: {n}: not started")
                continue
            left, step, rate = p
            flag = (
                "  STRAGGLER"
                if known and left > SLACK * median and left - median > 900
                else ""
            )
            print(
                f"{s}: {n}: step {step}, {rate:.2f} s/step, ~{left / 60:.0f} min left{flag}"
            )
    print(
        subprocess.run(
            ["date", "+%H:%M"], capture_output=True, text=True
        ).stdout.strip()
    )


if __name__ == "__main__":
    main()
