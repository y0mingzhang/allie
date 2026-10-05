"""Each searching cell's rungs for the bot (allie.lichess.calibration.CELLS). A "capped T bB" pick
(calib_cells.py --think, scored as the bot plays it under the think-time cap): every coverage rung up
to T at beta B. A fixed "coverage T bB" pick: for every budget up to T, the beta (None: the calibrated
distribution) with the smallest debiased Elo error among those whose human-move cross-entropy is
below the policy's at 95%; a budget with none is left out.

python calib_rungs.py TABLE.json [EXTRA_TABLE.json ...] --out rungs.json
TABLE: calib_cells.py's output (its picks set each cell's top budget); EXTRA tables add candidates and
override picks for their cells (e.g. the 1024-simulation run for classical 2400 and 2600).
"""

import argparse
import json

import numpy as np

LADDER = (8, 32, 128, 256, 1024)


def loss(s):
    return (
        (s["elo_accuracy"] ** 2 - s["se_accuracy"] ** 2)
        + (s["elo_blunder"] ** 2 - s["se_blunder"] ** 2)
    ) / 2


def qualifies(s):
    return s["ce"] + 1.96 * s["se_ce"] <= 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tables", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tables = [json.load(open(t)) for t in a.tables]
    out = {}
    for k, v in tables[0].items():
        cands, pick = dict(v["all"]), v
        for t in tables[1:]:
            if k in t:
                cands.update(t[k]["all"])
                pick = t[k] if t[k]["kind"] != "raw" else pick
        if pick["kind"] == "capped":
            lad = pick.get("ladder") or [str(r) for r in LADDER if r <= pick["budget"]]
            out[k] = {int(t) if t.isdigit() else t: pick["beta"] for t in lad}
            print(f"{k:15s} {pick['choice']}: rungs up to {pick['budget']} at beta {pick['beta']}")
            continue
        if pick["kind"] != "coverage" or not isinstance(pick["budget"], int):
            continue
        rungs = {}
        for r in LADDER:
            if r > pick["budget"]:
                break
            ok = [
                (loss(s), n)
                for n, s in cands.items()
                if n.split()[:2] == ["coverage", str(r)] and qualifies(s)
            ]
            if ok:
                n = min(ok)[1].split()
                rungs[r] = float(n[2][1:]) if len(n) > 2 else None
        assert rungs.get(pick["budget"], "x") == pick["beta"], (
            k,
            rungs,
            pick["choice"],
        )
        out[k] = rungs
        print(
            f"{k:15s} "
            + "  ".join(f"{r}: {b}" for r, b in sorted(rungs.items(), reverse=True)),
            f"(rms {np.sqrt(max(loss(pick['stats']), 0)):.0f})",
        )
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
