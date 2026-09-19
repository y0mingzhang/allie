"""Publish the data baseline B_2 for the model track: results/recipe10x/data-v1-B2.json.

usage: write_b2.py STUDY RUN_NAME [POLICY]
Copies the chosen run's data/input/objective settings from its frozen plan (POLICY, if given,
replaces its mixture policy; the ledger records why), adds the per-budget
repetition-matched pool fractions (screen tokens / final tokens) and the hashes the model track pins.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataexp

KEYS = (
    "policy",
    "clock",
    "feats",
    "input_lr",
    "aux_time",
    "aux_wdl",
    "history",
    "stores",
    "months",
)


def main(study, name, policy=None):
    root = dataexp.ROOT / "results/recipe10x"
    plan = json.loads((root / study / "plan.json").read_text())
    run = next(r for r in plan["runs"] if r["name"] == name)
    run = run | ({"policy": policy} if policy else {})
    counts = root / study / "history-counts.json"
    b2 = {k: run[k] for k in KEYS if k in run} | dict(
        source_study=study,
        source_run=name,
        final_tokens=dataexp.FINAL_TOKENS,
        pool_frac={b: dataexp.pool_frac(b) for b in ("3e16", "1e17")},
        history_counts=str(counts),
        history_counts_sha256=hashlib.sha256(counts.read_bytes()).hexdigest(),
        chessmix_sha256=hashlib.sha256(
            (root / study / "source-ours/chessmix.py").read_bytes()
        ).hexdigest(),
    )
    b2["sha256"] = hashlib.sha256(json.dumps(b2, sort_keys=True).encode()).hexdigest()
    out = root / "data-v1-B2.json"
    assert not out.exists(), (
        "B_2 already published; move it aside deliberately to replace it"
    )
    out.write_text(json.dumps(b2, indent=1) + "\n")
    print(out, b2["sha256"][:12])


if __name__ == "__main__":
    main(*sys.argv[1:])
