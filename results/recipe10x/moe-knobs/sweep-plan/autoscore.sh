#!/bin/bash
# autoscore.sh STUDY...: every 2 min submit moe-knobs/score.sbatch (1 x A6000, score_handback.py) once for each orchard
# hand-back of the given studies (results/recipe10x/STUDY; orchard.json + done.json at steps, no result yet); logged in
# moe-knobs/logs-score/autoscore-sweep.log. Never cancels anything.
cd /home/yimingz3/src/allie
K=results/recipe10x/moe-knobs L=$K/logs-score/autoscore-sweep.log
mkdir -p $K/logs-score
while true; do
	for st in "$@"; do
		python3 - "results/recipe10x/$st" <<'PY' | while read -r i name; do
import json, sys
from pathlib import Path
s = Path(sys.argv[1])
for i, r in enumerate(json.loads((s / "plan.json").read_text())["runs"]):
    out = Path("results/pretrain") / r["name"]
    done = out / "done.json"
    if (out / "orchard.json").exists() and done.exists() and json.loads(done.read_text())["stop_reason"] == "steps" \
            and not (s / "results" / f"{r['name']}.json").exists():
        print(i, r["name"])
PY
			grep -q "^$name " $L 2>/dev/null && continue
			job=$(sbatch --parsable --job-name=swscore-$name $K/score.sbatch results/recipe10x/$st $i)
			echo "$name $st $i job $job $(date -Is)" >> $L
		done
	done
	sleep 120
done
