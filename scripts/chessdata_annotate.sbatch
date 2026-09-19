#!/bin/bash
#SBATCH --job-name=data-annotate
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/data-v1/annotate-%A_%a.log

# Usage: sbatch --array=0-N%1 scripts/chessdata_annotate.sbatch MONTH_DIR...  (one month per task)
# Re-downloads the month from Hugging Face (verified), backfills end + evals, removes the download.
set -euo pipefail
dirs=("$@")
D=${dirs[$SLURM_ARRAY_TASK_ID]}
grep -qs '"evals": true' "$D/annotate-complete.json" && { echo "$D already annotated"; exit 0; }
PY=/home/yimingz3/src/allie/.venv/bin/python
S=/home/yimingz3/src/allie/scripts
if $PY $S/hf_fetch.py "$(basename "$D")" "$D/hf-annotate"; then
	$PY $S/chessdata_annotate.py "$D" "$D/hf-annotate"
else
	echo "no Hugging Face source for $D: end state only, evals left for a PGN pass"
	$PY $S/chessdata_annotate.py "$D"
fi
rm -rf "$D/hf-annotate"
