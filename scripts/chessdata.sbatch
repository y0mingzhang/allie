#!/bin/bash
#SBATCH --job-name=data-v1-build
#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/data-v1/%x-%j.log

# Usage: sbatch scripts/chessdata.sbatch 2026-08
set -euo pipefail
month=${1:?month, e.g. 2026-08}
D=${STORE:-/data/group_data/dei-group/yimingz3/allie/data-v1}/$month
PY=/home/yimingz3/src/allie/.venv/bin/python
S=/home/yimingz3/src/allie/scripts/chessdata.py
if [ ! -f "$D/build-complete.json" ]; then
	if [ -d "$D/hf" ]; then
		$PY "$S" build --hf "$D/hf" --out "$D"
	else
		$PY "$S" build --pgn "$D/lichess_db_standard_rated_$month.pgn.zst" --out "$D"
	fi
fi
[ -f "$D/stats.json" ] || $PY "$S" finalize --out "$D"  # never re-finalize a published month
