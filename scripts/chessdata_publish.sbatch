#!/bin/bash
#SBATCH --account=dippolit
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Usage (from chessdata_local.sbatch, pinned to its node, one at a time via singleton): copy a finalized month from
# /scratch to STORE at a bounded rate. stats.json is copied last: it marks the month built for the queue and sampler.
set -euo pipefail
month=${1:?} L=/scratch/$USER/chessdata/$1 D=${STORE:?}/$1
[ -f "$D/stats.json" ] && { rm -rf "$L"; exit 0; }
mkdir -p "$D"
ionice -c2 -n7 rsync -a --bwlimit=${BWLIMIT:-60000} --exclude stats.json "$L/" "$D/"
cp "$L/stats.json" "$D/stats.json.tmp" && mv "$D/stats.json.tmp" "$D/stats.json"
touch "$D/download-verified" "$D/cleaned"
rm -rf "$L" "$D.publish-job"
echo "$(date -Is) $month published from $(hostname -s)"
