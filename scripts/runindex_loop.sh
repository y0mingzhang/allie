#!/bin/bash
# runindex_loop.sh: the runs dashboard's refresh loop. Every EVERY s (600): orchard mirror, compute snapshot, index,
# page (results/runindex/allie-runs.html). CPU only, nice 19, idle IO class; one loop per lock.
# Start: setsid nohup scripts/runindex_loop.sh >> results/runindex/loop.log 2>&1 < /dev/null &
# Stop:  pkill -f scripts/runindex_loop.sh
set -u
A=/home/yimingz3/src/allie
O=$A/results/runindex every=${EVERY:-600} py=$A/.venv/bin/python
mkdir -p "$O"
exec 9> "$O/loop.lock"
flock -n 9 || { echo "runindex loop already running" >&2; exit 1; }
renice -n 19 -p $$ > /dev/null
ionice -c 3 -p $$ 2> /dev/null
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
step() { timeout "$@" 9>&- || echo "$(date -Is) failed ($?): ${*:2}"; }
while :; do
	t=$(date +%s)
	step 300 "$A/scripts/orchard_mirror.sh" "$O/orchard"
	step 300 "$py" "$A/scripts/runindex.py" compute
	step 900 "$py" "$A/scripts/runindex.py" index
	step 120 "$py" "$A/scripts/runindex.py" page "$O/allie-runs.html"
	dt=$(($(date +%s) - t))
	echo "$(date -Is) refreshed in ${dt} s"
	sleep $((every - dt > 60 ? every - dt : 60)) 9>&-
done
