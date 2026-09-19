#!/bin/bash
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
export PYTHONPATH=$PWD/vendor
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
while ! grep -q '"context_impl": "block-metadata-v1"' results/search-v1/server-ready.json 2>/dev/null; do
    [[ ! -f results/search-v1/STOP ]]
    [[ ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]]
    sleep 5
done
cpu_python=/home/yimingz3/src/allie/.venv/bin/python
case "$1" in
  reply)
    "$cpu_python" -B search/reply_confirm.py > results/search-v1/logs/reply-confirmation.log 2>&1
    "$cpu_python" -B search/reply_calibration_confirm.py > results/search-v1/logs/reply-calibration-confirmation.log 2>&1
    ;;
  player)
    "$cpu_python" -B search/player_pilot.py > results/search-v1/logs/player-pilot.log 2>&1
    ;;
  *) exit 2 ;;
esac
