#!/bin/bash
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
# Reuse the one allocation; defer CPU workers until the preceding study finishes.
while [[ ! -f results/search-v1/mcts-confirmation/results.json ]]; do
    [[ ! -f results/search-v1/STOP ]]
    [[ ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]]
    sleep 10
done
export PYTHONPATH=$PWD/vendor
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /home/yimingz3/src/allie/.venv/bin/python -B search/reply_confirm.py > results/search-v1/logs/reply-confirmation.log 2>&1
