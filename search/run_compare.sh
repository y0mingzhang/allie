#!/bin/bash
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
export PYTHONPATH=$PWD/vendor
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cpu_python=/home/yimingz3/src/allie/.venv/bin/python
"$cpu_python" -B search/reply_pilot.py > results/search-v1/logs/reply-pilot.log 2>&1
"$cpu_python" -B search/select_search.py > results/search-v1/logs/reply-selection.log 2>&1
"$cpu_python" -B search/mcts_confirm.py > results/search-v1/logs/mcts-confirmation.log 2>&1
