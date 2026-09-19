#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD/results/search-v1"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PYTHON="$ROOT/runtime/sglang-0.5.9/bin/python"
for name in 014-mcts1000 015-depth6; do
  while [[ ! -f "$ROOT/engine-queue/$name.result.json" ]]; do
    [[ ! -f "$ROOT/STOP" && ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]] || exit 0
    [[ ! -f "$ROOT/engine-queue/$name.error.json" ]] || exit 1
    sleep 15
  done
  if [[ "$name" == 014-mcts1000 ]]; then
    "$PYTHON" -B -m search.engine.analyze_adaptive "$ROOT/mcts1000-pilot"
  else
    "$PYTHON" -B -m search.engine.analyze_depth "$ROOT/depth6-pilot"
  fi
  "$PYTHON" -B -m search.report
 done
