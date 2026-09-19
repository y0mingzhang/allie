#!/usr/bin/env bash
# Run once in the independent GPU allocation; controller restarts are irrelevant.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD/results/search-v1"
while [[ ! -f "$ROOT/engine-queue/012-balanced-golden.result.json" ]]; do
  [[ ! -f "$ROOT/STOP" && ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]] || exit 0
  [[ ! -f "$ROOT/engine-queue/012-balanced-golden.error.json" ]] || exit 1
  sleep 10
done
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PYTHON="$ROOT/runtime/sglang-0.5.9/bin/python"
"$PYTHON" -B -m search.engine.analyze_balanced
"$PYTHON" -B -m search.report
