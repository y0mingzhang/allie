#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CACHE=/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128
RUNTIME="${SEARCH_ENGINE_RUNTIME:-}"
if [[ -z "$RUNTIME" ]]; then
  if [[ -f "$CACHE/STAGED.json" ]]; then RUNTIME="$CACHE"; else RUNTIME="$PWD/results/search-v1/runtime/sglang-0.5.9"; fi
fi
export ALLIE_SERVICE_STARTED="$(date +%s)"
export PATH="$RUNTIME/bin:$PATH" PYTHONPATH="$PWD" OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
exec "$RUNTIME/bin/python" -m search.engine.service
