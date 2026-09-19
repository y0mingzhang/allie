#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUNTIME="${SEARCH_ENGINE_RUNTIME:-$PWD/results/search-v1/runtime/sglang-0.5.9}"
export ALLIE_SERVICE_STARTED="$(date +%s)"
export PATH="$RUNTIME/bin:$PATH" PYTHONPATH="$PWD" OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
exec "$RUNTIME/bin/python" -m search.engine.service
