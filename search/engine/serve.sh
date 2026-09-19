#!/bin/bash
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
export PATH="$PWD/results/search-v1/runtime/sglang-0.5.9/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_EXTERNAL_MODEL_PACKAGE=search.engine.sglang_models
export TORCHINDUCTOR_COMPILE_THREADS=4
export OMP_NUM_THREADS=4
exec python -m sglang.launch_server \
  --model-path results/search-v1/serving-export --trust-remote-code \
  --skip-tokenizer-init --dtype bfloat16 --attention-backend flashinfer \
  --mem-fraction-static 0.25 --max-total-tokens 16384 --context-length 1025 \
  --disable-cuda-graph --disable-overlap-schedule \
  --host 127.0.0.1 --port 43552 "$@"
