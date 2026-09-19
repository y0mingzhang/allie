#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ARCHIVE="$PWD/results/search-v1/runtime/sglang-0.5.9-torch2.9.1-cu128.tar.zst"
while [[ ! -f "$ARCHIVE.sha256" ]]; do
  [[ ! -f results/search-v1/STOP && ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]] || exit 0
  # The durable flock prevents two writers. A controller-local build can finish
  # normally; this allocation takes over only if that process dies on restart.
  bash search/engine/pack_runtime.sh
  [[ ! -f "$ARCHIVE.sha256" ]] || break
  sleep 15
done
