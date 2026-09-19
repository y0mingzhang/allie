#!/usr/bin/env bash
# Stage the completed archive independently of controller lifetime.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD/results/search-v1"
ARCHIVE="$ROOT/runtime/sglang-0.5.9-torch2.9.1-cu128.tar.zst"
while [[ ! -f "$ARCHIVE.sha256" ]]; do
  [[ ! -f "$ROOT/STOP" && ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]] || exit 0
  sleep 10
done
/usr/bin/python3 -B -m search.engine.stage_archive
# An already-ready engine stays resident, even if it uses the durable runtime.
if /usr/bin/python3 - <<'PY'
import json,os
from pathlib import Path
p=Path('results/search-v1/engine-queue/ready.json')
try:
 r=json.loads(p.read_text());assert r['job']==os.environ['SLURM_JOB_ID'];os.kill(r['pid'],0)
except (AssertionError,KeyError,OSError):raise SystemExit(1)
PY
then
  echo 'Engine already ready; no restart.'
else
  echo 'Cold engine not ready; restarting only our engine pane from completed local cache.'
  tmux -L "search-v1-$SLURM_JOB_ID" respawn-window -k -t oracle:0 "bash search/engine/workbench.sh >> results/search-v1/logs/engine-$SLURM_JOB_ID.log 2>&1"
fi
