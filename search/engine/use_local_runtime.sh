#!/usr/bin/env bash
# Switch only our idle private engine after the local cache is complete.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="$PWD/results/search-v1"
CACHE=/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128
STOP=/data/group_data/dei-group/yimingz3/allie/controller/STOP
while [[ ! -f "$CACHE/STAGED.json" ]]; do
  [[ ! -f "$OUT/STOP" && ! -f "$STOP" ]] || exit 0
  sleep 5
done
[[ ! -f "$OUT/STOP" && ! -f "$STOP" ]] || exit 0
old_pid=$(/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
q=Path('results/search-v1/engine-queue')
for p in q.glob('*.request.json'):
    stem=p.name.removesuffix('.request.json')
    assert (q/(stem+'.result.json')).exists() or (q/(stem+'.error.json')).exists(),p
print(json.loads((q/'ready.json').read_text())['pid'])
PY
)
[[ "$old_pid" =~ ^[0-9]+$ ]]
command_line=$(ps -p "$old_pid" -o args=)
[[ "$command_line" == *" -m search.engine.service" ]] || { echo 'Engine PID identity mismatch'; exit 1; }
kill -TERM "$old_pid"
for _ in {1..60}; do
  kill -0 "$old_pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$old_pid" 2>/dev/null; then echo 'Old engine did not exit; no second runner launched'; exit 1; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
p=Path('results/search-v1/engine-queue/008-local-final.request.json')
assert not p.exists()
p.write_text(json.dumps(dict(kind='experiment',module='search.engine.benchmark_v2'))+'\n')
PY
export SEARCH_ENGINE_RUNTIME="$CACHE"
exec bash search/engine/workbench.sh >> "$OUT/logs/engine-local.log" 2>&1
