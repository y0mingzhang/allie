#!/bin/bash
# score-run.sh RUN BACKUP [VALUE_VIEWS] [shared]: a gpu_kl.py run under one kl backup as harness output (to_harness.py),
# scored by scale_score.py (family pikl, name "RUN BACKUP[ vVIEWS]") into R/pikl/curves.jsonl, or the sweep's shared
# curves with "shared"; one cpu job
set -euo pipefail
R=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/search-eff/pikl
run=$1 backup=$2 views=${3:-} where=${4:-}
tag=$(echo "$backup${views:+-v$views}" | tr ':,' '_-')
name="$run $backup${views:+ v$views}"
out=$R/curves.jsonl
[ "$where" = shared ] && out=$R/../scaling/curves.jsonl
H=/home/yimingz3/src/allie-wt-pikl/analysis/search_kl
sbatch --job-name=pikl-sc-$run-$tag $H/run-cpu.sbatch -c "
import runpy, sys
sys.argv = ['to_harness.py', '$R/$run', '$R/h/$run-$tag', '--backup', '$backup', '--budgets', '${BUDGETS:-8,16,32,64,128,256,512,1024,2048,4096}'] + (['--value-views', '$views'] if '$views' else [])
runpy.run_path('to_harness.py', run_name='__main__')
sys.path.insert(0, '/home/yimingz3/src/allie-wt-seff/analysis/search_eff')
sys.argv = ['scale_score.py', '$R/h/$run-$tag', '--family', 'pikl', '--name', '$name', '--out', '$out']
runpy.run_path('/home/yimingz3/src/allie-wt-seff/analysis/search_eff/scale_score.py', run_name='__main__')
"
