#!/bin/bash
#SBATCH --job-name=calib-final
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
# final.sbatch CALIB_DIR SEARCH_DIR TAG: calib_final.sh in run.sbatch's environment
set -uo pipefail
HERE=/home/yimingz3/src/allie-wt-elo/analysis/elo_strength
export PYTHONPATH=/home/yimingz3/src/allie-wt-elo-pkg/src:/data/group_data/dei-group/yimingz3/allie/envs/search-overlay:$HERE
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
date -Is; hostname
cd $HERE
bash calib_final.sh "$@"
echo "exit $?"; date -Is
