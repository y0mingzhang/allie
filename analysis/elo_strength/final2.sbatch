#!/bin/bash
#SBATCH --job-name=calib-final2
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
# final2.sbatch CALIB_DIR: refit coverage's output calibration on the annealed dev outputs, recompute the
# 220k positions' searched distributions under it, then the rule fits, the cross-entropy table and the figure
set -uo pipefail
C=$1
HERE=/home/yimingz3/src/allie-wt-elo/analysis/elo_strength
export PYTHONPATH=/home/yimingz3/src/allie-wt-elo-pkg/src:/data/group_data/dei-group/yimingz3/allie/envs/search-overlay:$HERE
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
PY=/home/yimingz3/src/allie/.venv/bin/python
date -Is; hostname
cd $HERE
$PY calib_outcal.py $C/positions-dev.npz $C/search-dev-ann2 /home/yimingz3/src/allie-wt-fasttree2/src/allie/lichess/calibration-allie-2.0.json $C/calibration-allie-2.0-annealed.json || exit 1
$PY calib_recal.py $C/positions-human.npz $C/search-human-ann2 $C/calibration-allie-2.0-annealed.json $C/search-human-ann2c || exit 1
bash calib_final.sh $C search-human-ann2c ann2c
echo "exit $?"; date -Is
