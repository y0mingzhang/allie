#!/bin/bash
#SBATCH --job-name=elo-strength
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/logs/%x-%A_%a.log
# run.sbatch SCRIPT ARGS...: a python script of analysis/elo_strength with allie.lichess pinned at 7e39e50
set -uo pipefail
HERE=/home/yimingz3/src/allie-wt-elo/analysis/elo_strength
export PYTHONPATH=/home/yimingz3/src/allie-wt-elo-pkg/src:/data/group_data/dei-group/yimingz3/allie/envs/search-overlay:$HERE
export ALLIE_CHESS_INCLUDE=/home/yimingz3/src/allie/vendor/chess-library/include
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
date -Is; hostname; lscpu | grep "Model name"
git -C /home/yimingz3/src/allie-wt-elo-pkg log --oneline -1 | cut -c1-60
cd $HERE
/home/yimingz3/src/allie/.venv/bin/python "$@"
echo "exit $?"; date -Is
