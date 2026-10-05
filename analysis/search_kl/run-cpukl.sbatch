#!/bin/bash
#SBATCH --job-name=pikl-cpukl
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/search-eff/pikl/logs/%x-%A_%a.log
# sbatch --array=0-(n-1) run-cpukl.sbatch OUT N [cpu_kl.py args]: shard $SLURM_ARRAY_TASK_ID of N on the bot's fast cpu backend
set -uo pipefail
export PYTHONPATH=/home/yimingz3/src/allie-wt-pikl/src:/data/group_data/dei-group/yimingz3/allie/envs/search-overlay
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
date -Is; hostname
cd /home/yimingz3/src/allie-wt-pikl/analysis/search_kl
/home/yimingz3/src/allie/.venv/bin/python cpu_kl.py "$1" --shard ${SLURM_ARRAY_TASK_ID:-0} --shards "$2" "${@:3}"
echo "exit $?"; date -Is
