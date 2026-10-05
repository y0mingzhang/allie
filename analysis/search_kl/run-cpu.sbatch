#!/bin/bash
#SBATCH --job-name=pikl-cpu
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/search-eff/pikl/logs/%x-%j.log
# run-cpu.sbatch SCRIPT [args]: a search_kl script on cpu
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd /home/yimingz3/src/allie-wt-pikl/analysis/search_kl
date -Is; hostname
/home/yimingz3/src/allie/.venv/bin/python "$@"
date -Is
