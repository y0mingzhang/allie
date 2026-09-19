#!/bin/bash
#SBATCH --job-name=search-v1-workbench
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|L40|6000Ada|RTX_PRO_6000|A100_40GB|A100_80G|H200|A6000
#SBATCH --exclude=babel-x9-32
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/worktrees/search-v1/results/search-v1/logs/%x-%j.out
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
python_bin=$(/home/yimingz3/src/allie/.venv/bin/python /data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours/modded_runtime_stage.py)
export PYTHONPATH=$PWD/vendor:/data/group_data/dei-group/yimingz3/allie/envs/chessmix-overlay
export SEARCH_PYTHON="$python_bin"
export SEARCH_TMUX_SOCKET="search-v1-$SLURM_JOB_ID"
printf '%s\n' "$SLURM_JOB_ID $(hostname) $SEARCH_TMUX_SOCKET $python_bin" > results/search-v1/allocation.txt
# This session is isolated from the user's tunnel/controller tmux server.
tmux -L "$SEARCH_TMUX_SOCKET" new-session -d -s oracle "bash search/worker.sh"
trap 'tmux -L "$SEARCH_TMUX_SOCKET" kill-server 2>/dev/null || true' EXIT
while tmux -L "$SEARCH_TMUX_SOCKET" has-session -t oracle 2>/dev/null; do
    if [[ -f results/search-v1/STOP ]]; then break; fi
    sleep 5
done
