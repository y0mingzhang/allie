#!/bin/bash
#SBATCH --job-name=search-engine
#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-x9-32,babel-q9-32
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/worktrees/search-v1/results/search-v1/logs/%x-%j.out
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
[[ ! -f results/search-v1/STOP && ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]] || exit 0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
export SEARCH_TMUX_SOCKET="search-v1-$SLURM_JOB_ID"
printf '%s\n' "$SLURM_JOB_ID $(hostname) $SEARCH_TMUX_SOCKET" > results/search-v1/allocation.txt
# A noninteractive pane avoids queued shell input respawning a stale engine.
tmux -L "$SEARCH_TMUX_SOCKET" new-session -d -s oracle "bash search/engine/workbench.sh >> results/search-v1/logs/engine-$SLURM_JOB_ID.log 2>&1"
trap 'tmux -L "$SEARCH_TMUX_SOCKET" kill-server 2>/dev/null || true' EXIT
while tmux -L "$SEARCH_TMUX_SOCKET" has-session -t oracle 2>/dev/null; do
    [[ ! -f results/search-v1/STOP && ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]] || break
    sleep 5
done
