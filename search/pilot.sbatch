#!/bin/bash
#SBATCH --job-name=search-v1-pilot
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-x9-32
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/worktrees/search-v1/results/search-v1/logs/%x-%j.out
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/search-v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
python_bin=$(/home/yimingz3/src/allie/.venv/bin/python /data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours/modded_runtime_stage.py)
export PYTHONPATH=$PWD/vendor:/data/group_data/dei-group/yimingz3/allie/envs/chessmix-overlay
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=1 search/cache.py
