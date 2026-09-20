#!/bin/bash
#SBATCH --job-name=search-time-router
#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-x9-32,babel-q9-32
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/worktrees/search-v1/results/search-v1/logs/%x-%j.out
set -euo pipefail
cd /data/group_data/dei-group/yimingz3/allie/worktrees/time-router-v1
[[ ! -f results/search-v1/time-router-v1/STOP && ! -f /data/group_data/dei-group/yimingz3/allie/controller/STOP ]] || exit 0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH="$PWD" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29849
export TORCHINDUCTOR_CACHE_DIR=/scratch/yimingz3/allie/search-transfer/inductor
export TRITON_CACHE_DIR=/scratch/yimingz3/allie/search-transfer/triton
RUNTIME=/scratch/yimingz3/allie/search-runtime/sglang-0.5.9-torch2.9.1-cu128
[[ -f "$RUNTIME/STAGED.json" ]] || /usr/bin/python3 -B -m search.engine.stage_archive
"$RUNTIME/bin/python" -m search.time_router.service
