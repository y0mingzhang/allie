#!/bin/bash
#SBATCH --job-name=calib-search
#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/logs/%x-%A_%a.log
# run-calib.sbatch POSITIONS OUT SHARDS: calib_search.py on the training model (Allie 2.0, step 143051), shard $SLURM_ARRAY_TASK_ID
set -euo pipefail
R=/home/yimingz3/src/allie
SRC=${SRC:-$R/results/recipe10x/bigfix-24x1536d75m4shipv2nf-c8s200f0v4/source-ours}
CKPT=${CKPT:-/data/group_data/dei-group/yimingz3/allie/maia3-bench/search/ckpt-bigrun-143051/model.pt}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=":4096:8" CUDA_MODULE_LOADING=LAZY DISABLE_FP8=1
export TORCHINDUCTOR_CACHE_DIR=/scratch/yimingz3/allie/strat-inductor
export TRITON_CACHE_DIR=/scratch/yimingz3/allie/strat-triton
export PYTHONPATH=/data/group_data/dei-group/yimingz3/allie/envs/search-overlay
PY=$($R/.venv/bin/python $SRC/modded_runtime_stage.py)
date -Is; hostname; nvidia-smi --query-gpu=name --format=csv,noheader
cd /home/yimingz3/src/allie-wt-elo/analysis/elo_strength
$PY calib_search.py "$1" "$2" --shard $SLURM_ARRAY_TASK_ID --shards "$3" --checkpoint $CKPT --source $SRC "${@:4}"
date -Is
