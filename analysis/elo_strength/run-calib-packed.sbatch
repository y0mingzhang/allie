#!/bin/bash
#SBATCH --job-name=calib-ann
#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/logs/%x-%j.log
# run-calib-packed.sbatch POSITIONS OUT SHARDS FIRST [calib_search args]: shards FIRST..FIRST+ngpu-1 of SHARDS,
# one process per GPU (CKPT, SRC from the environment, as run-calib.sbatch)
set -uo pipefail
R=/home/yimingz3/src/allie
SRC=${SRC:-$R/results/recipe10x/bigfix-24x1536d75m4shipv2nf-c8s200f0v4/source-ours}
CKPT=${CKPT:-/data/group_data/dei-group/yimingz3/allie/maia3-bench/search/ckpt-bigrun-143051/model.pt}
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=":4096:8" CUDA_MODULE_LOADING=LAZY DISABLE_FP8=1
export TORCHINDUCTOR_CACHE_DIR=/scratch/yimingz3/allie/strat-inductor
export TRITON_CACHE_DIR=/scratch/yimingz3/allie/strat-triton
export PYTHONPATH=/data/group_data/dei-group/yimingz3/allie/envs/search-overlay
PY=$($R/.venv/bin/python $SRC/modded_runtime_stage.py)
date -Is; hostname; nvidia-smi --query-gpu=name --format=csv,noheader
cd /home/yimingz3/src/allie-wt-elo/analysis/elo_strength
n=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((n - 1))); do
  CUDA_VISIBLE_DEVICES=$g $PY calib_search.py "$1" "$2" --shard $(($4 + g)) --shards "$3" --checkpoint $CKPT --source $SRC "${@:5}" &
done
wait
date -Is
