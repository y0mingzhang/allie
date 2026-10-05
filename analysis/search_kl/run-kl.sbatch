#!/bin/bash
#SBATCH --job-name=pikl
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/search-eff/pikl/logs/%x-%j.log
# run-kl.sbatch OUT [gpu_kl.py args]: gpu_kl.py on every GPU of the job, one shard each, annealed Allie 2.0 (ann-all-p05-t1907)
set -uo pipefail
R=/home/yimingz3/src/allie
SRC=$R/results/recipe10x/distill-v2/source3
CKPT=/data/group_data/dei-group/yimingz3/allie/results/pretrain/ann-all-p05-t1907/last.pt
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=":4096:8" CUDA_MODULE_LOADING=LAZY DISABLE_FP8=1
export TORCHINDUCTOR_CACHE_DIR=/scratch/yimingz3/allie/pikl-inductor TRITON_CACHE_DIR=/scratch/yimingz3/allie/pikl-triton
export PYTHONPATH=/data/group_data/dei-group/yimingz3/allie/envs/search-overlay
PY=$($R/.venv/bin/python $SRC/modded_runtime_stage.py)
date -Is; hostname; nvidia-smi --query-gpu=name --format=csv,noheader
H=/home/yimingz3/src/allie-wt-pikl/analysis/search_kl
L=/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/search-eff/pikl/logs
G=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((G - 1))); do
  CUDA_VISIBLE_DEVICES=$i $PY $H/gpu_kl.py "$1" --shard $i --shards $G --checkpoint $CKPT --source $SRC "${@:2}" \
    > $L/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu$i.log 2>&1 &
done
wait
date -Is
