#!/bin/bash
#SBATCH --job-name=kd
#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yimingz3/src/allie/results/recipe10x/distill-v2/logs/%x-%j.out
# sbatch --job-name=RUN kd.sbatch: fine-tune a 1e18 sweep MoE student from its weights (fresh optimizer) on recent
# Lichess months, optionally toward the big run's move distribution (modded_train --kd-*), then golden and
# Maia-bench scores. RUN = kd-sW-aA-tSTEPS[-TT][-gTEACHER][-eager]: student s896 / s1024 / s1152 (18x896 / 20x1024 /
# 22x1152), KD weight a0 = 0 (no teacher: the hard-label control), a05 = 0.5, a1 = 1 (teacher only), STEPS of 512 x 1024 tokens,
# KD temperature T (default 1), teacher 132k (default) / final / any dir under $G/distill-v2 (teacher-TEACHER),
# eager: --ckpt eager. Recipe as ft-blitz's p2 runs: warmup 96, peak 0.2, linear decay over the last 30% to 0.04,
# clock-input LR 1.25; data as ann.sbatch (all formats, 2024-01..2026-06 + 2026-08). Same seed and sampler stream
# for every run. CK: checkpoint every CK steps (default 256). A requeue resumes; scoring skips finished outputs.
# LOCAL=1: train in a node-local project root (/scratch) and copy only model.pt, pointers and logs to results/pretrain
# (evaluable, not resumable; a requeue on another node starts over); dei-group storage is nearly full.
set -eu
n=$SLURM_JOB_NAME
R=/home/yimingz3/src/allie
S=$R/results/recipe10x/distill-v2
SRC=$S/${SOURCE:-source} # SOURCE: frozen trainer copy: source = main bf035b7, source2 = 8aa5a2d (review fixes), source3 = 591a1cf (+ FMO), source4 = 9541bbe (+ moe_keep)
P=$R/results/pretrain
G=/data/group_data/dei-group/yimingz3/allie
SCORES=$G/distill-v2/scores
T=$P
if [[ ${LOCAL:-} ]]; then
	T=/scratch/yimingz3/allie/distill-root/results/pretrain
	mkdir -p $T
	ln -sfn $R/results/original-data.json $T/../original-data.json
fi
export ALLIE_PROJECT_ROOT=${T%/results/pretrain} OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_MODULE_LOADING=LAZY DISABLE_FP8=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR=/scratch/yimingz3/allie/distill-inductor TRITON_CACHE_DIR=/scratch/yimingz3/allie/distill-triton
export PYTHONPATH=$G/envs/chessmix-overlay
end=$(($(date +%s) + 48 * 3600))
PY=$($R/.venv/bin/python $SRC/modded_runtime_stage.py | tail -1 | awk '{print $NF}')
mkdir -p $SCORES
nvidia-smi --query-gpu=name --format=csv,noheader
months=$(printf "$G/data-v1-hist/2024-%02d," {1..12})$(printf "$G/data-v1/2025-%02d," {1..12})
months+=$(printf "$G/data-v1/2026-%02d," {1..6})$G/data-v1/2026-08
[[ $months != *2026-07* ]]
[[ $n =~ ^kd-s([0-9]+)-a([0-9]+)-t([0-9]+)(-T([0-9.]+))?(-g([a-z0-9]+))?(-eager)?$ ]]
width=${BASH_REMATCH[1]} a=${BASH_REMATCH[2]} steps=${BASH_REMATCH[3]} temp=${BASH_REMATCH[5]:-1}
teacher=${BASH_REMATCH[7]:-132k} eager=${BASH_REMATCH[8]}
case $a in 0 | 1) alpha=$a ;; 0?*) alpha=0.${a:1} ;; *) exit 1 ;; esac
case $width in
896) BASE=$P/sw-1e18-s16-18x896-c8s200f0v4-s42 ;;
1024) BASE=$P/sw-1e18-s16-20x1024-c8s200f0v4-s42 ;;
1152) BASE=$P/sw-1e18-s16-22x1152-c8s200f0v4-s42 ;;
*) exit 1 ;;
esac
case $teacher in
132k) TEACHER=$G/distill-v2/teacher-132096 ;;
*) TEACHER=$G/distill-v2/teacher-$teacher ;;
esac
sched='{"batch_rows": 512, "decay_shape": "linear", "final_lr": 0.04, "mtp_steps": 0, "plateau": 0.2, "split_step": 999999, "warmup_steps": 96}'
args=(
	--name $n --width $(jq -r .args.width $BASE/config.json) --layers $(jq -r .args.layers $BASE/config.json) --head-dim 64
	--steps $steps --initial-batch-rows 512 --micro-batch 1 --row-tokens 16384 --lr-scale 1.0 --seed 42
	--eval-every 10000000 --checkpoint-every ${CK:-256} --keep-checkpoints 2 --val-rows 1024 --deterministic
	--data $G/lichess_tokens_v2 --mix table:c8s200f0v4-fcfbf8858a28 --mix-pool-frac 1.0 --mix-stores $G/data-v1,$G/data-v1-hist
	--mix-months $months --mix-history $R/results/recipe10x/bigfix-24x1536d75m4shipv2nf-c8s200f0v4/history-counts.json
	--clock-feats 3 --input-lr-mul 1.25 --aux-time 0.2 --aux-wdl 0.2 --arch "$(jq -r .args.arch $BASE/config.json)"
	--wsd-schedule "$sched" --wsd-end-step $steps --wsd-decay-start $((steps * 7 / 10)) --init-from $BASE/last.pt
)
[[ $alpha == 0 ]] || args+=(--kd-teacher $TEACHER --kd-alpha $alpha --kd-temp $temp)
[[ $eager ]] && args+=(--ckpt eager)
status() { jq -r .stop_reason $1/$n/done.json 2>/dev/null || true; }
if [[ $(status $P) != steps ]]; then
	[[ -e $T/$n/last.pt ]] && args+=(--resume $T/$n/last.pt)
	date -Is
	$PY -m torch.distributed.run --standalone --nproc_per_node=1 $SRC/modded_train.py "${args[@]}" \
		--max-seconds $((end - $(date +%s) - 2400)) >> $S/logs/$n.train.log 2>&1
	case $(status $T) in
	steps) ;;
	wall_clock_cap) scontrol requeue $SLURM_JOB_ID && exit 0 ;;
	*) exit 1 ;;
	esac
	if [[ $T != "$P" ]]; then
		mkdir -p $P/$n
		$S/keep_model.sh $T/$n $P/$n
	fi
fi
date -Is
CK_PTR=$P/$n/final.pt # a --final-model-only run's model; else last.pt
[[ -e $CK_PTR ]] || CK_PTR=$P/$n/last.pt
[[ -e $R/results/lm-eval/$n/strat-v1.json ]] ||
	$PY -m torch.distributed.run --standalone --nproc_per_node=1 $R/scripts/eval_strat.py \
		--checkpoint $CK_PTR --source $SRC --split strat --batch 16 >> $S/logs/$n.strat.log 2>&1
[[ -e $SCORES/$n.json ]] ||
	$PY -m torch.distributed.run --standalone --nproc_per_node=1 $R/results/recipe10x/maia3-bench/score_moe.py \
		--checkpoint $CK_PTR --source $SRC --batch 16 --out $SCORES/$n.npz >> $S/logs/$n.maia.log 2>&1
date -Is
