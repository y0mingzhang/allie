#!/bin/bash
#SBATCH --job-name=ann
#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:8
#SBATCH --exclude=babel-q9-32,babel-x9-32,babel-t5-28
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yimingz3/src/allie/results/recipe10x/distill-v2/logs/%x-%j.out
# RUNS="RUN..." sbatch ann.sbatch: back to back on one 8-GPU allocation, each RUN a second anneal of the big run's final checkpoint (fresh optimizer, modded_train
# --init-from) on recent Lichess months, its own arch, batch, rows and (ckpt eager) memory plan, then golden and
# Maia-bench scores. RUN = ann-{all,bx2,old}-pD-tSTEPS[-kK][-aA]: all = the big run's table, bx2 = it with Lichess blitz x2
# (recipes/), old = the table on 2019-01..2022-12 only (data-v1-hist2); kK: arch moe_keep K (route through the best K of the top-16 experts); aA: KD weight 0.A toward the final
# big run (all 16 experts, TEACHER, default $G/distill-v2/teacher-final); INIT: the checkpoint to start from (default the final big
# run's, $G/distill-v2/bigrun-final); peak LR 0.D (warmup 96, constant, linear decay over the last 30% to 0.004, the big run's final);
# STEPS of 524,288 tokens. Months 2024-01..2026-06 + 2026-08 (never 2026-07). CK: checkpoint every CK steps
# (default: only at the end). Each run trains in a node-local project root (/scratch) and only its model.pt,
# pointers and logs are copied to results/pretrain (the 44 GB optimizer state stays on scratch, a cache: the
# copied checkpoint is evaluable, not resumable; dei-group storage is nearly full). A requeue on the same node
# resumes. Each finished RUN gets a score.sbatch job (golden + Maia bench) on a dei A6000.
set -eu
: "${RUNS:?RUNS}"
R=/home/yimingz3/src/allie
S=$R/results/recipe10x/distill-v2
SRC=$S/${SOURCE:-source} # SOURCE: frozen trainer copy: source = main bf035b7, source2 = 8aa5a2d (review fixes), source3 = 591a1cf (+ FMO), source4 = 9541bbe (+ moe_keep)
P=$R/results/pretrain
G=/data/group_data/dei-group/yimingz3/allie
BIG=$R/results/pretrain/bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42
INIT=${INIT:-$G/distill-v2/bigrun-final}
SCORES=$G/distill-v2/scores
L=/scratch/yimingz3/allie/distill-root # node-local project root of the training runs
mkdir -p $L/results/pretrain
ln -sfn $R/results/original-data.json $L/results/original-data.json
export ALLIE_PROJECT_ROOT=$L OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_MODULE_LOADING=LAZY DISABLE_FP8=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 # as the big run (modelexp)
export TORCHINDUCTOR_CACHE_DIR=/scratch/yimingz3/allie/distill-inductor TRITON_CACHE_DIR=/scratch/yimingz3/allie/distill-triton
export PYTHONPATH=$G/envs/chessmix-overlay
end=$(($(date +%s) + 12 * 3600))
PY=$($R/.venv/bin/python $SRC/modded_runtime_stage.py | tail -1 | awk '{print $NF}')
mkdir -p $SCORES
for _ in {1..60}; do [[ -e $INIT/model.pt ]] && break; sleep 30; done # link_final.sh links it when the run ends
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
months=$(printf "$G/data-v1-hist/2024-%02d," {1..12})$(printf "$G/data-v1/2025-%02d," {1..12})
months+=$(printf "$G/data-v1/2026-%02d," {1..6})$G/data-v1/2026-08
[[ $months != *2026-07* ]]
recent=$months stores=$G/data-v1,$G/data-v1-hist
for n in $RUNS; do
	mix=table:c8s200f0v4-fcfbf8858a28 months=$recent stores=$G/data-v1,$G/data-v1-hist
	case ${n#ann-} in
	all-*) ;;
	bx2-*) mix=table:$(cd $S/recipes && ls bx2-c8s200f0v4-*.json | sed 's/\.json$//') ;;
	old-*) # old months only: 2019-01..2022-12, the same table and history counts
		months=$(printf "$G/data-v1-hist2/%s," {2019..2022}-{01..12}) months=${months%,} stores=$G/data-v1-hist2 ;;
	*) exit 1 ;;
	esac
	[[ $n =~ -p([0-9]+)-t([0-9]+)(-k([0-9]+))?(-a([0-9]+))?$ ]]
	peak=0.${BASH_REMATCH[1]} steps=${BASH_REMATCH[2]} keep=${BASH_REMATCH[4]} a=${BASH_REMATCH[6]}
	arch=$(jq -r .args.arch $BIG/resume-config.json)
	[[ $keep ]] && arch=$(jq -c ".moe_keep = $keep" <<< "$arch")
	sched='{"batch_rows": 512, "decay_shape": "linear", "final_lr": 0.004, "mtp_steps": 0, "plateau": '$peak', "split_step": 999999, "warmup_steps": 96}'
	args=(
		--name $n --width 1536 --layers 24 --head-dim 64 --steps $steps --initial-batch-rows 512 --micro-batch 4
		--row-tokens 16384 --lr-scale 1.0 --seed 42 --eval-every 10000000 --checkpoint-every ${CK:-1000000}
		--keep-checkpoints 2 --val-rows 1024 --deterministic --data $G/lichess_tokens_v2 --mix $mix --mix-pool-frac 1.0
		--mix-stores $stores --mix-months $months --mix-history $(jq -r .args.mix_history $BIG/resume-config.json)
		--ckpt eager --clock-feats 3 --input-lr-mul 1.25 --aux-time 0.2 --aux-wdl 0.2 --arch "$arch"
		--wsd-schedule "$sched" --wsd-end-step $steps --wsd-decay-start $((steps * 7 / 10)) --init-from $INIT
	)
	[[ $a ]] && args+=(--kd-teacher ${TEACHER:-$G/distill-v2/teacher-final} --kd-alpha 0.$a)
	status() { jq -r .stop_reason $P/$n/done.json 2>/dev/null || true; }
	if [[ $(status) != steps ]]; then
		[[ -e $L/results/pretrain/$n/last.pt ]] && args+=(--resume $L/results/pretrain/$n/last.pt)
		date -Is
		$PY -m torch.distributed.run --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) $SRC/modded_train.py "${args[@]}" \
			--max-seconds $((end - $(date +%s) - 2400)) >> $S/logs/$n.train.log 2>&1
		case $(jq -r .stop_reason $L/results/pretrain/$n/done.json) in
		steps) ;;
		wall_clock_cap) scontrol requeue $SLURM_JOB_ID && exit 0 ;;
		*) exit 1 ;;
		esac
		mkdir -p $P/$n
		$S/keep_model.sh $L/results/pretrain/$n $P/$n
	fi
	[[ -e $SCORES/$n.json ]] || squeue -h -n score-$n | grep -q . || sbatch --job-name=score-$n --export=ALL,RUN=$n $S/score.sbatch
done
date -Is
