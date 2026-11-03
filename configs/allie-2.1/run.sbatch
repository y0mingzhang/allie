#!/bin/bash
#SBATCH --job-name=v4
#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:8
#SBATCH --exclude=babel-t5-28,babel-x9-32,babel-q9-32,babel-p9-32,babel-l9-32,babel-n9-24,babel-o5-28,babel-q5-32,babel-p5-16,babel-n5-32
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-0%1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yimingz3/src/allie/results/recipe10x/bigrun-v4-24x2048d112-c4x-lr0p35-c8s200f0v4/logs/%x-%A_%a.out

export TORCH_NCCL_HIGH_PRIORITY=1  # as the paused v3 run's run.sbatch (part of its speed set)
# once, before the first leg (~40 min, bigrun-v3/perf/perfbench.py, node scratch; rows in bigrun-v3/perf/bench.jsonl, trainer
# logs in bigrun-v3/perf/prof/ARM-JOB.log; informational, the leg starts either way):
#   gate22       the paused v3 run's leg-0 config (study ...d112f) on this study's source, steps 1-50: train CE at 25 / 50 must equal
#                v3's bitwise (7.1451999643772925 / 6.226525646816707), i.e. no code drift between v3's and v4's trainer
#   tail22       this run's config, 50-step schedule, tail EMA opened at once (0.34, every 4): moe_shard update + write
#   resume-tail22  as tail22: 0 -> 24 vs 0 -> 12, --resume -> 24 (saves every 12), step-24
#                model.pt and every rank file (incl. the EMA state) bitwise
S=/home/yimingz3/src/allie/results/recipe10x/bigrun-v4-24x2048d112-c4x-lr0p35-c8s200f0v4
if mkdir "$S/logs/gate" 2>/dev/null; then
	PY=/home/yimingz3/src/allie/.venv/bin/python PB=/home/yimingz3/src/allie/results/recipe10x/bigrun-v3/perf/perfbench.py
	PERF_STUDY=bigrun-v3-24x2048d112f-c8s200f0v4 timeout -k 60 1800 $PY $PB "$S/source-ours" gate22 > "$S/logs/gate/out.log" 2>&1
	echo "$(date -Is) gate22 exit $?" >> "$S/logs/gate/out.log"
	PERF_STUDY=bigrun-v4-24x2048d112-c4x-lr0p35-c8s200f0v4 timeout -k 60 3600 $PY $PB "$S/source-ours" \
		"tail22:--steps=50,--wsd-end-step=50,--wsd-decay-start=-1,--tail-ema=0.34,--tail-ema-every=4" \
		"resume-tail22:--steps=50,--wsd-end-step=50,--wsd-decay-start=-1,--tail-ema=0.34,--tail-ema-every=4" >> "$S/logs/gate/out.log" 2>&1
	echo "$(date -Is) tail22 / resume-tail22 exit $?" >> "$S/logs/gate/out.log"
	$PY /home/yimingz3/src/allie/results/recipe10x/bigrun-v4/gate_verdict.py "$SLURM_JOB_ID" > "$S/logs/gate/verdict" 2>&1
	rm -rf "/scratch/yimingz3/allie/perfbench-$SLURM_JOB_ID"  # perfbench leaves it if timed out
	for _ in $(seq 60); do [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] && break; sleep 5; done
fi
exec /home/yimingz3/src/allie/.venv/bin/python /home/yimingz3/src/allie/results/recipe10x/bigrun-v4-24x2048d112-c4x-lr0p35-c8s200f0v4/modelexp.py task /home/yimingz3/src/allie/results/recipe10x/bigrun-v4-24x2048d112-c4x-lr0p35-c8s200f0v4/round.py v4-24x2048d112
