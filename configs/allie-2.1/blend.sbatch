#!/bin/bash
#SBATCH --job-name=r21-blend
#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yimingz3/src/allie/results/recipe10x/bigrun-v4/release/logs/%x-%j.log
# blend.sbatch RUN D NAME: Allie 2.1's release weights D/NAME = 0.5 final + 0.5 tail EMA (horizon 0.08, every 64), from
# RUN/last.pt and RUN/tail-ema-0.08.pt (allie.experiments.tail_ema blend; refuses an EMA of another step or a
# restarted one). Prints the readout record (sha256 of both inputs); a finished blend is kept.
set -euo pipefail
RUN=$1 D=$2 NAME=$3
CODE=${CODE:-/home/yimingz3/src/allie-wt-infer21}
date -Is; hostname
mkdir -p $D
[ -e $D/$NAME/last.pt ] || {
	PYTHONPATH=$CODE/src /home/yimingz3/src/allie/.venv/bin/python -m allie.experiments.tail_ema blend $RUN/last.pt \
		$RUN/tail-ema-0.08.pt 0.5 $D/$NAME > $D/$NAME.readout.tmp
	mv $D/$NAME.readout.tmp $D/$NAME.readout.json
}
cat $D/$NAME.readout.json
date -Is
