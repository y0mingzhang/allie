#!/bin/bash
# launch.sh LABEL [orchard|babel]: freeze sweep-{1e17,3e17,1e18}-LABEL (freeze.py) and submit every run, one race per run:
# 1e17 on 1 x A6000 (dei-group requeue-ok, raced with preempt A6000), 3e17 on 4 x L40S (general + preempt), 1e18 on orchard
# 4 x H100 (omake.py + orun.sh, default) or on 4 x L40S raced like 3e17 (babel). Never resubmits a frozen study.
set -euo pipefail
label=$1 where=${2:-orchard}
A=/home/yimingz3/src/allie R=$A/results/recipe10x
cd $A
.venv/bin/python $R/moe-knobs/sweep-plan/freeze.py $label
runs() { python3 -c "import json; print(len(json.load(open('$1/plan.json'))['runs']))"; }
race() {  # race BUDGET STAGGER OPTS... : one race per run of sweep-BUDGET-LABEL, run i eligible STAGGER x i minutes from now
	local b=$1 m=$2 s=$R/sweep-$1-$label i o
	shift 2
	for ((i = 0; i < $(runs $s); i++)); do
		o=()
		for x in "$@"; do o+=("$x --array=$i --begin=now+$((m * i))minutes"); done
		scripts/race_submit.sh sw$b-$i-$label-a $s/run.sbatch "${o[@]}"
	done
}
# staggered starts (NFS: every run opens >= ~220 GB of shards); scontrol hold what has not started if sampler_wait climbs
L40S=("-A dippolit -p general --qos=normal --gres=gpu:L40S:4" "-A dippolit -p preempt --qos=preempt_qos --gres=gpu:L40S:4")
[ $where = orchard ] || race 1e18 ${STAGGER:-10} "${L40S[@]}"
race 3e17 ${STAGGER:-10} "${L40S[@]}"
# 1e17 on the two dei A6000 nodes only (1 TB RAM each: the runs' nested pool slices share the page cache)
race 1e17 ${STAGGER:-10} "-A dippolit -p dei-group --qos=dei_group_qos --gres=gpu:A6000:1 --comment=requeue-ok"
if [ $where = orchard ]; then
	.venv/bin/python $R/orchard/omake.py sweep-1e18-$label --gpus 4
	ssh orchard "cd ~/allie/src && sbatch --parsable --array=0-$(($(runs $R/sweep-1e18-$label) - 1))%8 -J sw1e18 -o ~/allie/src/logs/%x-%A_%a.out orun.sh sweep-1e18-$label"
	nohup $R/moe-knobs/sweep-plan/autoscore.sh sweep-1e18-$label > /dev/null 2>&1 &
	echo "autoscore pid $!"
fi
