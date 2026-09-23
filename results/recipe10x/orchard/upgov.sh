#!/bin/bash
# upgov.sh: run the f <= 0.19 subset upload (oupload.sh SUBSET, orchard) once the data confirm's control arm is done and
# $GO exists; pause it (scancel, oupload is idempotent) while the winner arm's last 3 train rows all have sampler_wait > 0.1,
# resubmit after 10 min once they are all < 0.05; after the winner finishes, run it unthrottled. Log: logs/upgov.log.
cd /home/yimingz3/src/allie/results/recipe10x/orchard
L=logs/upgov.log GO=logs/upgov.go P=/home/yimingz3/src/allie/results/pretrain
CTL=$P/dc3-3e17-control-v4-s42 WIN=$P/dc3-3e17-c8s200f0v4-v4-s42
RELS=$(cd subset-f019 && find . -name '*.list' | sed 's|^\./||; s|\.list$||' | sort | tr '\n' ' ')
log() { echo "$(date -Is) $*" >> $L; }
waits() { tail -3 $WIN/train.jsonl | python3 -c "import json,sys; print(' '.join(str(json.loads(l)['sampler_wait']) for l in sys.stdin))"; }
all() { for w in $(waits); do python3 -c "import sys; sys.exit(0 if $w $1 else 1)" || return 1; done; }
left() { timeout 180 ssh -o BatchMode=yes orchard 'cd ~/allie/src && python3 osubcheck.py subset-f019 v1-f0.19' | grep -o "^[0-9]*"; }  # months complete (osubcheck.py)
submit() {
	job=$(timeout 120 ssh -o BatchMode=yes orchard "cd ~/allie/src && sbatch --parsable -J oupload-sub --export=ALL,SUBSET=\$HOME/allie/src/subset-f019,SUBSET_F=0.19,STREAMS=$1,PIN=\$HOME/allie/src/v4-data-pin.json oupload.sh $RELS")
	log "submitted $job streams $1 ($(left) of 50 done)"
}
running() { timeout 60 ssh -o BatchMode=yes orchard "squeue -h -j $job -o %T" 2>/dev/null | grep -q .; }
log "start: waiting for $CTL/done.json and $GO"
until [ -f $CTL/done.json ] && [ -f $GO ]; do sleep 60; done
job= paused=0 t=0
submit 3
while true; do
	sleep 60
	if [ -f $WIN/done.json ]; then
		running || { [ "$(left)" -ge 50 ] && { log "all 50 subset months on GCS"; exit 0; }; submit 6; }
		sleep 240; continue
	fi
	if [ $paused = 0 ] && all "> 0.1"; then
		timeout 60 ssh -o BatchMode=yes orchard "scancel $job"; paused=1 t=$(date +%s); log "paused: winner sampler_wait $(waits)"
	elif [ $paused = 1 ] && [ $(($(date +%s) - t)) -ge 600 ] && all "< 0.05"; then
		paused=0; submit 3
	elif [ $paused = 0 ] && ! running; then
		[ "$(left)" -ge 50 ] && { log "all 50 subset months on GCS"; exit 0; }
		log "job $job ended with $(left) of 50 done"; submit 3
	fi
done
