#!/bin/bash
# dei_governor.sh: our dei-group GPUs may fill the partition unless another user's dei job waits for resources;
# then ours are capped at CAP: our pending dei GPU jobs are held, and after GRACE s of waiting our newest running jobs
# that opted in (sbatch --comment=requeue-ok, or requeue-ok as one ;-separated token: they resume from their
# checkpoints) are requeued held until we hold <= CAP. Everything it held (STATE/held, one line per job or array task)
# is released once nobody waits. Jobs in PROTECT (the controller) are never touched.
# To hold one of our dei GPU jobs yourself, always run dei_governor.sh --forget ID... (a job, an array: all its tasks,
# or JOBID_TASK), never scontrol hold/release: it holds ID under the pass lock and the governor never releases it.
# DRY=1: one pass (or --forget), printing actions. One governor per STATE; it logs to stdout.
set -u
cap=${CAP:-8} grace=${GRACE:-600} every=${EVERY:-120} dry=${DRY:-0} me=$USER
state=${STATE:-/data/group_data/dei-group/yimingz3/allie/controller/dei-governor}
protect=${PROTECT:-/data/group_data/dei-group/yimingz3/allie/controller/job.id}
mkdir -p "$state"
touch "$state/held"
exec 8> "$state/held.lock"
log() { echo "$(date -Is) $*"; }
unheld() {
	awk -v ids="$*" 'BEGIN { split(ids, a, " "); for (i in a) d[a[i]] } !($0 in d)' "$state/held" > "$state/held.tmp" &&
		mv "$state/held.tmp" "$state/held"
}
kept() { local j; for j; do grep -qxF -e "$j" -e "${j%%_*}" <<< "$keep" && return; done; return 1; }
# own ID CMD...: record ID in STATE/held, then run CMD ID; the record outlives a failed CMD until release drops it
own() {
	local j=$1
	shift
	((dry)) && { log "DRY $* $j"; return; }
	echo "$j" >> "$state/held" || return
	if "$@" "$j"; then log "$* $j"; else log "FAILED $* $j"; return 1; fi
}
release() {
	local j l failed=()
	while read -r j; do
		if [[ -z $j ]]; then continue
		elif kept "$j"; then log "protected $j: kept, not released"
		elif ((dry)); then log "DRY scontrol release $j"
		elif scontrol release "$j"; then log "release $j"; unheld "$j"
		else failed+=("$j")
		fi
	done < <(sort -u "$state/held")
	((${#failed[@]})) || return 0
	if ! l=$(squeue -h -r -u "$me" -t PD -O 'JobArrayID:0|,PriorityLong:0'); then
		log "release failed, squeue failed: keeping ${failed[*]}"
		return
	fi
	for j in "${failed[@]}"; do
		if grep -qxF "$j|0" <<< "$l"; then
			log "release $j failed: still held, keeping"
		else
			log "release $j failed: no longer held, dropping"
			unheld "$j"
		fi
	done
}
if [[ ${1:-} == --forget ]]; then
	shift
	flock 8 && keep=$(< "$protect") || exit 1
	rc=0
	for j; do
		if ! [[ $j =~ ^[0-9]+(_[0-9]+)?$ ]]; then echo "--forget $j: need JOBID or JOBID_TASK" >&2; rc=1
		elif ! ids=$(squeue -h -r -j "$j" -o %i) || [[ -z $ids ]]; then echo "--forget $j: not in squeue" >&2; rc=1
		elif kept "$j" $ids; then echo "--forget $j: protected" >&2; rc=1
		elif ((dry)); then log "DRY scontrol hold $j; forget" $ids
		elif ! scontrol hold "$j"; then rc=1
		elif unheld $ids; then log "forget $j:" $ids "held, now the caller's" | tee -a "$state/log"
		else echo "--forget $j: held, but $state/held was not updated: the governor still owns" $ids >&2; rc=1
		fi
	done
	exit $rc
fi
if ! ((dry)); then
	exec 9> "$state/lock"
	flock -n 9 || { echo "dei_governor already running on $state" >&2; exit 1; }
fi
since=0
while :; do
	if ! flock 8; then
		log "flock failed; skipping"
	elif ! keep=$(< "$protect"); then
		log "cannot read $protect; skipping"
	elif ! pd=$(squeue -h -r -p dei-group -t PD -O 'UserName:0|,JobArrayID:0|,Reason:0|,PriorityLong:0|,tres-alloc:0') ||
		! run=$(squeue -h -p dei-group -u "$me" -t R -O 'JobArrayID:0|,StartTime:0|,tres-alloc:0|,Comment:0'); then
		log "squeue failed; skipping"
	elif awk -F'|' -v me="$me" '$1 != me && $3 ~ /^(Resources|Priority)$/ { w = 1 } END { exit !w }' <<< "$pd"; then
		((since)) || since=$(date +%s)
		while IFS='|' read -r u j r prio tres; do
			[[ $u == "$me" && $prio != 0 && ,$tres, =~ ,gres/gpu=[1-9] ]] && ! kept "$j" && own "$j" scontrol hold
		done <<< "$pd"
		if (($(date +%s) - since >= grace)); then
			total=0 jobs=()
			while IFS='|' read -r j start tres comment; do
				[[ ,$tres, =~ ,gres/gpu=([1-9][0-9]*), ]] || continue
				n=${BASH_REMATCH[1]}
				total=$((total + n))
				[[ ";$comment;" == *";requeue-ok;"* ]] && ! kept "$j" && jobs+=("$j:$n")
			done < <(sort -t'|' -k2 -r <<< "$run")
			for jn in "${jobs[@]}"; do
				((total > cap)) || break
				own "${jn%:*}" scontrol requeuehold Incomplete && total=$((total - ${jn##*:}))
			done
			((total > cap)) && log "still $total dei GPUs > $cap: no more opted-in jobs to requeue"
		fi
	else
		since=0
		release
	fi
	flock -u 8
	((dry)) && break
	sleep "$every" 9>&-
done
