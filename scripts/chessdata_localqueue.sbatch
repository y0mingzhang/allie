#!/bin/bash
#SBATCH --job-name=data-v1-localqueue
#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

# Node-local month builds (chessdata_local.sbatch: HF download, build and finalize on /scratch; one sequential
# publish copy per month to STORE). Newest first, at most MAX_BUILDS months in flight, at most one on dei-group.
set -uo pipefail
ROOT=${STORE:?} S=${SCRIPTS:?} MONTHS=${MONTHS:?} MAX_BUILDS=${MAX_BUILDS:-3}
J=$ROOT/.jobs
mkdir -p "$J"
COUNTS=$(cat "$ROOT/counts.txt")
log() { echo "$(date -Is) $*"; }
state() { [ -n "$1" ] && sacct -n -X -j "$1" -o State 2>/dev/null | head -1 | tr -d ' +'; }
active() { case "$(state "$1")" in PENDING | RUNNING | REQUEUED | CONFIGURING | COMPLETING) return 0 ;; esac; return 1; }

cleanup() {  # months built by the old NFS path: drop their raw once the game count matches
	local m D want got
	for m in $MONTHS; do
		D=$ROOT/$m
		[ -f "$D/stats.json" ] && [ ! -f "$D/cleaned" ] || continue
		want=$(awk -v f="lichess_db_standard_rated_$m.pgn.zst" '$1==f {print $2}' <<<"$COUNTS")
		got=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['games'])" "$D/stats.json")
		if [ -n "$want" ] && [ $((got * 1000)) -ge $((want * 998)) ]; then
			ionice -c3 rm -rf "$D/parts" "$D/hf" && touch "$D/cleaned" && log "$m cleaned ($got/$want games)"
		fi
	done
}

inflight() {
	local m n=0
	for m in $MONTHS; do
		[ -f "$ROOT/$m/stats.json" ] && continue
		{ active "$(cat "$J/$m.build" 2>/dev/null)" || active "$(cat "$ROOT/$m.publish-job" 2>/dev/null)"; } && n=$((n + 1))
	done
	echo $n
}

submit() {  # first attempt, or retry a failed build / publish
	local m=$1 b p node n res where tries
	[ -f "$ROOT/$m/stats.json" ] && return 0
	b=$(cat "$J/$m.build" 2>/dev/null) p=$(cat "$ROOT/$m.publish-job" 2>/dev/null)
	active "$b" || active "$p" && return 0
	if [ "$(state "$b")" = COMPLETED ] && [ -n "$p" ]; then  # finalized; the publish copy failed: retry on that node
		node=$(sacct -n -X -j "$b" -o NodeList 2>/dev/null | head -1 | tr -d ' ')
		sbatch --parsable -p "$(sacct -n -X -j "$b" -o Partition | head -1 | tr -d ' ')" \
			-q "$(sacct -n -X -j "$b" -o QOS | head -1 | tr -d ' ')" -w "$node" --job-name=data-v1-hist2-publish \
			--dependency=singleton --cpus-per-task=2 --mem=4G --time=04:00:00 --requeue --output="$ROOT/%x-%j.log" \
			--export=ALL,STORE="$ROOT" "$S/chessdata_publish.sbatch" "$m" >"$ROOT/$m.publish-job" && log "$m publish retried on $node"
		return 0
	fi
	tries=$(cat "$J/$m.tries" 2>/dev/null || echo 0)
	[ "$tries" -ge 4 ] && { log "$m failed $tries times; skipping"; return 0; }
	echo $((tries + 1)) >"$J/$m.tries"
	n=$(awk -v f="lichess_db_standard_rated_$m.pgn.zst" '$1==f {print $2}' <<<"$COUNTS")
	if [ "${n:-0}" -lt 12000000 ]; then res="--cpus-per-task=16 --mem=96G --time=08:00:00"
	elif [ "${n:-0}" -lt 40000000 ]; then res="--cpus-per-task=32 --mem=200G --time=12:00:00"
	else res="--cpus-per-task=32 --mem=300G --time=16:00:00"
	fi
	where="-p preempt -q preempt_cpu_qos"
	squeue -h -u "$USER" -p dei-group -o %j | grep -q '^data-v1-build-' || where="-p dei-group -q dei_group_qos"
	sbatch --parsable $where $res --job-name="data-v1-build-$m" --output="$ROOT/%x-%j.log" \
		--export=ALL,STORE="$ROOT",SCRIPTS="$S",PUBLISH_AFTER="${PUBLISH_AFTER:-}" "$S/chessdata_local.sbatch" "$m" >"$J/$m.build" &&
		log "$m build submitted $(cat "$J/$m.build") ($where, try $((tries + 1)))"
}

for m in $MONTHS; do
	[ -f "$ROOT/$m/stats.json" ] && continue
	while [ "$(inflight)" -ge "$MAX_BUILDS" ]; do
		sleep 600
		cleanup
	done
	submit "$m"
	sleep 5
done
while :; do
	cleanup
	left=$(for m in $MONTHS; do [ -f "$ROOT/$m/stats.json" ] || echo "$m"; done)
	[ -z "$left" ] && break
	for m in $left; do submit "$m"; done
	log "waiting on: $(echo $left)"
	sleep 900
done
log QUEUE_DONE
