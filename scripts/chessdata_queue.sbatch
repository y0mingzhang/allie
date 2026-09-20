#!/bin/bash
#SBATCH --job-name=data-v1-queue
#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/group_data/dei-group/yimingz3/allie/data-v1/%x-%j.log

# Download, verify, build and clean up Lichess months, newest first. Every step is idempotent.
set -uo pipefail
ROOT=${STORE:-/data/group_data/dei-group/yimingz3/allie/data-v1}
BASE=https://database.lichess.org/standard
SCRIPTS=${SCRIPTS:-/home/yimingz3/src/allie/scripts}  # frozen checkout for long backfills
BUILD=$SCRIPTS/chessdata.sbatch
SOURCE=${SOURCE:-lichess}
MIN_FREE_G=${MIN_FREE_G:-400}  # no new download below this much free space
MAX_RAW=${MAX_RAW:-4}  # raw months awaiting a build
MONTHS=${MONTHS:-"2026-07 2026-06 2026-05 2026-04 2026-03 2026-02 2026-01
2025-12 2025-11 2025-10 2025-09 2025-08 2025-07 2025-06 2025-05 2025-04 2025-03 2025-02 2025-01"}
ALL=${ALL:-"2026-08 $MONTHS"}
# published game counts gate the raw cleanup; keep a copy in the store in case lichess.org throttles
for f in sha256sums counts; do
	curl -sS --fail --retry 5 --max-time 300 $BASE/$f.txt >"$ROOT/$f.txt.new" && mv "$ROOT/$f.txt.new" "$ROOT/$f.txt"
	rm -f "$ROOT/$f.txt.new"
done
SUMS=$(cat "$ROOT/sha256sums.txt" 2>/dev/null)
COUNTS=$(cat "$ROOT/counts.txt" 2>/dev/null)
log() { echo "$(date -Is) $*"; }

fetch() {
	local m=$1 D=$ROOT/$1 F=$ROOT/$1/lichess_db_standard_rated_$1.pgn.zst
	local URL=$BASE/lichess_db_standard_rated_$m.pgn.zst total step
	mkdir -p "$D"
	[ -f "$D/download-verified" ] && return 0
	if [ "$SOURCE" = hf ]; then
		/home/yimingz3/src/allie/.venv/bin/python "$SCRIPTS/hf_fetch.py" "$m" "$D/hf" &&
			touch "$D/download-verified" && log "$m verified (hf)"
		return
	fi
	total=$(curl -sSI --retry 5 "$URL" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r')
	[ -n "$total" ] || return 1
	step=$(((total + 3) / 4))
	if [ "$(stat -c %s "$F.tmp" 2>/dev/null || echo 0)" -ne "$total" ]; then
		for i in 0 1 2 3; do
			(
				a=$((i * step)) b=$((i * step + step - 1))
				[ $b -ge $((total - 1)) ] && b=$((total - 1))
				want=$((b - a + 1)) P=$D/range.$i
				for _ in $(seq 1 100); do
					have=$(stat -c %s "$P" 2>/dev/null || echo 0)
					[ "$have" -eq "$want" ] && exit 0
					[ "$have" -gt "$want" ] && { rm -f "$P"; have=0; }
					curl -sS --fail --max-time 7200 -r "$((a + have))-$b" "$URL" >>"$P" || sleep $((30 + RANDOM % 60))
				done
				exit 1
			) &
		done
		wait
		for i in 0 1 2 3; do
			a=$((i * step)) b=$((i * step + step - 1))
			[ $b -ge $((total - 1)) ] && b=$((total - 1))
			[ "$(stat -c %s "$D/range.$i" 2>/dev/null || echo 0)" -eq $((b - a + 1)) ] || return 1
		done
		cat "$D"/range.{0,1,2,3} >"$F.tmp" && rm -f "$D"/range.*
	fi
	want=$(awk -v f="lichess_db_standard_rated_$m.pgn.zst" '$2==f {print $1}' <<<"$SUMS")
	if [ -n "$want" ] && [ "$(sha256sum "$F.tmp" | cut -d' ' -f1)" = "$want" ]; then
		mv "$F.tmp" "$F" && touch "$D/download-verified" && log "$m verified"
	else
		log "$m checksum mismatch; discarding"
		rm -f "$F.tmp"
		return 1
	fi
}

submit() {
	local D=$ROOT/$1 job state
	[ -f "$D/stats.json" ] || [ ! -f "$D/download-verified" ] && return 0
	job=$(cat "$D/build-job" 2>/dev/null)
	state=$([ -n "$job" ] && sacct -n -X -j "$job" -o State 2>/dev/null | head -1 | tr -d ' ')
	case "$state" in PENDING | RUNNING | REQUEUED | COMPLETED) return 0 ;; esac
	local where="-p dei-group -q dei_group_qos"
	# PREEMPT_BUILDS=1: one build on dei-group at a time, the rest on preempt CPUs (restartable per chunk)
	if [ -n "${PREEMPT_BUILDS:-}" ] && squeue -h -u "$USER" -p dei-group -o %j | grep -q '^data-v1-build-'; then
		where="-p preempt -q preempt_cpu_qos"
	fi
	# resources by the month's published game count (finalize holds the month in memory: ~3-5 GB per M games)
	local n res
	n=$(awk -v f="lichess_db_standard_rated_$1.pgn.zst" '$1==f {print $2}' <<<"$COUNTS")
	if [ -n "${BUILD_RES:-}" ]; then res=$BUILD_RES
	elif [ "${n:-0}" -lt 12000000 ]; then res="--cpus-per-task=16 --mem=96G --time=06:00:00"
	elif [ "${n:-0}" -lt 40000000 ]; then res="--cpus-per-task=32 --mem=200G --time=12:00:00"
	else res="--cpus-per-task=32 --mem=300G --time=12:00:00"
	fi
	sbatch --parsable $where $res --export=ALL,STORE="$ROOT",SCRIPTS="$SCRIPTS" --job-name="data-v1-build-$1" \
		--output="$ROOT/%x-%j.log" "$BUILD" "$1" >"$D/build-job" &&
		log "$1 build submitted $(cat "$D/build-job")"
}

cleanup() {
	local m D want got
	for m in $ALL; do
		D=$ROOT/$m
		[ -f "$D/stats.json" ] && [ ! -f "$D/cleaned" ] || continue
		want=$(awk -v f="lichess_db_standard_rated_$m.pgn.zst" '$1==f {print $2}' <<<"$COUNTS")
		got=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['games'])" "$D/stats.json")
		if [ -n "$want" ] && [ $((got * 1000)) -ge $((want * 998)) ]; then
			rm -rf "$D/parts" "$D"/*.pgn.zst "$D/hf" && touch "$D/cleaned" && log "$m cleaned ($got/$want games)"
		else
			log "$m kept raw files: $got games vs $want published"
		fi
	done
}

for m in $MONTHS; do
	cleanup
	while [ "$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)" -lt "$MIN_FREE_G" ]; do
		log "under ${MIN_FREE_G}G free; waiting"
		sleep 900
	done
	# raw months awaiting a build are large (~73 GB each from Hugging Face): keep at most MAX_RAW
	while [ "$(for x in $ALL; do [ -f "$ROOT/$x/download-verified" ] && [ ! -f "$ROOT/$x/cleaned" ] && echo "$x"; done | wc -l)" -ge "$MAX_RAW" ]; do
		log "$MAX_RAW months await builds; waiting"
		sleep 900
		cleanup
		for x in $ALL; do submit "$x"; done
	done
	until fetch "$m"; do
		log "$m fetch failed; retrying in 10 minutes"
		sleep 600
	done
	submit "$m"
done
while :; do
	cleanup
	for m in $ALL; do submit "$m"; done
	left=$(for m in $ALL; do [ -f "$ROOT/$m/cleaned" ] || echo "$m"; done)
	[ -z "$left" ] && break
	log "waiting on builds: $(echo $left)"
	sleep 900
done
log QUEUE_DONE
