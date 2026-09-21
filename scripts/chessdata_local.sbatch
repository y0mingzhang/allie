#!/bin/bash
#SBATCH --job-name=data-v1-build
#SBATCH --account=dippolit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append

# Usage: sbatch chessdata_local.sbatch 2022-04 (STORE, SCRIPTS set by the queue). Download, build and finalize one
# month on node-local /scratch, then hand the finished store to chessdata_publish.sbatch, which copies it to the
# shared STORE one month at a time. NFS sees only that final sequential copy. FASTBUILD=1 builds with fastbuild.py
# (the same store byte for byte from a C core, in bounded memory; lower requests in chessdata_localqueue.sbatch).
set -euo pipefail
month=${1:?month, e.g. 2022-04}
S=${SCRIPTS:?} D=${STORE:?}/$month L=/scratch/$USER/chessdata/$month
PY=/home/yimingz3/src/allie/.venv/bin/python B=$([ "${FASTBUILD:-0}" = 1 ] && echo fastbuild.py || echo chessdata.py)
[ -f "$D/stats.json" ] && exit 0
trap 'rm -rf "$L"' TERM  # preempted: a requeue may land elsewhere; free this node's scratch
mkdir -p "$L"
free=$(df --output=avail -BG /scratch | tail -1 | tr -dc 0-9)
[ "$free" -ge 150 ] || { echo "$(date -Is) $(hostname): only ${free}G on /scratch"; exit 1; }
[ -f "$L/download-verified" ] || { $PY "$S/hf_fetch.py" "$month" "$L/hf" && touch "$L/download-verified"; }
[ -f "$L/build-complete.json" ] || $PY "$S/$B" build --hf "$L/hf" --out "$L"
[ -f "$L/stats.json" ] || $PY "$S/$B" finalize --out "$L"
rm -rf "$L/hf" "$L/parts" "$L/spill"
part=$SLURM_JOB_PARTITION qos=$([ "$part" = dei-group ] && echo dei_group_qos || echo preempt_cpu_qos)
sbatch --parsable -p "$part" -q "$qos" -w "$(hostname -s)" --job-name=data-v1-hist2-publish \
	--dependency="singleton${PUBLISH_AFTER:+,afterany:$PUBLISH_AFTER}" --cpus-per-task=2 --mem=4G --time=04:00:00 \
	--requeue --output="$STORE/%x-%j.log" --export=ALL,STORE="$STORE" "$S/chessdata_publish.sbatch" "$month" >"$D.publish-job"
echo "$(date -Is) $month finalized on $(hostname -s); publish job $(cat "$D.publish-job")"
