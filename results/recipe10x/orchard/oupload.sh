#!/bin/bash
#SBATCH -p preempt --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G -t 8:00:00 -J oupload --requeue --open-mode=append -o /home/yimingz3/allie/src/logs/oupload-%j.out
# usage: sbatch oupload.sh <path under the babel store root> ...   e.g. data-v1/2025-01 ext-v1/otb/20xx-twic lichess_tokens_v2
# One tar per month -> gs://cmu-gpucloud-yimingz3/data/v1/<path>.tar + <path>.tar.json (bytes, sha256, file count).
# PIN=<pin json>: buckets/stats sha256 must match its "files". Idempotent (skips paths whose .json exists), so preemption/requeue just redoes unfinished months. STREAMS babel readers (default 4).
# SUBSET=<dir> SUBSET_F=<f>: tar only the files listed in <dir>/<path>.list (osubset.py: top-level files + the shards a pool_frac <= f run opens) to data/v1-f<f>/ instead of data/v1/.
cd ~/allie/src && source ostage.sh
D=gs://cmu-gpucloud-yimingz3/data/v1${SUBSET:+-f$SUBSET_F}
echo "$(date) job $SLURM_JOB_ID on $(hostname), babel via oxfer, $# paths"
one() {
  local rel=$1 f=$LOCAL/up/$1.tar sel="-type f ! -path '*/cleaned/*'"
  [ "$rel" = lichess_tokens_v2 ] && sel="-name val.npy"
  gcloud storage ls $D/$rel.tar.json >/dev/null 2>&1 && { echo "$rel: already uploaded"; return 0; }
  mkdir -p $(dirname $f); local t0=$(date +%s)
  { if [ -n "${SUBSET:-}" ]; then cp $SUBSET/$rel.list $f.list; else ox "cd $G && find $rel $sel | sort" > $f.list; fi; } && wc -l < $f.list > $f.n && ox "cd $G && tar -cf - -T -" < $f.list > $f || { echo "$rel: pull FAILED"; rm -f $f $f.n $f.list; return 1; }
  rm -f $f.list
  local want=$(head -1 $f.n) got=$(tar -tf $f | grep -vc '/$') bytes=$(stat -c %s $f) sha=$(sha256sum $f | cut -c1-64) t1=$(date +%s)
  [ "$want" = "$got" ] || { echo "$rel: file count $got != $want"; rm -f $f $f.n; return 1; }
  local hb="" hs=""
  if [ "$rel" != lichess_tokens_v2 ]; then  # pinned months: buckets.json/stats.json must match the pin
    mkdir -p $f.x && tar -xf $f -C $f.x $rel/buckets.json $rel/stats.json && hb=$(sha256sum < $f.x/$rel/buckets.json | cut -c1-64) && hs=$(sha256sum < $f.x/$rel/stats.json | cut -c1-64); rm -rf $f.x
    [ -z "${PIN:-}" ] || $PY -c "import json, sys; f = json.load(open('$PIN'))['files'].get('$G/$rel'); sys.exit(0 if f is None or (f['buckets.json'], f['stats.json']) == ('$hb', '$hs') else 1)" || { echo "$rel: buckets/stats sha differs from the pin"; rm -f $f $f.n; return 1; }
  fi
  gcloud storage cp -q $f $D/$rel.tar || { echo "$rel: upload FAILED"; return 1; }
  printf '{"path": "%s", "source": "%s/%s", "bytes": %s, "sha256": "%s", "files": %s, "buckets_sha256": "%s", "stats_sha256": "%s", "markers": "%s", "uploaded": "%s", "job": "%s", "subset_f": %s}\n' \
    $rel $G $rel $bytes $sha $got "$hb" "$hs" "$(ox "ls $G/$rel | grep complete | tr '\n' ' '")" "$(date -Is)" $SLURM_JOB_ID "${SUBSET_F:-null}" > $f.json
  gcloud storage cp -q $f.json $D/$rel.tar.json && rm -f $f $f.n $f.json
  echo "$rel: $((bytes / 1000000)) MB, $got files, pull $((t1 - t0))s, upload $(( $(date +%s) - t1 ))s"
}
for rel in "$@"; do one $rel & while [ $(jobs -rp | wc -l) -ge ${STREAMS:-4} ]; do wait -n; done; done; wait
gcloud storage cat "$D/**.tar.json" 2>/dev/null | $PY -c "import json, sys; rows = [json.loads(l) for l in sys.stdin]; print(json.dumps(dict(objects=len(rows), bytes=sum(r['bytes'] for r in rows), rows=sorted(rows, key=lambda r: r['path'])), indent=1))" > $LOCAL/MANIFEST.json && gcloud storage cp -q $LOCAL/MANIFEST.json $D/MANIFEST.json
echo "$(date) done: $(grep -m1 objects $LOCAL/MANIFEST.json) $(grep -m1 '"bytes"' $LOCAL/MANIFEST.json | tail -1)"
