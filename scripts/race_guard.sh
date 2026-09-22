# sourced first by a race_submit.sh copy (RACE_DIR set on the same line): the first registered single-task copy to
# start links its id to RACE_DIR/owner (atomic) and cancels its twins; a requeue of the owner passes, others exit
id=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
[[ ${SLURM_ARRAY_TASK_COUNT:-1} == 1 ]] || { echo "race: $SLURM_ARRAY_TASK_COUNT array tasks, need 1" >&2; exit 1; }
grep -qx "$id" "$RACE_DIR/jobs" || { echo "race: job $id is not registered in $RACE_DIR/jobs" >&2; exit 1; }
echo "$id" > "$RACE_DIR/cand.$id" || exit 1
if ! ln "$RACE_DIR/cand.$id" "$RACE_DIR/owner" 2>/dev/null; then
	owner=$(< "$RACE_DIR/owner") || exit 1
	[[ $owner == "$id" ]] || { echo "race lost to job $owner"; exit 0; }
fi
grep -vx "$id" "$RACE_DIR/jobs" | xargs -r scancel 2>/dev/null || :
return 0
