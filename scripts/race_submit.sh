#!/bin/bash
# race_submit.sh NAME SCRIPT OPTS...: queue SCRIPT once per OPTS (one sbatch option string per queue, e.g.
# "-p general --qos=normal --gres=gpu:L40S:4"); the first copy to start runs it, the others are cancelled or exit.
# Every copy must be the same single task (an array script needs one --array=N). RACE_DIR keeps the guard and wrapper
# the copies run, their hashes, OPTS, and submits: each sbatch before it is sent and the id it returned. On failure the
# known copies are cancelled and RACE_DIR is kept; a copy it does not list in RACE_DIR/jobs cannot run. Every copy
# excludes the shared bad-node list (BAD_NODES, one node per line, # comments); OPTS must not pass --exclude.
set -euo pipefail
(($# >= 3)) || { echo "usage: race_submit.sh NAME SCRIPT OPTS..." >&2; exit 1; }
name=$1 script=$(readlink -f "$2")
shift 2
root=${RACE_ROOT:-/data/group_data/dei-group/yimingz3/allie/race}
bad=$(sed 's/#.*//' "${BAD_NODES:-/data/group_data/dei-group/yimingz3/allie/bad-nodes}" | xargs | tr ' ' ,)
for o in "$@"; do
	[[ " $o" != *" --exclude"* && " $o" != *" -x"* ]] || { echo "race $name: the shared bad-node list sets --exclude" >&2; exit 1; }
done
mkdir -p "$root"
dir=$root/$name
mkdir "$dir" || { echo "race $name exists: $dir" >&2; exit 1; }
ids=() sent=0
fail() {
	trap - ERR INT TERM HUP
	echo "race $name: ${1:-failed}; cancelling ${ids[*]:-nothing}; receipt in $dir" >&2
	((sent == ${#ids[@]})) || echo "an sbatch gave no id; a copy it made is held and cannot run:" \
		"squeue --me -h -o '%i %o' | grep -F $dir/" >&2
	((${#ids[@]} == 0)) || scancel "${ids[@]}"
	exit 1
}
trap fail ERR INT TERM HUP
cp "$(dirname "$(readlink -f "$0")")/race_guard.sh" "$dir/race_guard.sh"
body=$dir/$(basename "$script")
g="RACE_DIR=$(printf %q "$dir") source $(printf %q "$dir/race_guard.sh") || exit 1"
g=$g awk '!done && !/^[[:space:]]*(#|$)/ { print ENVIRON["g"]; done = 1 } { print }' "$script" > "$body"
grep -qxF "$g" "$body" || fail "$script has no commands"
sha256sum "$script" "$body" "$dir/race_guard.sh" > "$dir/sha256"
printf '%s\n' "$@" > "$dir/opts"
# sbatch exports our env and Slurm sets SLURM_ARRAY_* only for arrays: a leaked one would fool the guard
unset "${!SLURM_ARRAY_@}"
task=
for opts in "$@"; do
	read -ra o <<< "$opts"
	echo "sbatch $opts" >> "$dir/submits"
	sent=$((sent + 1))
	id=$(sbatch --parsable --hold ${bad:+--exclude=$bad} "${o[@]}" "$body")
	id=${id%%;*}
	[[ $id =~ ^[0-9]+$ ]] || fail "sbatch returned '$id'"
	ids+=("$id")
	echo "id $id" >> "$dir/submits"
	t=$(squeue -h -j "$id" -o %K)
	[[ $t == N/A || $t =~ ^[0-9]+$ ]] || fail "job $id is an array of tasks $t; pass one --array=N"
	[[ -z $task || $t == "$task" ]] || fail "job $id runs task $t, an earlier copy $task"
	task=$t
done
printf '%s\n' "${ids[@]}" > "$dir/jobs.tmp"
mv "$dir/jobs.tmp" "$dir/jobs"
trap - ERR INT TERM HUP
echo "${ids[@]}"
scontrol release "${ids[@]}"
