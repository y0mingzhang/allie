#!/bin/bash
# orchard_mirror.sh [DEST]: copy the small live logs of our orchard runs to babel for the runs dashboard.
# One listing over `ssh orchard`: our squeue and sacct (for the compute log), then every trainer train.log
# (~/allie/studies/*/logs) and orun job log (~/allie/src/logs) written in the last 6 h, with size and mtime.
# Each file that grew is fetched from the local copy's size up to the listed size, exactly those bytes, so a
# broken transfer never appends a partial tail. Read-only on orchard; writes only under DEST (default
# results/runindex/orchard). The files are KB-MB; the IAP tunnel moves ~1.5 MB/s.
set -euo pipefail
dest=${1:-/home/yimingz3/src/allie/results/runindex/orchard}
fmt='%A|%i|%j|%K|%T|%P|%q|%b|%D|%N|%S|%M|%r|%k|%o' # runindex.py SQFMT
cols=JobIDRaw,JobID,JobName,Partition,AllocTRES,ElapsedRaw,State,Start,End
o() { timeout 120 ssh -n -o BatchMode=yes -o ConnectTimeout=30 orchard "$@"; }
mkdir -p "$dest/studies" "$dest/logs"
tmp=$(mktemp -d "$dest/.pull.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
o "squeue --me -h -o '$fmt'" > "$tmp/squeue.txt"
o "sacct -u \$USER -S 2026-09-21T00:00:00 -X -D -n -P -o $cols" > "$tmp/sacct.txt"
o 'find ~/allie/studies/*/logs -maxdepth 1 -name "*.train.log" -mmin -360 -size -64M -printf "%p %s %T@\n" 2> /dev/null
	find ~/allie/src/logs -maxdepth 1 -name "*.out" -mmin -360 -size -8M -printf "%p %s %T@\n" 2> /dev/null
	true' > "$tmp/files.txt"
while read -r path size _; do
	case $path in
	*/allie/studies/*/logs/*.train.log) rel=studies/$(basename "$(dirname "$(dirname "$path")")")/${path##*/} ;;
	*/allie/src/logs/*.out) rel=logs/${path##*/} ;;
	*) continue ;;
	esac
	copy=$dest/$rel
	mkdir -p "${copy%/*}"
	have=$(stat -c %s "$copy" 2> /dev/null || echo 0)
	if ((size < have)); then # replaced or truncated on orchard: start over
		rm -f "$copy"
		have=0
	fi
	((size > have)) || continue
	o "tail -c +$((have + 1)) '$path' | head -c $((size - have))" > "$tmp/part" || continue
	if (($(stat -c %s "$tmp/part") == size - have)); then
		cat "$tmp/part" >> "$copy"
	else
		echo "$(date -Is) short read of $path, retried next pass" >&2
	fi
done < "$tmp/files.txt"
mv "$tmp/squeue.txt" "$tmp/sacct.txt" "$tmp/files.txt" "$dest/"
