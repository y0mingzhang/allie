#!/bin/bash
# render.sh [SCENE...]: renders the named scenes of how_allie_works.py (default all) at 1920x1080 30 fps, one process
# each, into $SCENES (set it to keep them, so a re-render touches only what changed), then joins all scenes into how-allie-works.mp4
# (H.264 at the lowest CRF from 20 that fits 20 MB), the GIF teaser (under 8 MB) and the poster.
# Needs manim 0.19, ffmpeg and the cairo and pango libraries. PY: the python that has manim.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
PY=${PY:-python}
if [[ -n ${MEDIA:-} ]]; then
	media=$MEDIA
else
	media=$(mktemp -d)
	trap 'rm -rf "$media"' EXIT
fi
SCENES=${SCENES:-$media/scenes}
mkdir -p "$SCENES"
all=$($PY -c "import how_allie_works as m; print(' '.join(m.VIDEO['scenes']))")
todo=${*:-$all}
pids=()
for s in $todo; do
	$PY -m manim -v WARNING --progress_bar none -r 1920,1080 --frame_rate 30 --media_dir "$media/$s" how_allie_works.py "$s" &
	pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for s in $todo; do
	f=$media/$s/videos/how_allie_works/1080p30/$s.mp4
	if [[ -f $f ]]; then cp "$f" "$SCENES/$s.mp4"; else echo "scene $s failed"; fi
done
((fail == 0)) || { echo "a scene failed, not joining"; exit 0; }
list=$(mktemp)
at=()
t=0
for s in $all; do
	[[ -f $SCENES/$s.mp4 ]] || { echo "missing scene $s, not joining"; exit 0; }
	echo "file '$SCENES/$s.mp4'" >>"$list"
	d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SCENES/$s.mp4")
	at+=("$s=$t")
	t=$(awk "BEGIN {print $t + $d}")
	echo "$s $d"
done
echo "total $t"
read -r gif_ss gif_t poster < <($PY -c "
import sys, how_allie_works as m
at = dict(kv.split('=') for kv in sys.argv[1:])
f = lambda x: float(at[x[0]]) + x[1]
v = m.VIDEO
print(f(v['gif'][0]), v['gif'][1], f(v['poster']))" "${at[@]}")
mp4=how-allie-works.mp4
for crf in 20 22 24 26 28 30 32; do
	ffmpeg -loglevel error -y -f concat -safe 0 -i "$list" -an -c:v libx264 -preset slow -tune animation \
		-crf $crf -pix_fmt yuv420p -movflags +faststart "$mp4"
	(($(wc -c <"$mp4") <= 20000000)) && break
done
echo "crf $crf $(wc -c <"$mp4") bytes"
ffmpeg -loglevel error -y -ss "$poster" -i "$mp4" -frames:v 1 how-allie-works.png
for w in 800 640 480; do
	ffmpeg -loglevel error -y -ss "$gif_ss" -t "$gif_t" -i "$mp4" -vf "fps=15,scale=$w:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 how-allie-works.gif
	(($(wc -c <how-allie-works.gif) <= 8000000)) && break
done
echo "gif $w $(wc -c <how-allie-works.gif) bytes"
rm -f "$list"
