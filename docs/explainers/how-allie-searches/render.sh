#!/bin/bash
# render.sh [SCENE ...]: renders the video's scenes in parallel and joins them into how-allie-searches.mp4 (1080p, under
# 20 MB), .gif (under 8 MB) and .png. With scene names: those scenes only, at PREVIEW=1 quality when set, kept in $MEDIA.
# Needs PY (a Python with manim 0.19), ffmpeg.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
PY=${PY:-python}
media=${MEDIA:-$(mktemp -d)}
export PYTHONDONTWRITEBYTECODE=1
read -r name scenes < <($PY -c "import how_allie_searches as m; print(m.VIDEO['name'], ' '.join(m.VIDEO['scenes']))")
[[ ${PREVIEW:-0} == 1 ]] && res=960,540 fr=15 || res=1920,1080 fr=30
todo=${*:-$scenes}
pids=()
for s in $todo; do
	$PY -m manim -v WARNING --progress_bar none --disable_caching -r $res --frame_rate $fr --media_dir "$media/$s" how_allie_searches.py "$s" &
	pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
[[ $# -gt 0 ]] && exit 0
list=$media/list.txt
: >"$list"
durs=()
for s in $scenes; do
	f=$media/$s/videos/how_allie_searches/${res#*,}p$fr/$s.mp4
	echo "file '$f'" >>"$list"
	durs+=("$s=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")")
done
read -r gif_at gif_t poster < <($PY -c "
import sys, how_allie_searches as m
at, t = {}, 0.0
for kv in sys.argv[1:]:
    k, d = kv.split('=')
    at[k], t = t, t + float(d)
v = m.VIDEO
print(at[v['gif'][0][0]] + v['gif'][0][1], v['gif'][1], at[v['poster'][0]] + v['poster'][1])" "${durs[@]}")
for crf in 20 23 26 29 32; do
	ffmpeg -loglevel error -y -f concat -safe 0 -i "$list" -an -c:v libx264 -preset slow -tune animation -crf $crf -pix_fmt yuv420p -movflags +faststart "$name.mp4"
	(($(wc -c <"$name.mp4") <= 20000000)) && break
done
ffmpeg -loglevel error -y -ss "$poster" -i "$name.mp4" -frames:v 1 "$name.png"
for w in 960 720 640 480; do
	ffmpeg -loglevel error -y -ss "$gif_at" -t "$gif_t" -i "$name.mp4" -vf "fps=15,scale=$w:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 "$name.gif"
	(($(wc -c <"$name.gif") <= 8000000)) && break
done
echo "crf $crf, gif width $w"
ls -l "$name".*
ffprobe -v error -show_entries format=duration -of csv=p=0 "$name.mp4"
