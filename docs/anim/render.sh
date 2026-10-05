#!/bin/bash
# Usage: render.sh MODULE   renders the video MODULE.py describes in VIDEO = dict(name, scenes, gif=(start, seconds),
#   poster=seconds), a start or poster time being seconds or [SCENE, seconds into it]. Each scene renders in parallel
#   at 1920x1080 30 fps; they are joined into ../videos/NAME.mp4 (H.264 at the lowest CRF from 20 that fits MAXMB,
#   default 20), a looping GIF teaser NAME.gif (<= 8 MB) and a poster NAME.png. PREVIEW=1 renders 960x540 at 15 fps
#   into $OUT (default ../videos/preview) for quick checks.
# Needs uv (it runs manim; override with PY or MANIM), cairo and pango development libraries, ffmpeg, and the
# DejaVu Sans font for chess pieces. MEDIA sets manim's scratch directory (default a temporary one).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
PY=${PY:-uv run -q --no-project --python 3.12 --with manim==0.19.0 python}
MANIM=${MANIM:-$PY -m manim}
media=${MEDIA:-$(mktemp -d)}
[[ -n ${MEDIA:-} ]] || trap 'rm -rf "$media"' EXIT

mod=$1
read -r name scenes < <($PY -c "import $mod as m; print(m.VIDEO['name'], ' '.join(m.VIDEO['scenes']))")
if [[ ${PREVIEW:-0} == 1 ]]; then
	res=960,540 fr=15 out=${OUT:-../videos/preview}
else
	res=1920,1080 fr=30 out=${OUT:-../videos}
fi
mkdir -p "$out"
pids=()
for s in $scenes; do
	$MANIM -v WARNING --progress_bar none -r $res --frame_rate $fr --media_dir "$media/$s" "$mod.py" "$s" &
	pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
list=$media/list.txt
: >"$list"
durs=()
for s in $scenes; do
	f=$media/$s/videos/$mod/${res#*,}p$fr/$s.mp4
	echo "file '$f'" >>"$list"
	durs+=("$s=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")")
done
# scene-relative times to seconds into the video
read -r gif_ss gif_t poster < <($PY -c "
import sys, $mod as m
at, t = {}, 0.0
for kv in sys.argv[1:]:
    k, d = kv.split('=')
    at[k], t = t, t + float(d)
f = lambda x: at[x[0]] + x[1] if isinstance(x, (list, tuple)) else x
v = m.VIDEO
print(f(v['gif'][0]), v['gif'][-1], f(v['poster']))" "${durs[@]}")

mp4=$out/$name.mp4 max=$((${MAXMB:-20} * 1000000))
for crf in 20 23 26 29 32; do
	ffmpeg -loglevel error -y -f concat -safe 0 -i "$list" -an -c:v libx264 -preset slow -tune animation \
		-crf $crf -pix_fmt yuv420p -movflags +faststart "$mp4"
	(($(wc -c <"$mp4") <= max)) && break
done
ffmpeg -loglevel error -y -ss "$poster" -i "$mp4" -frames:v 1 "$out/$name.png"
for w in 640 480; do
	ffmpeg -loglevel error -y -ss "$gif_ss" -t "$gif_t" -i "$mp4" -vf "fps=15,scale=$w:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 "$out/$name.gif"
	(($(wc -c <"$out/$name.gif") <= 8000000)) && break
done
echo "crf $crf, gif width $w"
ls -l "$out/$name".*
ffprobe -v error -show_entries format=duration -of csv=p=0 "$mp4"
