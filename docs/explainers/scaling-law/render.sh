#!/bin/bash
# render.sh: every scene of VIDEO at 1920x1080 30 fps, in parallel, joined into $OUT (default here): scaling-law.mp4 under
#   20 MB, a looping scaling-law.gif under 8 MB, a poster scaling-law.png. Needs manim 0.19 ($PY runs it), ffmpeg, cairo, pango.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
out=${OUT:-.}
media=${MEDIA:-$(mktemp -d)}
[[ -n ${MEDIA:-} ]] || trap 'rm -rf "$media"' EXIT
mod=scaling_law
scenes=$($PY -c "import $mod as m; print(' '.join(m.VIDEO['scenes']))")
pids=()
for s in $scenes; do
	$PY -m manim -v WARNING --progress_bar none -r 1920,1080 --frame_rate 30 --media_dir "$media/$s" $mod.py "$s" &
	pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
list=$media/list.txt
: >"$list"
durs=()
for s in $scenes; do
	f=$media/$s/videos/$mod/1080p30/$s.mp4
	echo "file '$f'" >>"$list"
	durs+=("$s=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")")
done
printf '%s\n' "${durs[@]}"
read -r gif_ss gif_t poster < <($PY -c "
import sys, $mod as m
at, t = {}, 0.0
for kv in sys.argv[1:]:
    k, d = kv.split('=')
    at[k], t = t, t + float(d)
g, p = m.VIDEO['gif'], m.VIDEO['poster']
print(at[g[0]] + g[1], g[2], at[p[0]] + p[1])" "${durs[@]}")
mp4=$out/scaling-law.mp4
for crf in 16 18 20 23 26 29; do
	ffmpeg -loglevel error -y -f concat -safe 0 -i "$list" -an -c:v libx264 -preset slow -tune animation -crf $crf -pix_fmt yuv420p -movflags +faststart "$mp4"
	(($(wc -c <"$mp4") <= 20000000)) && break
done
ffmpeg -loglevel error -y -ss "$poster" -i "$mp4" -frames:v 1 "$out/scaling-law.png"
for w in 960 800 720 640 480; do
	ffmpeg -loglevel error -y -ss "$gif_ss" -t "$gif_t" -i "$mp4" -vf "fps=15,scale=$w:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 "$out/scaling-law.gif"
	(($(wc -c <"$out/scaling-law.gif") <= 8000000)) && break
done
echo "crf $crf, gif width $w"
ls -l "$out"/scaling-law.*
ffprobe -v error -show_entries format=duration -of csv=p=0 "$mp4"
