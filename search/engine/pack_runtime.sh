#!/usr/bin/env bash
# One durable stream avoids tens of thousands of cold shared-filesystem lookups.
set -euo pipefail
cd "$(dirname "$0")/../.."
SOURCE="$PWD/results/search-v1/runtime/sglang-0.5.9"
ARCHIVE="$PWD/results/search-v1/runtime/sglang-0.5.9-torch2.9.1-cu128.tar.zst"
exec 9>"$ARCHIVE.lock"
flock -n 9 || exit 0
[[ ! -f "$ARCHIVE.sha256" ]] || exit 0
START=$(date +%s)
/usr/bin/python3 -B -m search.engine.pack_runtime "$SOURCE" | zstd -T2 -1 -f -o "$ARCHIVE.partial"
mv "$ARCHIVE.partial" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256.partial"
mv "$ARCHIVE.sha256.partial" "$ARCHIVE.sha256"
printf 'runtime archive ready; elapsed=%s seconds\n' "$(( $(date +%s) - START ))"
