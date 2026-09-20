"""Live CUDA allocations of a torch memory snapshot, grouped by their innermost repo frame:
memsnap.py SNAPSHOT.pickle"""

import collections
import pickle
import sys

snap = pickle.load(open(sys.argv[1], "rb"))
live = collections.Counter()
count = collections.Counter()
for seg in snap["segments"]:
    for b in seg["blocks"]:
        if b["state"] != "active_allocated":
            continue
        frames = b.get("frames") or []
        site = next(
            (
                f"{f['filename'].rsplit('/', 1)[-1]}:{f['line']} {f['name']}"
                for f in frames
                if "/src/" in f["filename"] or "worktrees" in f["filename"]
            ),
            frames[0]["name"] if frames else "?",
        )
        live[site] += b["size"]
        count[site] += 1
total = sum(live.values())
print(f"live {total / 1e9:.2f} GB")
for site, size in live.most_common(25):
    print(f"{size / 1e9:8.2f} GB {count[site]:6d}  {site}")
