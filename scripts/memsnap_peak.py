"""Peak of a torch CUDA memory snapshot's allocation history, and what was live at the peak, grouped by
the innermost repo frame of each allocation: memsnap_peak.py SNAPSHOT.pickle [DEVICE]"""

import collections
import pickle
import sys

snap = pickle.load(open(sys.argv[1], "rb"))
trace = snap["device_traces"][int(sys.argv[2]) if len(sys.argv) > 2 else 0]


def site(e):
    frames = e.get("frames") or []
    return next(
        (
            f"{f['filename'].rsplit('/', 1)[-1]}:{f['line']} {f['name']}"
            for f in frames
            if "/src/" in f["filename"] or "worktrees" in f["filename"]
        ),
        frames[0]["name"] if frames else "?",
    )


live, cur, peak, at = {}, 0, 0, None
for i, e in enumerate(trace):
    if e["action"] == "alloc":
        live[e["addr"]] = e
        cur += e["size"]
        if cur > peak:
            peak, at = cur, dict(live)
    elif e["action"] in ("free_completed",) and e["addr"] in live:
        cur -= live.pop(e["addr"])["size"]
by, n = collections.Counter(), collections.Counter()
for e in (at or {}).values():
    by[site(e)] += e["size"]
    n[site(e)] += 1
print(f"history peak {peak / 1e9:.2f} GB over {len(trace)} events (allocations before recording are not counted)")
for s, v in by.most_common(25):
    print(f"{v / 1e9:8.2f} GB {n[s]:6d}  {s}")
