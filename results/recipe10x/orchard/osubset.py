"""osubset.py F OUT REL...: per month dir REL (under the babel store), OUT/REL.list = its top-level files plus the shards a
sampler with pool_frac <= F can open (chessmix._index with scale F: the first ceil(ceil(F x games) / 50000) shards of
every bucket), relative to the store and sorted: oupload.sh's input with SUBSET=OUT SUBSET_F=F. Valid only where the
run's per-bucket scale is at most F (history counts == disk counts, as for the v4 pin; orun.py re-checks every run)."""

import json, math, sys
from pathlib import Path

G = Path("/data/group_data/dei-group/yimingz3/allie")
f, out, rels = float(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
for rel in rels:
    m = G / rel
    files = [p.relative_to(G) for p in m.iterdir() if p.is_file()]
    for b in json.loads((m / "buckets.json").read_text()):
        left = math.ceil(f * b["games"])
        for k, s in enumerate(b["shards"]):
            if min(50000, b["games"] - k * 50000, left) <= 0:
                break
            files.append(Path(rel) / s)
            left -= 50000
    p = out / f"{rel}.list"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{x}\n" for x in sorted(map(str, files))))
    print(rel, len(files), "files")
