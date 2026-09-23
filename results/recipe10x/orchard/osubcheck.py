"""osubcheck.py LISTDIR PREFIX (orchard login): every LISTDIR/<rel>.list has gs://.../data/PREFIX/<rel>.tar and .tar.json
with subset_f recorded, the list's file count and the tar object's byte size. Exit 1 unless all months are complete."""
import json, subprocess, sys
from pathlib import Path

d, pre = Path(sys.argv[1]), sys.argv[2]
B = f"gs://cmu-gpucloud-yimingz3/data/{pre}"
gs = lambda *a: subprocess.run(["gcloud", "storage", *a], capture_output=True, text=True).stdout
rels = sorted(str(p.relative_to(d))[: -len(".list")] for p in d.rglob("*.list"))
meta = {r["path"]: r for r in map(json.loads, gs("cat", f"{B}/**.tar.json").splitlines())}
size = {}
for line in gs("ls", "-l", f"{B}/**.tar").splitlines():
    p = line.split()
    if len(p) >= 3 and p[-1].endswith(".tar"):
        size[p[-1][len(B) + 1 : -len(".tar")]] = int(p[0])
n = lambda r: len((d / f"{r}.list").read_text().split())
bad = [r for r in rels if r not in meta or meta[r].get("subset_f") is None or meta[r]["files"] != n(r) or size.get(r) != meta[r]["bytes"]]
print(f"{len(rels) - len(bad)} of {len(rels)} subset months complete" + (f"; not yet: {bad[:4]}..." if bad else ""))
sys.exit(1 if bad else 0)
