"""osubcheck.py LISTDIR PREFIX F LISTS_SHA256 (orchard login): exit 0 only if LISTDIR holds exactly the frozen list set
(digest over sorted rel + list contents) and every <rel>.list has gs://.../data/PREFIX/<rel>.tar and .tar.json with
subset_f == F, the list's file count and the tar object's byte size. Checks archive availability, not content (orun.py
verifies the pin and the exact sampler shards at staging)."""
import hashlib, json, subprocess, sys
from pathlib import Path

d, pre, f, digest = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
B = f"gs://cmu-gpucloud-yimingz3/data/{pre}"
rels = sorted(str(p.relative_to(d))[: -len(".list")] for p in d.rglob("*.list"))
lists = {r: (d / f"{r}.list").read_text() for r in rels}
h = hashlib.sha256("".join(f"{r}\n{lists[r]}" for r in rels).encode()).hexdigest()
assert rels and h == digest, f"list set {len(rels)} months, digest {h} != {digest}"


def gs(*a):
    p = subprocess.run(["gcloud", "storage", *a], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 or "matched no objects" in p.stderr, f"gcloud {a}: {p.returncode} {p.stderr[-300:]}"
    return p.stdout if p.returncode == 0 else ""


meta = {r["path"]: r for r in map(json.loads, gs("cat", f"{B}/**.tar.json").splitlines())}
size = {}
for line in gs("ls", "-l", f"{B}/**.tar").splitlines():
    p = line.split()
    if len(p) >= 3 and p[-1].endswith(".tar"):
        size[p[-1][len(B) + 1 : -len(".tar")]] = int(p[0])
ok = lambda r: r in meta and meta[r].get("subset_f") == f and meta[r]["files"] == len(lists[r].split()) and size.get(r) == meta[r]["bytes"]
bad = [r for r in rels if not ok(r)]
print(f"{len(rels) - len(bad)} of {len(rels)} subset months complete" + (f"; not yet: {bad[:4]}..." if bad else ""))
sys.exit(1 if bad else 0)
