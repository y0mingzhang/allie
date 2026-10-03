"""A frozen data selection: explicit month dirs, the sha256 of each month's buckets.json
(games per bucket and the shards a sampler reads) and stats.json, their summed inventory
and a hash of it all. Stores that gain months never change a pin; a pinned month rebuilt
or edited fails verify().

  datapin.py OUT FIRST HELD_OUT   built Lichess months >= FIRST but HELD_OUT, + ext-v1
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ALLIE = Path(os.environ.get("ALLIE_DATA", "/data/group_data/dei-group/yimingz3/allie"))
LICHESS = [ALLIE / s for s in ("data-v1", "data-v1-hist", "data-v1-hist2")]
EXT = [ALLIE / "ext-v1/otb", ALLIE / "ext-v1/engine"]
FILES = ("buckets.json", "stats.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def built(m):
    """Finalized: stats and buckets written and every listed shard present."""
    b = m / "buckets.json"
    return (
        (m / "stats.json").exists()
        and b.exists()
        and all(
            (m / s).exists() for x in json.loads(b.read_text()) for s in x["shards"]
        )
    )


def pin(first, held_out):
    months = sorted(
        (m for s in LICHESS for m in s.glob("20[0-9][0-9]-[0-9][0-9]")),
        key=lambda m: m.name,
    )
    months = [m for m in months if first <= m.name != held_out and built(m)]
    months += [m for s in EXT for m in sorted(s.glob("20xx-*")) if built(m)]
    names = [m.name for m in months]
    assert len(set(names)) == len(names), "a month is in two stores"
    inventory = {}
    for m in months:
        for b in json.loads((m / "buckets.json").read_text()):
            inventory[str(b["code"])] = inventory.get(str(b["code"]), 0) + b["games"]
    p = dict(
        first=first,
        held_out=held_out,
        stores=sorted({str(m.parent) for m in months}),
        months=[str(m) for m in months],
        files={str(m): {f: sha(m / f) for f in FILES} for m in months},
        inventory=inventory,
    )
    return p | dict(sha256=digest(p))


def verify(path, months):
    """The pin is unedited, the run has exactly its months, and none of them changed."""
    p = json.loads(Path(path).read_text())
    assert p.pop("sha256") == digest(p), f"{path} edited after pinning"
    assert sorted(months) == sorted(p["months"]), "run months differ from the pin"
    bad = [m for m, h in p["files"].items() if {f: sha(Path(m) / f) for f in h} != h]
    assert not bad, f"pinned months changed on disk: {bad}"


if __name__ == "__main__":
    out, first, held_out = sys.argv[1:]
    p = pin(first, held_out)
    assert not Path(out).exists(), "never overwrite a pin"
    Path(out).write_text(json.dumps(p, indent=1) + "\n")
    games = sum(p["inventory"].values())
    print(f"{len(p['months'])} months, {games / 1e9:.3f}B games, sha256 {p['sha256']}")
