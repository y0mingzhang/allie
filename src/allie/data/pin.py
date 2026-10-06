"""A frozen data selection: explicit month dirs, the sha256 of each month's buckets.json
(games per bucket and the shards a sampler reads) and stats.json, their summed inventory
and a hash of it all. Stores that gain months never change a pin; a pinned month rebuilt
or edited fails verify(). A pin may exclude, per month, data-v1 format ids wholesale (their
buckets are neither drawn nor in the inventory) and the games whose token_hash is in a file
of sorted little-endian uint64 (never accepted by data.mix; they stay in the inventory).
Pins without exclusions are as before.

  data.pin OUT FIRST HELD_OUT [MONTH:FORMAT,...[:HASHES] ...]
      built Lichess months >= FIRST but HELD_OUT, + ext-v1; e.g. 2026-07:blitz:H also
      drops July's blitz buckets and its games in H
  data.pin hashes OUT GOLDEN    the token hashes of a strat-eval golden's games
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from allie.paths import DATA

LICHESS = [DATA / s for s in ("data-v1", "data-v1-hist", "data-v1-hist2")]
EXT = [DATA / "ext-v1/otb", DATA / "ext-v1/engine"]
FILES = ("buckets.json", "stats.json")
FORMATS = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")


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


def exclusion(months, arg):
    """MONTH:FORMAT,...[:HASHES] as (month dir, spec)."""
    name, formats, *h = arg.split(":")
    dirs = {m.name: str(m) for m in months}
    assert name in dirs, f"{name} is not pinned"
    spec = {"formats": sorted(FORMATS.index(f) for f in formats.split(",") if f)}
    if h:
        path = Path(h[0]).resolve()
        spec |= {"hashes": str(path), "hashes_sha256": sha(path)}
    return dirs[name], spec


def pin(first, held_out, *exclude):
    months = sorted(
        (m for s in LICHESS for m in s.glob("20[0-9][0-9]-[0-9][0-9]")),
        key=lambda m: m.name,
    )
    months = [m for m in months if first <= m.name != held_out and built(m)]
    months += [m for s in EXT for m in sorted(s.glob("20xx-*")) if built(m)]
    names = [m.name for m in months]
    assert len(set(names)) == len(names), "a month is in two stores"
    ex = dict(exclusion(months, e) for e in exclude)
    inventory = {}
    for m in months:
        drop = ex.get(str(m), {}).get("formats", ())
        for b in json.loads((m / "buckets.json").read_text()):
            if b["code"] // 10000 % 10 - 1 not in drop:
                c = str(b["code"])
                inventory[c] = inventory.get(c, 0) + b["games"]
    p = dict(
        first=first,
        held_out=held_out,
        stores=sorted({str(m.parent) for m in months}),
        months=[str(m) for m in months],
        files={str(m): {f: sha(m / f) for f in FILES} for m in months},
        inventory=inventory,
    )
    p |= {"exclusions": ex} if ex else {}
    return p | dict(sha256=digest(p))


def verify(path, months):
    """The pin is unedited, the run has exactly its months, and none of them changed."""
    p = json.loads(Path(path).read_text())
    assert p.pop("sha256") == digest(p), f"{path} edited after pinning"
    assert sorted(months) == sorted(p["months"]), "run months differ from the pin"
    bad = [m for m, h in p["files"].items() if {f: sha(Path(m) / f) for f in h} != h]
    bad += [
        e["hashes"]
        for e in p.get("exclusions", {}).values()
        if "hashes" in e and sha(e["hashes"]) != e["hashes_sha256"]
    ]
    assert not bad, f"pinned months changed on disk: {bad}"


def hashes(out, golden):
    """Every game of GOLDEN/strat.npz (BOS to its terminal token; rows pad with BOS) by
    data.store.hash64 of its tokens, the store's token_hash."""
    from allie.data.store import hash64
    from allie.data.vocab import BOS, TERM_NORMAL, TERM_OTHER

    found = set()
    for r in np.load(Path(golden) / "strat.npz")["rows"]:
        b = np.flatnonzero(r == BOS)
        for s, e in zip(b, [*b[1:], len(r)]):
            if e - s > 1:
                assert r[e - 1] in (TERM_NORMAL, TERM_OTHER), "a truncated game"
                found.add(hash64(r[s:e]))
    assert not Path(out).exists(), "never overwrite a pin's hashes"
    np.array(sorted(found), "<u8").tofile(out)
    print(f"{len(found)} token hashes")


if __name__ == "__main__":
    if sys.argv[1] == "hashes":
        hashes(*sys.argv[2:])
        sys.exit()
    out, first, held_out, *exclude = sys.argv[1:]
    p = pin(first, held_out, *exclude)
    assert not Path(out).exists(), "never overwrite a pin"
    Path(out).write_text(json.dumps(p, indent=1) + "\n")
    games = sum(p["inventory"].values())
    print(f"{len(p['months'])} months, {games / 1e9:.3f}B games, sha256 {p['sha256']}")
