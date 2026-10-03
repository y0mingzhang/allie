"""Exact all-history games per bucket from built stores (the sampler's `history` counts).

Usage: data.history OUT.json STORE [STORE ...]. Sums buckets.json over every built month (stats.json present) and
lists the published Lichess months not built yet (counts.txt), so a partial file is never mistaken for all history.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

COUNTS = "https://database.lichess.org/standard/counts.txt"


def published(stores):
    for s in stores:
        cached = Path(s) / "counts.txt"
        if cached.exists():
            text = cached.read_text()
            break
    else:
        text = urllib.request.urlopen(COUNTS).read().decode()
    return {
        m[1]: int(m[2])
        for m in re.finditer(r"rated_(\d{4}-\d{2})\.pgn\.zst\s+(\d+)", text)
    }


def main(out, *stores):
    counts, months, games = {}, {}, 0
    for s in stores:
        for stats in sorted(Path(s).glob("20*/stats.json")):
            m = stats.parent.name
            assert m not in months, f"{m} built in two stores: {months[m]}, {s}"
            months[m] = s
            for b in json.loads((stats.parent / "buckets.json").read_text()):
                counts[str(b["code"])] = counts.get(str(b["code"]), 0) + b["games"]
                games += b["games"]
    pub = published(stores)
    missing = sorted(m for m in pub if m not in months)
    Path(out).write_text(
        json.dumps(
            dict(
                months=len(months),
                games=games,
                exact_months=sorted(months),
                exact_games=games,
                missing_months=missing,
                missing_published_games=sum(pub[m] for m in missing),
                complete=not missing,
                stores=list(stores),
                samples={},
                counts=dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
            ),
            indent=1,
        )
        + "\n"
    )
    print(
        f"{len(months)} months, {games / 1e9:.3f}B games, {len(counts)} buckets; {len(missing)} published months missing"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
