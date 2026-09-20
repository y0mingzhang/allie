"""Policy tokens per mix pass, per built month, with chessmix's own shard reader and policy weights.

Usage: pass_tokens.py POLICY OUT.json MONTH_DIR [MONTH_DIR ...]. The sampler draws each game in proportion to its
policy weight w (validation leaks 0), so one pass of the mix over a month is sum(w * tokens); `unique` counts each
kept game once (sum(min(w, 1) * tokens)); tokens per game are chessmix's (11 header + moves + termination, cut at
the 1025-token row).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chessmix as cm


def shard(args):
    path, policy = args
    s = cm.Shard(path)
    s.policy(cm.POLICIES[policy] if policy in cm.POLICIES else cm.compose(policy), 0.0)
    n = np.minimum(np.diff(s.g.off) + 12, cm.ROW)
    return (
        s.g.month,
        s.g.n,
        int(n.sum()),
        float((s.w * n).sum()),
        float((np.minimum(s.w, 1) * n).sum()),
    )


def main(policy, out, *months):
    jobs = [
        (str(p), policy)
        for m in months
        for p in sorted(Path(m).glob("games/b*/shard-*.parquet"))
    ]
    rows = {}
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as ex:
        for month, games, raw, passed, unique in ex.map(shard, jobs, chunksize=8):
            r = rows.setdefault(
                month, dict(games=0, raw_tokens=0, pass_tokens=0.0, unique_tokens=0.0)
            )
            r["games"] += games
            r["raw_tokens"] += raw
            r["pass_tokens"] += passed
            r["unique_tokens"] += unique
    total = {
        k: sum(r[k] for r in rows.values())
        for k in ("games", "raw_tokens", "pass_tokens", "unique_tokens")
    }
    Path(out).write_text(
        json.dumps(
            dict(policy=policy, months=dict(sorted(rows.items())), total=total),
            indent=1,
        )
        + "\n"
    )
    print(
        f"{policy}: {len(rows)} months, {total['games'] / 1e9:.3f}B games, raw {total['raw_tokens'] / 1e9:.1f}B, "
        f"pass {total['pass_tokens'] / 1e9:.2f}B, unique {total['unique_tokens'] / 1e9:.2f}B tokens"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
