"""Policy tokens per mix pass, per built month and bucket group, with chessmix's own shard reader and policy weights.

Usage: pass_tokens.py POLICY OUT.json MONTH_DIR [MONTH_DIR ...]. The sampler draws each game in proportion to its
policy weight w (validation leaks 0), so one pass of the mix is sum(w * tokens) and each game is seen w times per
pass (up4 experts and otb_x4 games ~4x). unique = sum over games with w > 0 of their tokens (distinct content);
tokens per game are chessmix's (11 header + moves + termination, cut at the 1025-token row). Groups: expert (Lichess,
max Elo >= 2400; >= 2600 also split out), general (other Lichess), otb and engine (ext-v1 sources).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chessmix as cm


def group(g):
    top = np.maximum(g.welo, g.belo)
    return np.select(
        [np.isin(g.src, cm.OTB), np.isin(g.src, cm.ENGINE), top >= 2600, top >= 2400],
        ["otb", "engine", "expert2600", "expert2400"],
        "general",
    )


def shard(args):
    path, policy = args
    s = cm.Shard(path)
    s.policy(cm.POLICIES[policy] if policy in cm.POLICIES else cm.compose(policy), 0.0)
    n = np.minimum(np.diff(s.g.off) + 12, cm.ROW)
    grp = np.broadcast_to(group(s.g), s.g.n)
    out = {}
    for k in np.unique(grp):
        sel = grp == k
        w, t = s.w[sel], n[sel]
        out[str(k)] = (
            int(sel.sum()),
            int(t.sum()),
            float((w * t).sum()),
            float(((w > 0) * t).sum()),
        )
    return s.g.month, out


def main(policy, out, *months):
    jobs = [
        (str(p), policy)
        for m in months
        for p in sorted(Path(m).glob("games/b*/shard-*.parquet"))
    ]
    rows = {}
    keys = ("games", "raw_tokens", "pass_tokens", "unique_tokens")
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as ex:
        for month, got in ex.map(shard, jobs, chunksize=8):
            for k, v in got.items():
                r = rows.setdefault(month, {}).setdefault(k, dict.fromkeys(keys, 0))
                for key, x in zip(keys, v):
                    r[key] += x
    groups = sorted({k for m in rows.values() for k in m})
    by_group = {
        k: {key: sum(m.get(k, {}).get(key, 0) for m in rows.values()) for key in keys}
        for k in groups
    }
    for v in by_group.values():
        v["seen_per_pass"] = (
            v["pass_tokens"] / v["unique_tokens"] if v["unique_tokens"] else 0.0
        )
    total = {key: sum(v[key] for v in by_group.values()) for key in keys}
    Path(out).write_text(
        json.dumps(
            dict(
                policy=policy,
                total=total,
                groups=by_group,
                months=dict(sorted(rows.items())),
            ),
            indent=1,
        )
        + "\n"
    )
    print(
        f"{policy}: {len(rows)} months, {total['games'] / 1e9:.3f}B games, raw {total['raw_tokens'] / 1e9:.1f}B, "
        f"pass {total['pass_tokens'] / 1e9:.2f}B, unique {total['unique_tokens'] / 1e9:.2f}B tokens"
    )
    for k, v in by_group.items():
        print(
            f"  {k}: unique {v['unique_tokens'] / 1e9:.3f}B, pass {v['pass_tokens'] / 1e9:.3f}B, "
            f"seen {v['seen_per_pass']:.2f}x per pass"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
