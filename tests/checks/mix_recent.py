"""data.mix recentNN(YYYY-MM):POLICY, a recent-only tail: recent90(2024-01):TABLE on a real pin (CPU).

Before 90% of training the composed policy's weights equal the table's on every shard (old and recent months), after it
they equal the table's on recent Lichess months and external sources and are 0 on Lichess months before 2024-01; the
bucket caps (draw weights) stay the table's in both phases; batches draw across the switch, and after it every accepted
Lichess game comes from 2024 or later. The months / stores / history of a 1e17 control run of the next run's recipe.

    PYTHONPATH=<data.mix overlay> .venv/bin/python tests/checks/mix_recent.py RUN_CONFIG.json
"""

import json
import sys

import numpy as np

from allie.data import mix as cm

TABLE = "table:c8s200f0v4-fcfbf8858a28"


def main():
    a = json.loads(open(sys.argv[1]).read())["args"]
    kw = dict(
        stores=a["mix_stores"].split(","),
        months=a["mix_months"].split(","),
        history=a["mix_history"],
        pool_frac=a["mix_pool_frac"],
        row=16385,
    )
    rows = 64
    s = cm.Sampler(f"recent90(2024-01):{TABLE}", 42, total_rows=rows, **kw)
    t = cm.Sampler(TABLE, 42, total_rows=rows, **kw)
    ok = cm.marks(s.policy) == (0.0, 0.9) and cm.recent(TABLE) is None
    s._weights(0.0), t._weights(0.0)
    caps0 = np.array_equal(s.caps, t.caps)
    s._weights(0.95), t._weights(0.95)
    caps1 = np.array_equal(s.caps, t.caps)
    rng = np.random.default_rng(0)
    paths = [p for c in s.codes for p, _ in s.units[c]]
    pick = rng.choice(len(paths), 40, replace=False)
    before = after = True
    old = recent = 0
    for i in pick:
        sh = cm.Shard(paths[i])
        w_t = np.asarray(t.fn(sh.g, 0.0)[0], float) * np.ones(sh.g.n)
        w0 = np.asarray(s.fn(sh.g, 0.0)[0], float) * np.ones(sh.g.n)
        w1 = np.asarray(s.fn(sh.g, 0.9)[0], float) * np.ones(sh.g.n)
        before &= np.array_equal(w0, w_t)
        is_old = sh.g.src == 0 and sh.g.month < "2024-01"
        old, recent = old + is_old, recent + (not is_old)
        after &= np.array_equal(w1, 0 * w_t) if is_old else np.array_equal(w1, w_t)
    # draw before and after the switch
    seen = []
    orig = cm.Shard.policy

    def spy(self, fn, ph):
        orig(self, fn, ph)
        if ph >= 0.9:
            seen.append((self.g.month, self.g.src, float(np.max(self.w, initial=0))))

    cm.Shard.policy = spy
    for _ in range(2):  # before the switch
        s.batch(1)
    s.seen, s.pool = (
        int(0.95 * rows),
        [],
    )  # past it: the next chunk is drawn at progress 0.95
    for _ in range(4):
        s.batch(1)
    cm.Shard.policy = orig
    drawn = all(src != 0 or m >= "2024-01" or w == 0 for m, src, w in seen)
    drawn &= any(src == 0 and m >= "2024-01" and w > 0 for m, src, w in seen)
    print(json.dumps(dict(caps_equal=[caps0, caps1], before_equal_table=bool(before), after_old_zero_recent_table=bool(after),
                          shards_checked=dict(old=int(old), recent=int(recent)), tail_shards=len(seen), tail_old_weights_zero=drawn)))  # fmt: skip
    good = (
        caps0
        and caps1
        and before
        and after
        and old
        and recent
        and seen
        and drawn
        and ok
    )
    print("PASS" if good else "FAIL")
    sys.exit(not good)


if __name__ == "__main__":
    main()
