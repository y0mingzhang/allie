"""data.mix recentNN(YYYY-MM[,p=P]):POLICY, the bucket-draw-preserving recent tail, on a real pin (CPU).

For the hard (p = 1) and a soft (p = 0.75) tail at 90% of training over Allie 2.0's table: before the switch the
weights equal the table's on every sampled shard and the batches and sampler states equal the plain table run's
bitwise; after it the weights, bucket caps and draw distribution are still the table's (every cell's and store's
share of game draws unchanged, checked on replayed draws; token shares are not checked), each split Lichess bucket
draws its months since YYYY-MM at p and its older ones at 1 - p (0 under the hard tail), external stores and buckets
without both groups draw as before; a table run's checkpoint from before the switch resumes under the tail policy and
continues exactly as a tail run from scratch, one from past it is refused; same_until holds before the switch and not
after. Months, stores and history come from a run's resume config.

    .venv/bin/python tests/checks/mix_recent.py RUN_CONFIG.json
"""

import json
import sys

import numpy as np

from allie.data import mix as cm

TABLE = "table:c8s200f0v4-fcfbf8858a28"
SINCE = "2024-01"
HARD, SOFT = f"recent90({SINCE}):{TABLE}", f"recent90({SINCE},p=0.75):{TABLE}"
ROWS, CHUNK = 40, 64  # rows of a run and draws per chunk: the switch at row 36
NAMES = (
    "ultrabullet",
    "bullet",
    "blitz",
    "rapid",
    "classical",
    "correspondence",
    "other",
)
BANDS = ("<1400", "1400-2000", "2000-2400", ">=2400")


def refused(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


def grammar():
    ok = cm.tail(HARD) == (0.9, SINCE, 1.0) and cm.tail(SOFT) == (0.9, SINCE, 0.75)
    ok &= cm.tail(TABLE) is None and cm.marks(HARD) == cm.marks(SOFT) == (0.0, 0.9)
    ok &= cm.same_until(HARD, TABLE, 0.9) and cm.same_until(HARD, SOFT, 0.9)
    ok &= cm.same_until(f"{HARD}+otb_x2", f"{TABLE}+otb_x2", 0.5)
    ok &= not cm.same_until(HARD, TABLE, 0.91) and not cm.same_until(HARD, SOFT, 0.95)
    ok &= not cm.same_until(HARD, f"{TABLE}+otb_x2", 0.5)
    bad = (
        f"cool80:{HARD}",  # a tail inside a cooldown would never switch
        f"recent95({SINCE}):{HARD}",
        f"recent90({SINCE},p=0):{TABLE}",
        f"recent90({SINCE},p=1.5):{TABLE}",
        f"{TABLE}+recent90({SINCE}):otb_x2",
    )
    return ok and all(refused(lambda n=n: (cm.cool(n), cm.tail(n))) for n in bad)


def cell(code):
    """Format x Elo band of a Lichess bucket (the stronger player's 100-Elo bin), or the external source."""
    src, fmt, hi = code // 100000, code // 10000 % 10 - 1, code // 100 % 100 * 100
    if src:
        return "otb" if src in cm.OTB else "engine"
    return (
        f"{NAMES[fmt]}/{BANDS[np.searchsorted((1400, 2000, 2400), hi, side='right')]}"
    )


def weights(s, t, rng):
    """The tail policy's weights equal the table's on sampled shards in both phases."""
    same = True
    for i in rng.choice(len(t.paths), 24, replace=False):
        g = cm.Shard(t.paths[i]).g
        w = np.asarray(t.fn(g, 0.0)[0], float) * np.ones(g.n)
        for ph in (0.0, 0.9):
            same &= np.array_equal(np.asarray(s.fn(g, ph)[0], float) * np.ones(g.n), w)
    return same


def draws(s, t):
    """Bucket caps and the draw distribution are the table's in both phases."""
    t._weights(0.0)
    ok = True
    for ph in (0.0, 0.9):
        s._weights(ph)
        ok &= np.array_equal(s.caps, t.caps) and np.array_equal(s.cdf, t.cdf)
    return ok


def split(s):
    """The tail's pools: Lichess buckets only, recent / older by month, together the bucket's shards."""
    ok = all(c // 100000 == 0 for c in s.split)
    for c, (recent, old) in s.split.items():
        ok &= all(cm.month(p) >= SINCE for p, _ in recent)
        ok &= all(cm.month(p) < SINCE for p, _ in old)
        ok &= sorted(recent + old) == sorted(s.units[c])
    lichess = [c for c in s.codes if c // 100000 == 0]
    unsplit = [c for c in lichess if c not in s.split]
    return ok, dict(
        split=len(s.split),
        lichess_unsplit=len(unsplit),
        external=len(s.codes) - len(lichess),
    )


def shares(s, chunks, seed):
    """Tail draws replayed without reading shards: per cell and store within 5 sigma of the table's draw
    distribution; each split bucket's recent fraction at the tail's p (old months 0 under the hard tail)."""
    s._weights(0.9)
    rng, cursor = np.random.default_rng(seed), {}
    per, group = np.zeros(len(s.codes)), {}
    for _ in range(chunks):
        for k, pieces in s._chunk(rng, cursor):
            for path, _, idx in pieces:
                per[k] += len(idx)
                if s.codes[k] in s.split:
                    key = (cell(s.codes[k]), cm.month(path) >= SINCE)
                    group[key] = group.get(key, 0) + len(idx)
    cells = np.array([cell(c) for c in s.codes])
    mass = np.diff(s.cdf, prepend=0.0) * chunks * CHUNK
    within = per.sum() == chunks * CHUNK
    for c in dict.fromkeys(cells):
        want, got = mass[cells == c].sum(), per[cells == c].sum()
        within &= abs(got - want) <= 5 * np.sqrt(want) + 1
    p, frac, ok = s.tail[2], {}, True
    for c in sorted({c for c, _ in group}):
        recent, old = group.get((c, True), 0), group.get((c, False), 0)
        frac[c] = round(recent / (recent + old), 3)
        if p == 1:
            ok &= old == 0
        elif recent + old >= 400:
            ok &= abs(frac[c] - p) <= 4 * np.sqrt(p * (1 - p) / (recent + old)) + 0.01
    return dict(
        cells_within_5sigma=bool(within), recent_frac=frac, recent_frac_ok=bool(ok)
    )


def strip(state):
    return {k: v for k, v in state.items() if k != "policy"}


def run(policy, kw):
    """A tail run from scratch against a table run switched to the tail at a checkpoint before the switch: batches
    and states bitwise equal before the switch, the switched run continues as the tail run after it, the table run
    itself diverges, and a checkpoint from past the switch is refused. Returns these and the shards the tail read."""
    new = lambda p: cm.Sampler(p, 42, total_rows=ROWS, chunk=CHUNK, **kw)
    a, b = new(policy), new(TABLE)
    before = all(np.array_equal(a.batch(4), b.batch(4)) for _ in range(2))
    before &= strip(a.state_dict()) == strip(b.state_dict())
    a.seen = b.seen = int(0.9 * ROWS)  # the next chunk is drawn at or past the switch
    c = new(policy)
    c.load_state_dict(b.state_dict())
    shards, orig = [], cm.Shard.policy

    def spy(self, fn, ph):
        orig(self, fn, ph)
        if ph >= 0.9:
            shards.append((self.g.code, self.g.month))

    cm.Shard.policy = spy
    ra, rb, rc = ([s.batch(4) for _ in range(3)] for s in (a, b, c))
    cm.Shard.policy = orig
    after = all(np.array_equal(x, y) for x, y in zip(ra, rc))
    after &= strip(a.state_dict()) == strip(c.state_dict())
    bites = not all(np.array_equal(x, y) for x, y in zip(ra, rb))
    try:
        new(policy).load_state_dict(b.state_dict())
        late = False
    except ValueError:
        late = True
    recent = [m >= SINCE for code, m in shards if code in a.split]
    return dict(
        before_bitwise=before, switch_bitwise=after, tail_bites=bites, late_refused=late
    ), recent


def main():
    a = json.loads(open(sys.argv[1]).read())["args"]
    kw = dict(
        stores=a["mix_stores"].split(","),
        months=a["mix_months"].split(","),
        history=a["mix_history"],
        pool_frac=a["mix_pool_frac"],
    )
    t = cm.Sampler(TABLE, 42, total_rows=ROWS, chunk=CHUNK, **kw)
    out, rng = dict(grammar=grammar()), np.random.default_rng(0)
    for name, policy in (("hard", HARD), ("soft", SOFT)):
        s = cm.Sampler(policy, 42, total_rows=ROWS, chunk=CHUNK, **kw)
        ok, counts = split(s)
        r = dict(
            weights_table=weights(s, t, rng),
            caps_cdf_table=draws(s, t),
            split_ok=ok,
            buckets=counts,
        )
        r |= shares(s, 2000, 1)
        checks, recent = run(policy, kw)
        r |= checks | dict(
            tail_split_shards=dict(recent=sum(recent), old=len(recent) - sum(recent))
        )
        r["tail_shards_ok"] = bool(recent) and (
            all(recent) if name == "hard" else not all(recent) and any(recent)
        )
        out[name] = r
    print(json.dumps(out, indent=1))
    flags = lambda r: [v for k, v in r.items() if isinstance(v, bool)]
    good = out["grammar"] and all(all(flags(out[k])) for k in ("hard", "soft"))
    print("PASS" if good else "FAIL")
    sys.exit(not good)


if __name__ == "__main__":
    main()
