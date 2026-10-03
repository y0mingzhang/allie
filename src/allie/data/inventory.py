"""Pool inventory and recipe tables.

inventory PIN OUT_DIR [TOKEN_MONTH ...]
    Per bucket code of a data.pin selection: pinned games (disk); the full-pool estimate
    (games): every clocked Lichess month >= the pin's first but its held-out one, unbuilt
    months spread by the bucket shares interpolated between the pinned months around them,
    ext stores exact; data.mix tokens per game (12 + plies, cut at the row) from one shard
    per bucket of the token months (default: all pinned) and of every ext store, weighted
    by games (unbuilt months are assumed to share the pinned per-bucket moments). Writes
    OUT_DIR/inventory.json and OUT_DIR/history-counts.json (the full-pool estimate as the
    sampler's --mix-history file).
recipe INVENTORY {disk|full} TOKENS LABEL KIND [KEY=VALUE ...]
    Per-code weights = expected passes over the disk or full pool of a run of TOKENS
    training tokens (packing overshoot included), written content-addressed to
    data.mix.RECIPES/LABEL-<sha12>.json and trained as --mix table:LABEL-<sha12>. Kinds:
      passcap cap=1           expected-pass-capped: equal game draws over {bullet, blitz,
                              rapid, classical} x 200-Elo cells, each at most cap passes of
                              its supply; ultrabullet, correspondence at the pool-average rate
      elo s= f= engine= cap=8 within a format a game is seen 2x as often every +s Elo
                              (s=inf: flat), capped; format slices ~ supply^(1-f);
                              ultrabullet, correspondence at the pool-average rate; engine a
                              fixed share of the tokens
      hand cells= otb= engine= relative weights per format x Elo band (cells =
                              fmt:w,w,w,w/... over <1400, 1400-2000, 2000-2400, >=2400),
                              OTB x otb, engine a fixed share
      topup base= p= [elo= otb=]  control (base: a replay of control on this pool, its realized
                              passes per bucket) with every human game whose stronger player
                              is >= elo (2400) raised to p passes, OTB games only if otb=1;
                              p=0: control as a table
      anneal base= start= [elo=]  the late table of coolNN:table:NAME (start = NN / 100): every
                              human game whose stronger player is >= elo (2400) reaches one
                              pass over the run, the rest control-shaped
    Elo is the stronger player's (bucket max-Elo bin centre); OTB games fall in their
    format's cells.
"""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

from allie.data import mix as cm
from allie.data.history import published

EVAL, OTHER = (1, 2, 3, 4), (0, 5)  # bullet..classical; ultrabullet, correspondence
BANDS = (1400, 2000, 2400)


def index(m):
    y, mo = map(int, m.split("-"))
    return 12 * y + mo


def estimate(month, have):
    """Bucket shares of an unbuilt month: linear in time between the nearest pinned months on each side."""
    x = index(month)
    lo = max((m for m in have if index(m) < x), key=index, default=None)
    hi = min((m for m in have if index(m) > x), key=index, default=None)
    share = lambda m: {c: n / sum(have[m].values()) for c, n in have[m].items()}
    if lo is None or hi is None:
        return share(lo or hi)
    a = (x - index(lo)) / (index(hi) - index(lo))
    s0, s1 = share(lo), share(hi)
    return {c: (1 - a) * s0.get(c, 0) + a * s1.get(c, 0) for c in s0.keys() | s1.keys()}


def sample(d, b):
    """(code, N, N x mean tokens, N x mean squared tokens) of a month's bucket of N games, the
    means from its smallest shard holding >= min(5000, N) games (shards are pre-shuffled), so
    sums over months weight each by its population."""
    last = b["games"] - cm.SHARD_GAMES * (len(b["shards"]) - 1)
    rel = b["shards"][-1] if last >= min(5000, b["games"]) else b["shards"][0]
    moves = cm.read(d / rel, ["moves"]).column("moves")
    t = np.minimum(pc.list_value_length(moves).to_numpy() + 12, cm.ROW).astype(
        np.float64
    )
    n = b["games"]
    return b["code"], n, n * t.mean(), n * (t * t).mean()


def inventory(pin, out, *token_months):
    raw = Path(pin).read_bytes()
    p, out = json.loads(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(m) for m in p["months"]]
    lichess = {d.name: d for d in dirs if not d.name.startswith("20xx")}
    ext = [d for d in dirs if d.name.startswith("20xx")]
    bucket = {d: json.loads((d / "buckets.json").read_text()) for d in dirs}
    disk = {d: {b["code"]: b["games"] for b in bucket[d]} for d in dirs}
    have = {m: disk[d] for m, d in lichess.items()}
    pub = published(sorted({str(d.parent) for d in lichess.values()}))
    pool = sorted(m for m in pub if p["first"] <= m != p["held_out"])
    assert set(lichess) <= set(pool), "pinned months outside the pool"
    games = {}
    for m in pool:
        got = (
            have[m]
            if m in have
            else {c: s * pub[m] for c, s in estimate(m, have).items()}
        )
        for c, g in got.items():
            games[c] = games.get(c, 0) + g
    for d in ext:
        for c, g in disk[d].items():
            games[c] = games.get(c, 0) + g
    token_months = token_months or sorted(lichess)
    jobs = [(lichess[m], b) for m in token_months for b in bucket[lichess[m]]]
    jobs += [(d, b) for d in ext for b in bucket[d]]
    acc = {}
    with ThreadPoolExecutor(16) as ex:
        for code, *v in ex.map(lambda j: sample(*j), jobs):
            acc[code] = acc.get(code, 0) + np.array(v)
    coarse = lambda c, k: (c // 100000, c // 10000 % 10, c // 100 % 100)[:k]
    fallback = {}
    for c, a in acc.items():
        for k in (3, 2, 1):
            fallback[coarse(c, k)] = fallback.get(coarse(c, k), 0) + a
    codes = {}
    for c in sorted(games):
        a = next(
            x
            for x in (acc.get(c), *(fallback.get(coarse(c, k)) for k in (3, 2, 1)))
            if x is not None
        )
        on_disk = sum(disk[d].get(c, 0) for d in dirs)
        codes[str(c)] = dict(
            games=round(games[c]), disk=on_disk, tok=a[1] / a[0], tok2=a[2] / a[0]
        )
    assert {k: v["disk"] for k, v in codes.items() if v["disk"]} == p[
        "inventory"
    ], "disk differs from the pin"
    unbuilt = sorted(set(pool) - set(lichess))
    inv = dict(
        pin=str(pin),
        pin_sha256=hashlib.sha256(raw).hexdigest(),
        pin_digest=p["sha256"],
        months=p["months"],
        unbuilt=unbuilt,
        published={m: pub[m] for m in pool},
        unbuilt_published_games=sum(pub[m] for m in unbuilt),
        token_months=list(token_months),
        disk_games=sum(v["disk"] for v in codes.values()),
        disk_tokens=sum(v["disk"] * v["tok"] for v in codes.values()),
        games=sum(v["games"] for v in codes.values()),
        tokens=sum(v["games"] * v["tok"] for v in codes.values()),
        codes=codes,
    )
    (out / "inventory.json").write_text(json.dumps(inv, indent=1) + "\n")
    hist = dict(
        estimate=f"{len(lichess)} pinned + {len(unbuilt)} interpolated Lichess months, ext exact",
        pin_digest=p["sha256"],
        counts={c: v["games"] for c, v in codes.items()},
    )
    (out / "history-counts.json").write_text(json.dumps(hist, indent=1) + "\n")
    print(
        f"disk {inv['disk_games'] / 1e9:.3f}B games / {inv['disk_tokens'] / 1e9:.1f}B tokens; full pool "
        f"{inv['games'] / 1e9:.3f}B / {inv['tokens'] / 1e9:.1f}B ({len(unbuilt)} months interpolated)"
    )


class Pool:
    """Per-code supply (games), tokens per game and cell keys of an inventory basis."""

    def __init__(self, inv, basis):
        c = inv["codes"]
        self.code = np.array([int(k) for k in c])
        self.games = np.array(
            [v["disk" if basis == "disk" else "games"] for v in c.values()], float
        )
        self.tok = np.array([v["tok"] for v in c.values()])
        self.tok2 = np.array([v["tok2"] for v in c.values()])
        self.src, self.fmt = self.code // 100000, self.code // 10000 % 10 - 1
        hi = self.code // 100 % 100
        self.elo = hi * 100 + 50.0
        self.band = np.searchsorted(BANDS, hi * 100, side="right")
        self.engine = np.isin(self.src, cm.ENGINE)
        self.otb = np.isin(self.src, cm.OTB)
        self.supply = self.games * self.tok  # tokens

    def drawn(self, tokens, w):
        """Game tokens a run of `tokens` training tokens draws at weights w: each 1025-token row discards the
        overshoot of its last game, E[L^2] / 2E[L] for length-biased residuals."""
        n = self.games * w
        return tokens * (1 + (n @ self.tok2) / (2 * (n @ self.tok)) / cm.ROW)


def bisect(f, target, lo, hi, n=200):
    """x in [lo, hi] with f(x) = target, f increasing."""
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def passcap(pool, total, cap=1.0):
    """Equal game draws per {bullet..classical} x 200-Elo cell, each at most cap x its supply."""
    cap = float(cap)
    cell = np.isin(pool.fmt, EVAL) & ~pool.engine
    key = pool.fmt * 100 + pool.code // 100 % 100 // 2
    other = np.isin(pool.fmt, OTHER) & ~pool.engine
    human = pool.supply[~pool.engine].sum()
    keys = np.unique(key[cell])
    g = np.array([pool.games[cell & (key == k)].sum() for k in keys])
    t = np.array([pool.supply[cell & (key == k)].sum() for k in keys]) / g
    w = np.zeros(len(pool.code))
    for _ in range(5):  # the overshoot depends on the mix
        T = pool.drawn(total, w) if w.any() else total
        want = T * (1 - pool.supply[other].sum() / human)
        assert cap * (g * t).sum() >= want, "the cells cannot supply the tokens"
        lam = bisect(
            lambda x: (np.minimum(cap * g, x) * t).sum(), want, 0, cap * g.max()
        )
        per = dict(zip(keys, np.minimum(cap * g, lam) / g))
        w = np.where(cell, [per.get(k, 0) for k in key], 0) + np.where(
            other, T / human, 0
        )
    return w


def slices(budget, supply, f, cap):
    """Format slices ~ supply^(1-f), each at most cap passes, the excess spread over the others."""
    out, free = {}, dict(supply)
    while free:
        share = {k: s ** (1 - f) for k, s in free.items()}
        left = budget - sum(out.values())
        full = {
            k for k in free if left * share[k] / sum(share.values()) > cap * free[k]
        }
        if not full:
            return out | {k: left * share[k] / sum(share.values()) for k in free}
        out |= {k: cap * free.pop(k) for k in full}
    raise AssertionError(f"every format capped: {sum(out.values()):.4g} < {budget:.4g}")


def elo(pool, total, s, f, engine, cap=8.0):
    s, f, engine, cap = map(float, (s, f, engine, cap))
    human = pool.supply[~pool.engine].sum()
    other = np.isin(pool.fmt, OTHER) & ~pool.engine
    w = np.zeros(len(pool.code))
    for _ in range(5):
        T = pool.drawn(total, w) if w.any() else total
        r = T / human
        w = np.where(other, r, 0.0)
        w[pool.engine] = engine * T / pool.supply[pool.engine].sum()
        budget = T * (1 - engine) - r * pool.supply[other].sum()
        sup = {k: pool.supply[(pool.fmt == k) & ~pool.engine].sum() for k in EVAL}
        for k, want in slices(budget, sup, f, cap).items():
            m = (pool.fmt == k) & ~pool.engine
            rule = lambda e0: np.minimum(cap, 2.0 ** ((pool.elo[m] - e0) / s))
            if np.isinf(s):
                w[m] = want / sup[k]
                continue
            assert cap * pool.supply[m].sum() >= want, f"format {k} short"
            e0 = bisect(
                lambda e0: -(pool.supply[m] * rule(e0)).sum(), -want, -5000, 10000
            )
            w[m] = rule(e0)
    return w


def hand(pool, total, cells, otb=1.0, engine=0.0):
    otb, engine = float(otb), float(engine)
    rel = np.zeros((6, 4))
    for part in cells.split("/"):
        k, v = part.split(":")
        rel[int(k)] = [float(x) for x in v.split(",")]
    human = pool.supply[~pool.engine].sum()
    other = np.isin(pool.fmt, OTHER) & ~pool.engine
    base = (
        rel[pool.fmt, pool.band] * np.where(pool.otb, otb, 1.0) * ~pool.engine * ~other
    )
    w = base
    for _ in range(5):
        T = pool.drawn(total, w)
        budget = T * (1 - engine) - T / human * pool.supply[other].sum()
        w = base * budget / (pool.supply @ base) + np.where(other, T / human, 0)
        w[pool.engine] = engine * T / pool.supply[pool.engine].sum()
    return w


def realized(pool, base, minpool, base_sha256):
    """Control's realized passes per code from its replay; buckets retaining fewer than minpool games take the
    supply-weighted passes of their source x format x 200-Elo cell."""
    raw = Path(base).read_bytes()
    assert base_sha256 in (None, hashlib.sha256(raw).hexdigest()), "base replay changed"
    rep = json.loads(raw)["buckets"]
    get = lambda k, d: np.array(
        [rep.get(str(c), {}).get(k, d) for c in pool.code], float
    )
    seen, ok = get("passes", np.nan), get("pool", 0) >= int(minpool)
    cell = pool.src * 10000 + pool.fmt * 100 + pool.code // 100 % 100 // 2
    for k in np.unique(cell[~ok]):
        m = (cell == k) & ok
        fill = (
            np.average(seen[m], weights=pool.supply[m]) if pool.supply[m].sum() else 0.0
        )
        seen[(cell == k) & ~ok] = fill
    return seen


def fit(pool, total, w):
    """w(c) at the scale c whose run draws exactly its tokens."""
    return w(
        bisect(lambda c: pool.supply @ w(c) / pool.drawn(total, w(c)), 1.0, 0.0, 5.0)
    )


def topup(pool, total, base, p, minpool=200, base_sha256=None, elo=2400, otb=0):
    """Control with every human game (OTB only if otb) whose stronger player is >= elo raised to p expected passes, never
    below control's own; the rest, engine included, keeps control's weights, scaled to fit the run."""
    seen, p = realized(pool, base, minpool, base_sha256), float(p)
    top = (pool.code // 100 % 100 * 100 >= int(elo)) & ~pool.engine & (p > 0)
    top &= pool.otb | (not int(otb))
    return fit(pool, total, lambda c: np.where(top, np.maximum(p, seen), c * seen))


def anneal(pool, total, base, start, minpool=200, base_sha256=None, elo=2400):
    """The late table of coolNN:table:NAME, start = NN / 100: after control's first start of the run, every human game
    whose stronger player is >= elo reaches one pass over the whole run (never fewer than control's), the rest keeps
    control's shape, scaled to fit the last 1 - start."""
    seen, s = realized(pool, base, minpool, base_sha256), float(start)
    top = (pool.code // 100 % 100 * 100 >= int(elo)) & ~pool.engine
    late = np.maximum(seen, (1 - s * seen) / (1 - s))
    return fit(pool, total, lambda c: np.where(top, late, c * seen))


def summary(pool, w, total):
    n = pool.games * w * pool.tok
    T = n.sum()
    out = dict(
        drawn_tokens=T,
        training_tokens=total,
        max_passes=float(w[pool.games > 0].max()),
        unique_tokens=float((pool.supply * np.minimum(w, 1)).sum()),
        engine_share=float(n[pool.engine].sum() / T),
        otb_share=float(n[pool.otb].sum() / T),
        formats={},
    )
    names = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")
    labels = ("<1400", "1400-2000", "2000-2400", ">=2400")
    for k, name in enumerate(names):
        m = (pool.fmt == k) & ~pool.engine
        cells = {}
        for b, label in enumerate(labels):
            mb = m & (pool.band == b)
            if pool.supply[mb].sum():
                cells[label] = dict(
                    share=float(n[mb].sum() / T),
                    passes=float(n[mb].sum() / pool.supply[mb].sum()),
                )
        out["formats"][name] = dict(share=float(n[m].sum() / T), cells=cells)
    return out


def recipe(inv_path, basis, tokens, label, kind, *kv):
    raw = Path(inv_path).read_bytes()
    pool, total = Pool(json.loads(raw), basis), float(tokens)
    args = dict(x.split("=", 1) for x in kv)
    w = dict(passcap=passcap, elo=elo, hand=hand, topup=topup, anneal=anneal)[kind](
        pool, total, **args
    )
    keep = pool.games > 0
    assert np.isfinite(w).all() and (w >= 0).all()
    got = pool.supply @ w / pool.drawn(total, w)
    assert abs(got - 1) < 1e-6, f"drawn tokens off by {got - 1:.2e}"
    hist = Path(inv_path).with_name("history-counts.json")
    body = dict(
        label=label,
        kind=kind,
        args=args,
        basis=basis,
        training_tokens=total,
        inventory_sha256=hashlib.sha256(raw).hexdigest(),
        pin_digest=json.loads(raw)["pin_digest"],
        history_sha256=hashlib.sha256(hist.read_bytes()).hexdigest()
        if basis == "full"
        else None,
        summary=summary(pool, w, total),
        weights={str(c): float(x) for c, x in zip(pool.code[keep], w[keep])},
    )
    text = json.dumps(body, indent=1) + "\n"
    name = f"{label}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
    path = cm.RECIPES / f"{name}.json"
    cm.RECIPES.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text)
    assert path.read_text() == text
    s = body["summary"]
    print(
        f"table:{name}  max passes {s['max_passes']:.2f}, engine {s['engine_share']:.3f}, otb {s['otb_share']:.3f}"
    )
    for f, v in s["formats"].items():
        cells = "  ".join(
            f"{b} {c['share']:.3f}/{c['passes']:.2f}x" for b, c in v["cells"].items()
        )
        print(f"  {f:14} {v['share']:.3f}  {cells}")


if __name__ == "__main__":
    dict(inventory=inventory, recipe=recipe)[sys.argv[1]](*sys.argv[2:])
