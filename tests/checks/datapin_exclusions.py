"""Pin exclusions (data.pin) and the sampler honouring them (data.mix), on the real
July 2026 store (CPU).

NEW_PIN is OLD_PIN plus July 2026 without its blitz buckets (they hold the dev/test games),
never accepting a game whose token_hash is in strat-eval-v1. Checks:
  pin      both pins verify and rebuild bitwise (the old one without exclusions); the new
           one is the old one plus July: its months, and its inventory plus July's
           non-blitz buckets; the hashes file holds the golden's games, all found in July
  index    the new pin's sampler lists every non-blitz July game and no blitz one, and
           every other month's shards as the old pin's sampler does
  draws    one chunk over July alone at a small pool_frac (the golden fills bucket
           prefixes): no blitz or golden game is accepted, the accepted games are exactly
           those of the same draws without the hash mask but the golden ones, and every
           game of a batch is an accepted one
  mask     every non-blitz July shard holding golden games, read by the sampler: exactly
           its golden games and validation leaks weigh 0
  resume   a checkpoint loads only into a sampler of its own exclusions (none included)
  old pin  Allie 2.1's table, history and pin: index, batches and state as REF_MIX's

    PYTHONPATH=src .venv/bin/python tests/checks/datapin_exclusions.py NEW_PIN OLD_PIN [REF_MIX]

(REF_MIX, for the old pin check: data/mix.py of the commit before the exclusions, from git show.)
"""

import functools
import gc
import importlib.util
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from allie import paths
from allie.data import mix as cm
from allie.data import pin as datapin
from allie.data.store import hash64
from allie.data.vocab import BOS, TERM_NORMAL, TERM_OTHER

GOLDEN = paths.DATA / "strat-eval-v1"
TABLE = "table:c8s200f0v4d112-b8276fd781ac"  # Allie 2.1's
BLITZ = 2
PF, CHUNK = 0.0005, 16384


def fmt(code):
    return code // 10000 % 10 - 1


def games(rows):
    """The complete games of packed rows."""
    out = []
    for r in rows:
        b = np.flatnonzero(r == BOS)
        out += [
            r[s:e]
            for s, e in zip(b, [*b[1:], len(r)])
            if e - s > 1 and r[e - 1] in (TERM_NORMAL, TERM_OTHER)
        ]
    return out


def pins(new_path, old_path):
    new, old = (json.loads(Path(p).read_text()) for p in (new_path, old_path))
    datapin.verify(new_path, new["months"])
    datapin.verify(old_path, old["months"])
    ((july, spec),) = new["exclusions"].items()
    ok = july not in old["months"] and set(new["months"]) == {*old["months"], july}
    ok &= new["stores"] == old["stores"] and spec["formats"] == [BLITZ]
    add = Counter({str(b["code"]): b["games"] for b in cm.listed(july, spec)})
    ok &= Counter(old["inventory"]) + add == Counter(new["inventory"])
    datapin.built = functools.cache(datapin.built)
    rebuilt = datapin.pin(old["first"], old["held_out"]) == old
    arg = f"{Path(july).name}:blitz:{spec['hashes']}"
    rebuilt &= datapin.pin(new["first"], new["held_out"], arg) == new
    golden = cm.masked(spec)
    want = {hash64(g) for g in games(np.load(GOLDEN / "strat.npz")["rows"])}
    ok &= set(golden.tolist()) == want
    return {"pin": ok, "rebuilt": rebuilt}, july, spec, golden


def scan(july, golden):
    """(path, code, golden rows, val_leak or golden rows, golden hashes) of July's shards."""

    def one(job):
        code, rel = job
        t = cm.read(Path(july) / rel, ["token_hash", "val_leak"])
        h = t.column("token_hash").to_numpy()
        hit = np.isin(h, golden)
        return (
            str(Path(july) / rel),
            code,
            hit,
            t.column("val_leak").to_numpy() | hit,
            h[hit],
        )

    jobs = [(b["code"], s) for b in cm.listed(july) for s in b["shards"]]
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, jobs))


def index(new_path, old_path, july):
    new, old = (json.loads(Path(p).read_text()) for p in (new_path, old_path))
    s = cm.Sampler("natural", months=new["months"], exclusions=new["exclusions"])
    o = cm.Sampler("natural", months=old["months"])
    ours = lambda path: Path(path).parents[2] == Path(july)
    per = Counter()
    for c, units in s.units.items():
        per[fmt(c)] += sum(n for p, n in units if ours(p))
    stats = json.loads((Path(july) / "stats.json").read_text())["games_by_format"]
    want = {k: v for k, v in enumerate(stats.values()) if v and k != BLITZ}
    rest = {c: [u for u in us if not ours(u[0])] for c, us in s.units.items()}
    ok = {k: v for k, v in per.items() if v} == want
    ok &= {c: us for c, us in rest.items() if us} == o.units
    return {
        "index": ok,
        "july_listed": {datapin.FORMATS[k]: v for k, v in want.items()},
    }


def draws(july, spec, golden):
    """July alone, natural weights: the pin's exclusion, then blitz only on the same draws."""
    log, orig = [], cm.Games.tokens

    def spy(self, i, white, black):
        out = orig(self, i, white, black)
        log.append((int(self.code), hash64(out[0])))
        return out

    new = lambda e: cm.Sampler(
        "natural", 42, months=[july], exclusions={july: e}, pool_frac=PF, chunk=CHUNK
    )
    cm.Games.tokens = spy
    try:
        a = new(spec)
        rows = a.batch(64)
        got, log[:] = list(log), []
        del a
        gc.collect()
        b = new({"formats": spec["formats"]})
        b._accepted(0.0)
        ref = list(log)
        del b
        gc.collect()
    finally:
        cm.Games.tokens = orig
    gold = set(golden.tolist())
    hashes = {h for _, h in got}
    batch = [hash64(g) for g in games(rows)]
    drawn = sum(h in gold for _, h in ref)
    return {
        "no_blitz": all(fmt(c) != BLITZ for c, _ in got),
        "no_golden": not hashes & gold,
        "golden_drawn": drawn,
        "mask_bites": drawn > 0,
        "same_draws": Counter(got) == Counter(x for x in ref if x[1] not in gold),
        "batch_accepted": bool(batch) and set(batch) <= hashes,
        "accepted": {
            datapin.FORMATS[k]: v for k, v in Counter(fmt(c) for c, _ in got).items()
        },
    }


def mask(july, spec, shards):
    """The sampler's natural weights (1 but where masked or leaked) on every non-blitz July
    shard holding golden games."""
    s = cm.Sampler("natural", months=[july], exclusions={july: spec})
    ok, n, rows = True, 0, 0
    for path, code, hit, zero, _ in shards:
        if fmt(code) != BLITZ and hit.any():
            g = s._new(path)
            g.policy(s.fn, 0.0)
            ok &= g.g.code == code and np.array_equal(g.w, np.where(zero, 0, 1.0))
            n, rows = n + 1, rows + int(hit.sum())
    return {"mask": ok, "mask_shards": n, "masked_rows": rows}


def resume(july, spec):
    new = lambda e: cm.Sampler("natural", months=[july], exclusions=e and {july: e})

    def loads(a, b):
        try:
            new(b).load_state_dict(new(a).state_dict())
        except ValueError:
            return False
        return True

    other = {"formats": spec["formats"]}
    refused = ((None, spec), (spec, None), (other, spec), (spec, other))
    ok = loads(spec, spec) and loads(None, None)
    return {"resume": ok and not any(loads(a, b) for a, b in refused)}


def bitwise(old_path, ref_path):
    spec = importlib.util.spec_from_file_location("mix_ref", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    old = json.loads(Path(old_path).read_text())
    kw = {
        "months": old["months"],
        "stores": old["stores"],
        "history": str(Path(old_path).with_name("history-counts.json")),
        "pool_frac": 0.9999997805714286,
        "aux": True,
        "feats": True,
        "chunk": 128,
        "total_rows": 10**6,
    }
    out = []
    for m in (ref, cm):
        s = m.Sampler(TABLE, 42, **kw)
        s._weights(0.0)
        idx = (s.codes, s.units, s.games.tolist(), s.cdf.tolist(), s.infeasible)
        arrays = [a for _ in range(2) for a in (s.batch(4), *s.last.values())]
        out.append((idx, list(s.last), s.state_dict(), arrays))
        del s
        gc.collect()
    (*a, x), (*b, y) = out
    ok = a == b and all(np.array_equal(u, v) for u, v in zip(x, y, strict=True))
    return {"old_pin_bitwise": ok}


def main():
    new_path, old_path, *ref = sys.argv[1:]
    out, july, spec, golden = pins(new_path, old_path)
    shards = scan(july, golden)
    rows = Counter()
    for _, code, hit, _, _ in shards:
        rows[datapin.FORMATS[fmt(code)]] += int(hit.sum())
    found = {int(h) for *_, hs in shards for h in hs}
    out |= {"golden_in_july": found == set(golden.tolist()), "golden_rows": dict(rows)}
    out |= index(new_path, old_path, july)
    out |= draws(july, spec, golden)
    out |= mask(july, spec, shards)
    out["masked_all"] = out["masked_rows"] == sum(rows.values()) - rows["blitz"]
    out |= resume(july, spec)
    out |= bitwise(old_path, ref[0]) if ref else {}
    print(json.dumps(out, indent=1))
    good = all(v for v in out.values() if isinstance(v, bool))
    print("PASS" if good else "FAIL")
    sys.exit(not good)


if __name__ == "__main__":
    main()
