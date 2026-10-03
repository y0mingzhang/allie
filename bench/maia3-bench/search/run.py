"""Main's frozen search over the Maia-3 benchmark positions, one resident SGLang model per job.

Positions go in a stratified order (round-robin over the four blitz bands, random within each),
so the first 4n are n per band and --limit only ever extends the set. Every chunk of 1024 is
written once and skipped on requeue. Per position it keeps CE, top-1, new NN nodes and enough of
the tree output (root heads, legal logits, backed-up Q) to re-apply another output calibration
without the GPU.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import adapter


def order(cell, seed=20260924):
    rng = np.random.default_rng(seed)
    bands = [rng.permutation(np.flatnonzero(cell == c)) for c in (4, 5, 6, 7)]
    n = min(map(len, bands))
    return np.stack([b[:n] for b in bands], 1).ravel()


def ship_oracle(export):
    from search.runtime import ShipOracle

    return ShipOracle(export)


def moe_oracle(checkpoint, source=None, slots=1 << 18):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "moe-oracle"))
    from oracle import SOURCE, MoEOracle

    return MoEOracle(checkpoint, source or SOURCE, slots=slots, rows=min(slots, 1 << 17))


class Capture:
    """The oracle, keeping each batch's root logits for the saved outputs."""

    def __init__(self, oracle):
        self.oracle = oracle

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def handles(self, *a):
        h = self.oracle.handles(*a)
        self.root = np.asarray(h.root_logits)
        return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", help="SGLang export of a dense ship-recipe checkpoint")
    p.add_argument("--moe", help="MoE checkpoint, served by moe-oracle/oracle.py instead")
    p.add_argument("--source", help="the MoE checkpoint's training source (default: the sweep's)")
    p.add_argument("--slots", type=int, default=1 << 18, help="MoE tree-cache key/value slots")
    p.add_argument("--out", required=True)
    p.add_argument("--budgets", default="legal,8,25,128,460,adaptive")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--set", choices=("bench", "dev"), default="bench")
    p.add_argument("--chunk", type=int, default=1024)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.moe:  # chunks are skipped by path: bind the folder to one checkpoint, source and position set
        st = Path(a.moe).resolve().stat()
        want = dict(model=str(Path(a.moe).resolve()), inode=st.st_ino, size=st.st_size,
                    mtime_ns=st.st_mtime_ns, source=a.source, set=a.set)  # fmt: skip
        m = out / "manifest.json"
        if m.exists():
            assert json.loads(m.read_text()) == want, (m, want)
        else:
            m.write_text(json.dumps(want, indent=1) + "\n")
    P = adapter.positions() if a.set == "bench" else adapter.dev_positions()
    ix = order(np.array([q["cell"] for q in P]))[: a.limit]
    t0 = time.monotonic()
    oracle = Capture(moe_oracle(a.moe, a.source, a.slots) if a.moe else ship_oracle(a.export))
    print(f"startup {time.monotonic() - t0:.0f}s", flush=True)
    engine = adapter.Bench(
        oracle, batch_size=128, threads=min(8, len(os.sched_getaffinity(0)))
    )
    for budget in a.budgets.split(","):
        b = budget if budget in ("legal", "adaptive") else int(budget)
        folder = out / budget
        folder.mkdir(exist_ok=True)
        for lo in range(0, len(ix), a.chunk):
            dst = folder / f"{lo:06d}.npz"
            if dst.exists():
                continue
            part = ix[lo : lo + a.chunk]
            t = time.monotonic()
            res, roots = [], []
            for blo in range(0, len(part), 128):
                res += engine.predict([P[i] for i in part[blo : blo + 128]], b)
                roots.append(oracle.root[:, 2350:2416])
            dt = time.monotonic() - t
            hit = [r["tokens"].index(P[i]["target"]) for r, i in zip(res, part)]
            prob = np.array([r["probabilities"][h] for r, h in zip(res, hit)])
            top1 = np.array(
                [int(np.argmax(r["probabilities"])) == h for r, h in zip(res, hit)]
            )
            cat = lambda k: np.concatenate([r[k] for r in res])
            tmp = dst.with_suffix(".partial.npz")
            np.savez(
                tmp,
                index=part,
                ce=-np.log(prob),
                top1=top1,
                target_slot=np.array(hit),
                nodes=np.array([r["nodes"] for r in res]),
                simulations=np.array([r["simulations"] for r in res]),
                prefill=np.array([r["prefill_tokens"] for r in res]),
                offsets=np.cumsum([0] + [len(r["tokens"]) for r in res]),
                probabilities=cat("probabilities"),
                prior=cat("legal_prior"),
                values=cat("values"),
                root_heads=np.concatenate(roots),
                seconds=dt,
            )
            tmp.replace(dst)
            n = np.array([r["nodes"] for r in res])
            print(
                json.dumps(
                    dict(
                        budget=budget,
                        done=lo + len(part),
                        of=len(ix),
                        seconds=round(dt, 1),
                        nodes=round(float(n.mean()), 1),
                        ce=round(float(-np.log(prob).mean()), 4),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
