"""Stockfish MultiPV pass of the calibration eval: every legal move's eval at one depth (MultiPV 256,
depth 10, one thread), centipawns from the mover's view, mates and anything beyond capped at 1000.
Gives losses, win-probability losses, engine ranks and percentiles for any move distribution.

python calib_mpv.py POSITIONS.npz OUT_DIR [--shard i --shards n --workers 12 --depth 10]
"""

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from allie.data.vocab import MOVE_ID, MOVES

from play import SF, UCI

CAP = 1000


def multipv(sf, moves, depth):
    """{move token: cp} for every legal move, from the final-depth lines."""
    sf.send("position startpos moves " + " ".join(moves), f"go depth {depth}")
    out = {}
    while not (line := sf.p.stdout.readline()).startswith("bestmove"):
        f = line.split()
        if (
            len(f) < 4
            or f[0] != "info"
            or f[1:3] != ["depth", str(depth)]
            or any(x.endswith("bound") for x in f)
            or "pv" not in f
        ):
            continue
        k = f.index("score")
        cp = (
            int(f[k + 2])
            if f[k + 1] == "cp"
            else CAP * (1 if int(f[k + 2]) > 0 else -1)
        )
        out[MOVE_ID[f[f.index("pv") + 1]]] = max(-CAP, min(CAP, cp))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("out")
    p.add_argument(
        "--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--depth", type=int, default=10)
    a = p.parse_args()
    z = np.load(a.positions)
    off, tokens, meta = z["offsets"], z["tokens"], z["meta"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def one(i):
        if getattr(local, "sf", None) is None:
            local.sf = UCI(
                [SF], [("Threads", 1), ("Hash", 64), ("MultiPV", 256)], depth=None
            )
        moves = [MOVES[t - 378] for t in tokens[off[i] + 11 : off[i + 1]]]
        return multipv(local.sf, moves, a.depth)

    with ThreadPoolExecutor(a.workers) as pool:
        for lo in list(range(0, len(meta), 1024))[a.shard :: a.shards]:
            dst = out / f"{lo:06d}.npz"
            if dst.exists():
                continue
            idx = np.arange(lo, min(lo + 1024, len(meta)))
            res = list(pool.map(one, idx))
            np.savez(dst.with_suffix(".partial.npz"), index=idx, offsets=np.cumsum([0] + [len(r) for r in res]),
                     tokens=np.array([t for r in res for t in r], np.int16),
                     cp=np.array([c for r in res for c in r.values()], np.int16))  # fmt: skip
            dst.with_suffix(".partial.npz").replace(dst)
            print(
                f"{dst.name}: {len(idx)} positions, {np.mean([len(r) for r in res]):.1f} moves",
                flush=True,
            )


if __name__ == "__main__":
    main()
