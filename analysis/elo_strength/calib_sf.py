"""Stockfish pass of the calibration eval: per position, the root eval and each candidate move's eval
(go depth D searchmoves m, so both come from one search depth), in centipawns from the mover's view,
mates and anything beyond capped at 1000. Candidates: the human move and the moves covering 99% of
every distribution in the search file (legal policy, coverage, MCTS visits; at most 10 each), and the
top 3 by coverage-128 Q. --skip leaves out moves an earlier pass scored (the root is rescored).

python calib_sf.py POSITIONS.npz SEARCH_DIR|human OUT_DIR [--shard i --shards n --workers 16 --depth 12]
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from allie.data.vocab import MOVES

from play import stockfish

CAP = 1000


def cover(p, mass=0.99, most=10):
    order = np.argsort(-p)
    k = int(np.searchsorted(np.cumsum(p[order]), mass)) + 1
    return set(order[: min(k, most)].tolist())


def candidates(s, j, human, done=()):
    """Move tokens to score: the human move, the 99% covers of every distribution in the file, the
    top 3 by coverage-128 Q; minus those already scored (`done`)."""
    if not len(s["legal"]):
        return [] if human in done else [human]
    a, b = s["offsets"][j], s["offsets"][j + 1]
    legal = s["legal"][a:b]
    dists = [
        s[k][a:b].astype(float)
        for k in ("prior", "cov_p8", "cov_p32", "cov_p128")
        if k in s
    ]
    dists += [
        s[k][a:b].astype(float)
        for k in ("mcts_n8", "mcts_n32", "mcts_n128", "mcts_n512")
        if k in s
    ]
    slots = set()
    for d in dists:
        if d.sum() > 0:
            slots |= cover(d / d.sum())
    if "cov_q128" in s:
        slots |= set(np.argsort(-s["cov_q128"][a:b])[:3].tolist())
    moves = {int(legal[k]) for k in slots} | {human}
    return sorted(moves - set(done))


def score(sf, moves, uci, depth):
    """(root cp, [cp of each move in uci]); the root is the best of all."""
    clip = lambda cp: max(-CAP, min(CAP, cp))
    root = clip(sf.go(moves, depth=depth)[1])
    out = []
    for m in uci:
        sf.send(
            "position startpos moves " + " ".join(moves),
            f"go depth {depth} searchmoves {m}",
        )
        cp = None
        while not (line := sf.p.stdout.readline()).startswith("bestmove"):
            if line.startswith("info") and " multipv 1 " in line and " score " in line:
                f = line.split()
                k = f.index("score")
                cp = (
                    int(f[k + 2])
                    if f[k + 1] == "cp"
                    else CAP * (1 if int(f[k + 2]) > 0 else -1)
                )
        out.append(clip(cp))
    return max([root, *out]), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("search")
    p.add_argument("out")
    p.add_argument(
        "--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="earlier output dirs: their moves are not rescored",
    )
    a = p.parse_args()
    z = np.load(a.positions)
    off, tokens, meta = z["offsets"], z["tokens"], z["meta"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    done = {}  # position -> move tokens scored by earlier passes (--skip)
    for d in a.skip:
        for f in Path(d).glob("[0-9]*.npz"):
            if ".partial" in f.name:
                continue
            q = np.load(f)
            for n, i in enumerate(q["index"]):
                done.setdefault(int(i), set()).update(
                    q["tokens"][q["offsets"][n] : q["offsets"][n + 1]].tolist()
                )

    def one(args):
        s, j, i = args
        if getattr(local, "sf", None) is None:
            local.sf = stockfish()
        moves = [MOVES[t - 378] for t in tokens[off[i] + 11 : off[i + 1]]]
        todo = candidates(s, j, int(meta[i, 8]), done.get(int(i), ()))
        best, cps = score(local.sf, moves, [MOVES[t - 378] for t in todo], a.depth)
        return best, todo, cps

    mine = [f"{lo:06d}.npz" for lo in range(0, len(meta), 1024)][a.shard :: a.shards]
    with ThreadPoolExecutor(a.workers) as pool:
        while todo := [n for n in mine if not (out / n).exists()]:
            human = a.search == "human"  # no search files: score the human move only
            ready = [n for n in todo if human or (Path(a.search) / n).exists()]
            if not ready:  # the search or policy pass is still writing: wait for it
                time.sleep(60)
                continue
            n = ready[0]
            if human:
                lo = int(n[:6])
                s = dict(
                    index=np.arange(lo, min(lo + 1024, len(meta))),
                    offsets=np.zeros(1025, int),
                    legal=np.zeros(0, int),
                )
            else:
                s = dict(np.load(Path(a.search) / n))
            res = list(pool.map(one, [(s, j, i) for j, i in enumerate(s["index"])]))
            tmp = (out / n).with_suffix(".partial.npz")
            np.savez(tmp, index=s["index"], best=np.array([r[0] for r in res], np.int16),
                     tokens=np.concatenate([r[1] for r in res]).astype(np.int16),
                     cp=np.concatenate([r[2] for r in res]).astype(np.int16),
                     offsets=np.cumsum([0] + [len(r[1]) for r in res]))  # fmt: skip
            tmp.replace(out / n)
            print(
                f"{n}: {len(res)} positions, {np.mean([len(r[1]) for r in res]):.1f} moves each",
                flush=True,
            )


if __name__ == "__main__":
    main()
