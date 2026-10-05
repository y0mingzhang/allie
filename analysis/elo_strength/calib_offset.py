"""The policy with the mover's header rating shifted by DELTA (the bot asking for a stronger or weaker
player than its target), on the live bot's fast CPU backend (int8): per game and mover, one cache per
shift, extended position by position. Writes calib_search.py-format chunks: index, legal, offsets,
prior (no shift), heads, off_p{tag} (tag: m200 / p100 ... for -200 / +100).

python calib_offset.py POSITIONS.npz OUT_DIR [--max-bin 2000 --shard i --shards n --threads 8]
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import torch
from allie.data.vocab import MOVE_ID, MOVES
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import HEADER, START, advance

MODEL = "/data/group_data/dei-group/yimingz3/allie/lichess/allie-2.0-annealed"
DELTAS = (-200, -100, 100, 200, 300, 400)


def tag(d):
    return ("m" if d < 0 else "p") + str(abs(d))


def shifted(seq, white, delta):
    """seq with the white (or black) rating in the header moved by delta, within 0-9999."""
    s = seq.copy()
    lo = 3 if white else 7
    elo = int("".join(map(str, s[lo : lo + 4])))
    s[lo : lo + 4] = [int(c) for c in f"{min(max(elo + delta, 0), 9999):04d}"]
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("out")
    p.add_argument("--max-bin", type=int, default=2000)
    p.add_argument(
        "--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=0, help="positions per chunk (a timing test)"
    )
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    z = np.load(a.positions)
    off, tokens, feats, meta = z["offsets"], z["tokens"], z["feats"], z["meta"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(MODEL, "cpu", torch.bfloat16, None, True, None, a.threads)
    assert model.fast is not None
    keep = np.flatnonzero(meta[:, 3] <= a.max_bin)
    for lo in list(range(0, len(keep), a.chunk))[a.shard :: a.shards]:
        dst = out / f"{lo:06d}.npz"
        if dst.exists():
            continue
        t0 = time.monotonic()
        idx = keep[lo : lo + a.chunk][: a.limit or None]
        groups = defaultdict(list)  # (game, mover) -> positions, shortest prefix first
        for i in idx:
            k = off[i + 1] - off[i] - HEADER
            groups[meta[i, 0], meta[i, 1], k % 2].append(i)
        z_at, ntok = {}, 0
        for (_, _, black), members in groups.items():
            members.sort(key=lambda i: off[i + 1] - off[i])
            longest = members[-1]
            seq = tokens[off[longest] : off[longest + 1]].astype(np.int64)
            seqs = [seq] + [shifted(seq, not black, d) for d in DELTAS]
            boards = [START] * HEADER
            for t in seq[HEADER:]:
                boards.append(advance(boards[-1], int(t)))
            b = torch.tensor(np.frombuffer(b"".join(boards), np.uint8).reshape(-1, 68))
            f = torch.tensor(
                feats[off[longest] : off[longest + 1]], dtype=torch.float32
            )
            caches, prev = [Cache(model) for _ in seqs], 0
            for i in members:
                n = off[i + 1] - off[i]
                items = [
                    (c, torch.tensor(s[prev:n]), f[prev:n], b[prev:n])
                    for c, s in zip(caches, seqs)
                ]
                z_at[i] = step(model, items).double().numpy()
                ntok += len(seqs) * (n - prev)
                prev = n
        legal, cols, heads = [], defaultdict(list), []
        for i in idx:
            board = chess.Board()
            for t in tokens[off[i] + HEADER : off[i + 1]]:
                board.push_uci(MOVES[t - 378])
            ids = np.array([MOVE_ID[m.uci()] for m in board.legal_moves])
            legal.append(ids)
            for j, key in enumerate(["prior"] + [f"off_p{tag(d)}" for d in DELTAS]):
                zl = z_at[i][j][ids]
                cols[key].append(np.exp(zl - zl.max()) / np.exp(zl - zl.max()).sum())
            heads.append(z_at[i][0][2350:2416])
        if a.limit:  # a timing test: nothing saved
            print(f"{len(idx)} positions, {ntok} tokens, {time.monotonic() - t0:.0f} s", flush=True)
            break
        np.savez(dst.with_suffix(".partial.npz"), index=idx, legal=np.concatenate(legal).astype(np.int16),
                 offsets=np.cumsum([0] + [len(x) for x in legal]), heads=np.array(heads, np.float32),
                 **{k: np.concatenate(v).astype(np.float32) for k, v in cols.items()})  # fmt: skip
        dst.with_suffix(".partial.npz").replace(dst)
        dt = time.monotonic() - t0
        print(
            f"{dst.name}: {len(idx)} positions, {len(groups)} game sides, {ntok} tokens, {dt:.0f} s ({ntok / dt:.0f} tok/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
