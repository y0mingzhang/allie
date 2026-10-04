"""CPU pass of the calibration eval: the legal policy and root heads (think time, W/D/L) at every
position, from allie.lichess's model (the bot's own code), one forward per game covering all of
its sampled positions. Writes chunks in calib_search.py's format (index, legal, offsets, prior, heads).

python calib_policy.py POSITIONS.npz OUT_DIR [--shard i --shards n --threads 8]
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
from allie.lichess.model import Model
from allie.lichess.tokens import HEADER, START, advance
from torch.nn import functional as F

from play import MODEL


@torch.inference_mode()
def logits_at(model, ids, feats, boards, last):
    """Logits at rows `last` of one game's tokens from position 0 (no cache)."""

    def previous(e):
        p = torch.zeros_like(e)
        p[1:] = e[:-1]
        return p

    def attend(i, q, k, v):
        t = lambda x: x.transpose(0, 1)
        return t(
            F.scaled_dot_product_attention(
                t(q), t(k), t(v), is_causal=True, scale=model.scale
            )
        )

    pos = torch.arange(len(ids))
    return model.forward(ids, pos, feats, boards, previous, attend, torch.tensor(last))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("out")
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    z = np.load(a.positions)
    off, tokens, feats, meta = z["offsets"], z["tokens"], z["feats"], z["meta"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(MODEL)
    for lo in list(range(0, len(meta), a.chunk))[a.shard :: a.shards]:
        dst = out / f"{lo:06d}.npz"
        if dst.exists():
            continue
        t0 = time.monotonic()
        idx = np.arange(lo, min(lo + a.chunk, len(meta)))
        games = defaultdict(
            list
        )  # same game, longest prefix first: one forward for all
        for i in idx:
            games[meta[i, 0], meta[i, 1]].append(i)
        z_at = {}
        for members in games.values():
            longest = max(members, key=lambda i: off[i + 1] - off[i])
            seq = tokens[off[longest] : off[longest + 1]].astype(np.int64)
            boards = [START] * HEADER
            for t in seq[HEADER:]:
                boards.append(advance(boards[-1], int(t)))
            b = torch.tensor(np.frombuffer(b"".join(boards), np.uint8).reshape(-1, 68))
            f = torch.tensor(
                feats[off[longest] : off[longest + 1]], dtype=torch.float32
            )
            last = [off[i + 1] - off[i] - 1 for i in members]
            zs = logits_at(model, torch.tensor(seq), f, b, last).double().numpy()
            z_at.update(zip(members, zs))
        legal, prior, heads = [], [], []
        for i in idx:
            board = chess.Board()
            for t in tokens[off[i] + HEADER : off[i + 1]]:
                board.push_uci(MOVES[t - 378])
            ids = np.array([MOVE_ID[m.uci()] for m in board.legal_moves])
            zl = z_at[i][ids]
            legal.append(ids)
            prior.append(np.exp(zl - zl.max()) / np.exp(zl - zl.max()).sum())
            heads.append(z_at[i][2350:2416])
        np.savez(dst.with_suffix(".partial.npz"), index=idx, legal=np.concatenate(legal).astype(np.int16),
                 offsets=np.cumsum([0] + [len(x) for x in legal]), prior=np.concatenate(prior).astype(np.float32),
                 heads=np.array(heads, np.float32))  # fmt: skip
        dst.with_suffix(".partial.npz").replace(dst)
        print(
            f"{dst.name}: {len(idx)} positions in {len(games)} games, {time.monotonic() - t0:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
