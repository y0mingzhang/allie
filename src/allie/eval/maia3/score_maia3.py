"""Score a Maia-3 checkpoint on the golden eval's blitz positions, on CPU.

Per selected position: natural-log cross-entropy of the human move under the model's
legal-move-renormalized policy, and whether the legal argmax is the human move.
Inputs are built exactly as maia3.uci does: 8 past board tokenizations (each mirrored
by its own side to move), no clock channel (include_time_info=False), self/opponent Elo.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

from allie import paths
from allie.data.vocab import MOVES

sys.path.insert(0, str(paths.MAIA3_REPO))
from maia3.dataset import tokenize_board  # noqa: E402
from maia3.model_registry import resolve_checkpoint_path, resolve_model_spec  # noqa: E402
from maia3.models import MAIA3Model  # noqa: E402
from maia3.utils import get_all_possible_moves, mirror_move  # noqa: E402

DATA = paths.DATA / "maia3-bench"


def load(name, threads):
    spec = resolve_model_spec(name)
    cfg = argparse.Namespace(**spec.config)
    model = MAIA3Model(cfg)
    state = torch.load(resolve_checkpoint_path(spec), map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(
        {k.replace("smolgen", "gab"): v for k, v in state.items()}, strict=False
    )
    assert not missing, missing
    model.eval()
    torch.set_num_threads(threads)
    return model, cfg, sum(p.numel() for p in model.parameters()), unexpected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="score only this many positions (timing)")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(DATA), help="directory of games.npz")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    model, cfg, params, unexpected = load(a.model, a.threads)
    model.to(a.device)
    vocab = {m: i for i, m in enumerate(get_all_possible_moves())}
    with np.load(Path(a.data) / "games.npz") as z:
        moves, off, meta, sel, keep = (z[k] for k in ("moves", "offsets", "meta", "sel", "keep"))
    if a.limit:
        keep = keep[np.linspace(0, len(keep) - 1, a.limit).astype(int)]
    want = {}
    for i in keep:
        want.setdefault(int(sel[i, 0]), []).append(int(i))

    n = len(keep)
    slot = {int(i): j for j, i in enumerate(keep)}
    ce, top1 = np.zeros(n), np.zeros(n, bool)
    hist_dim, batch, done = cfg.history, [], 0
    forward = 0.0
    t0 = time.perf_counter()

    def flush():
        nonlocal batch, done, forward
        if not batch:
            return
        tok = torch.stack([b[0] for b in batch])
        se = torch.tensor([b[1] for b in batch], dtype=torch.long)
        oe = torch.tensor([b[2] for b in batch], dtype=torch.long)
        f0 = time.perf_counter()
        with torch.no_grad():
            logits, _, _ = model(*(x.to(a.device) for x in (tok, se, oe)))
            if a.device != "cpu":
                torch.cuda.synchronize()
        forward += time.perf_counter() - f0
        for (_, _, _, mask, tgt, j), row in zip(batch, logits.float().cpu()):
            row = row.masked_fill(~mask, float("-inf"))
            ce[j] = float(torch.logsumexp(row, 0) - row[tgt])
            top1[j] = int(row.argmax()) == tgt
        done += len(batch)
        batch = []

    for g in sorted(want):
        board, hist = chess.Board(), []
        plies = {int(sel[i, 1]): i for i in want[g]}
        uci = [MOVES[k] for k in moves[off[g] : off[g + 1]]]
        for m, u in enumerate(uci):
            hist.append(tokenize_board(board))
            if m in plies:
                h = hist[-hist_dim:]
                t = torch.cat(h, dim=1)
                if len(h) < hist_dim:
                    t = torch.cat([h[0].repeat(1, hist_dim - len(h)), t], dim=1)
                mask = torch.zeros(len(vocab), dtype=torch.bool)
                for lm in board.legal_moves:
                    mask[vocab[lm.uci() if board.turn else mirror_move(lm.uci())]] = True
                tgt = vocab[u if board.turn else mirror_move(u)]
                assert mask[tgt], (g, m, u)
                w, b = int(meta[g, 2]), int(meta[g, 3])
                self_elo, oppo_elo = (w, b) if board.turn else (b, w)
                batch.append((t, self_elo, oppo_elo, mask, tgt, slot[plies[m]]))
                if len(batch) == a.batch:
                    flush()
            board.push(chess.Move.from_uci(u))
        del hist
    flush()
    dt = time.perf_counter() - t0

    out = Path(a.out)
    np.savez(out, keep=keep, ce=ce, top1=top1)
    info = dict(model=a.model, parameters=int(params), positions=n, seconds=dt, forward_seconds=forward,
                positions_per_s=n / dt, threads=a.threads, batch=a.batch,
                unexpected_keys=list(unexpected), history=int(cfg.history),
                include_time_info=bool(cfg.include_time_info), device=a.device)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
