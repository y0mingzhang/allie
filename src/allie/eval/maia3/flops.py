"""Measured matmul FLOPs per scored move for each model (torch FlopCounterMode)."""
import json
import sys

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from allie import paths

sys.path.insert(0, str(paths.MAIA3_REPO))

DATA = paths.DATA / "maia3-bench"
OUT = paths.ROOT / "results/recipe10x/maia3-bench/flops.json"


def maia(name):
    import argparse
    from maia3.model_registry import resolve_model_spec
    from maia3.models import MAIA3Model
    spec = resolve_model_spec(name)
    cfg = argparse.Namespace(**spec.config)
    model = MAIA3Model(cfg).eval()
    x = torch.zeros(1, 64, 12 * cfg.history)
    e = torch.tensor([1500])
    with FlopCounterMode(display=False) as f, torch.no_grad():
        model(x, e, e)
    return sum(p.numel() for p in model.parameters()), f.get_total_flops()


def ours(length):
    from allie.search.model import DenseBackend, load_checkpoint
    from allie.search.board import encode
    ckpt = (paths.ROOT / "results/pretrain/msh-3e17-model-pf115h-s42/"
            "checkpoints/step-00004974-298d0686/model.pt")
    model, _ = load_checkpoint(ckpt, "cpu")
    model = model.float()
    import chess
    from allie.data.vocab import MOVE_ID
    b, moves = chess.Board(), []
    while len(moves) < length - 11 and b.legal_moves.count():
        mv = next(iter(b.legal_moves))
        moves.append(MOVE_ID[mv.uci()])
        b.push(mv)
    ids = np.array([2348, 199, 10, 1, 5, 0, 0, 1, 5, 0, 0] + moves, dtype=np.int64)
    states = torch.from_numpy(encode(ids[None])[0])
    ft = torch.full((len(ids), 3), -1.0)
    with FlopCounterMode(display=False) as f, torch.inference_mode():
        model(torch.as_tensor(ids), torch.arange(len(ids)), DenseBackend(), ft, states)
    params = sum(v.numel() for v in model.state_dict().values())
    return params, f.get_total_flops(), len(ids)


def main():
    out = {}
    for n in ("maia3-5m", "maia3-23m", "maia3-79m"):
        p, fl = maia(n)
        out[n] = dict(parameters=p, flops_per_move=fl,
                      note="one 64-square encoder forward = one move")
    with np.load(DATA / "games.npz") as z:
        lens = np.diff(z["offsets"]) + 11
    L = int(np.median(lens))
    p, fl, n = ours(L)
    out["ours-129m"] = dict(
        parameters=p, prefix_tokens=n, flops_whole_game=fl,
        flops_per_move_rescoring=fl / (n - 11),
        flops_per_move_incremental=fl / n,
        note=("full-prefix teacher-forced rescoring of a median-length blitz game "
              f"({L} tokens); rescoring divides the whole-game forward by its moves, "
              "incremental is the per-token cost a KV-cached server pays"),
    )
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
