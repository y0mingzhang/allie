"""header_feats (per-token Elo and time-control features), on one GPU.

features: on the golden eval's packed rows, header_features decodes every move position's mover
and opponent Elo, base time and increment exactly as the header says, is zero on headers and on
rows' headerless first games, and is causal (a position's features ignore every later token).
train: a small compiled MoE trains 4 steps with header_feats 0, 1 and 2 on real rows; with 1 and
2 the zero-init header table has moved after the first Adam step (and only its used columns),
with 0 the run is the plain model: its losses go to --dump, to compare with this test run from
another checkout (its own copy of this check; its code needs no header_feats).

    torchrun --standalone --nproc_per_node=1 tests/checks/header_feats.py [--dump losses.json]
"""

import json
import sys

import numpy as np
import torch
import torch.distributed as dist

from allie import paths
from allie.data.vocab import BOS, INCREMENTS, SECONDS
from allie.data.packed import Packed
from allie.model.network import (
    Config,
    TrainingManager,
    core,
    create_model,
    make_context,
    move_losses,
)
from allie.train.schedule import Schedule

DATA = str(paths.DATA / "lichess_tokens_v2")
STRAT = str(paths.DATA / "strat-eval-v1/strat.npz")
SCHEDULE = Schedule(warmup_steps=2, mtp_steps=4, split_step=5, batch_rows=2)


def reference(rows):
    """Per input position: mover Elo, opponent Elo, base seconds, increment; -1 = no feature."""
    out = np.full((*rows.shape, 4), -1)
    for i, row in enumerate(rows):
        bos = [*np.flatnonzero(row == BOS), len(row)]
        for s, e in zip(bos, bos[1:]):
            h = row[s : s + 11]
            if len(h) < 11 or h[1] == BOS:
                continue
            w, b = h[3:7] @ [1000, 100, 10, 1], h[7:11] @ [1000, 100, 10, 1]
            base, inc = SECONDS[h[1] - 192], INCREMENTS[h[2] - 10]
            for p in range(s + 10, e):
                white = (p - s) % 2 == 0
                out[i, p] = (
                    w if white else b,
                    b if white else w,
                    -1 if base == "*" else int(base),
                    -1 if inc == "*" else int(inc),
                )
    return out


def features():
    with np.load(STRAT) as z:
        rows = z["rows"][:24, :-1].astype(np.int64)
    x = torch.as_tensor(rows, device="cuda")
    same = x != BOS
    same[:, 0] = False
    same = same.flatten()
    f = core.header_features(x.flatten(), same, 2).view(*rows.shape, 128)[..., :72]
    f = f.view(*rows.shape, 4, 18).cpu().double()
    ref = reference(rows)
    have = f[..., 17] == 1
    assert torch.equal(have, torch.as_tensor(ref >= 0)), "feature presence"
    got = torch.stack((f[..., 0, 16] * 4000, f[..., 1, 16] * 4000), -1)
    got = torch.cat((got, torch.expm1(f[..., 2:, 16] * 10)), -1)
    ok = torch.as_tensor(ref >= 0)
    err = (got - torch.as_tensor(ref, dtype=torch.float64)).abs()[ok]
    assert err.max() < 0.02, float(err.max())  # float32 v and log1p
    assert (f[~ok] == 0).all(), "features on positions without a header"
    flat = x.flatten()
    full = core.header_features(flat, same, 2)
    for p in (5, 300, 1033, 5000, flat.numel() - 2):
        part = core.header_features(flat[: p + 1], same[: p + 1], 2)
        assert torch.equal(part, full[: p + 1]), f"not causal at {p}"
    print(
        json.dumps(dict(features_exact=int(ok.sum()), positions=ok.numel())), flush=True
    )


def train(n, val):
    torch.manual_seed(701)
    arch = dict(moe=[16, 2], moe_seq=0.001) | (dict(header_feats=n) if n else {})
    cfg = Config(
        width=128, head_dim=64, layers=8, max_tokens=1024, scheduled_steps=8, arch=arch
    )
    m = create_model(cfg)
    mgr = TrainingManager(m, cfg, SCHEDULE)
    net = torch.compile(m, dynamic=False, fullgraph=True)
    losses = []
    for step in range(4):
        mgr.advance_schedule(step)
        for micro in range(2):
            if micro == 1:
                mgr.activate_hooks(step)
            rows = torch.as_tensor(val.rows(np.array([100 * step + 10 * micro])))
            x, y = rows[:, :-1].cuda(), rows[:, 1:].cuda()
            ctx = make_context(x, mgr.ws_short * 128, mgr.ws_long * 128)
            logits = net(x.flatten(), y.flatten(), ctx, mgr.get_forward_args())
            loss, move, count = move_losses(logits, x, y, ctx, mgr.mtp_weights)
            (loss / 8).backward()
            losses.append(float(move.detach() / count))
        mgr.step_optimizers(step)
    if n:
        w = m.header_embed.weight.float()
        moved = w.abs().sum(1) > 0
        assert moved[: 36 * n].any() and not moved[36 * n :].any(), moved.nonzero()
    assert all(np.isfinite(losses)), losses
    return losses


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    val = Packed(DATA, "val")
    if hasattr(core, "header_features"):
        features()
    out = {
        n: train(n, val)
        for n in ((0, 1, 2) if hasattr(core, "header_features") else (0,))
    }
    print(json.dumps(out), flush=True)
    if "--dump" in sys.argv:
        with open(sys.argv[sys.argv.index("--dump") + 1], "w") as f:
            json.dump(out[0], f)
    dist.destroy_process_group()
    print("PASS header_feats", flush=True)


if __name__ == "__main__":
    main()
