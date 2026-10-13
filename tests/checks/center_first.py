"""arch moe_router_center_first: with centring on, only the first n MoE layers centre their router input; the init draws
are unchanged (same parameters as all-layer and no centring). CPU: .venv/bin/python tests/checks/center_first.py"""

import os

import torch
import torch.distributed as dist

from allie.model.moe import MoE
from allie.model.network import Config, create_model

S16 = dict(
    moe=[256, 16],
    moe_shared_frac=0.25,
    moe_round=16,
    moe_update="quantile",
    moe_log_gates=True,
    moe_dense_first=3,
)


def build(**arch):
    torch.manual_seed(5)
    cfg = Config(
        width=128,
        head_dim=64,
        layers=8,
        max_tokens=1024,
        scheduled_steps=8,
        arch=S16 | arch,
    )
    return create_model(cfg, device="cpu")


def centres(m):
    return [blk.mlp.center if isinstance(blk.mlp, MoE) else None for blk in m.blocks]


def params(m):
    return {k: v for k, v in m.state_dict().items() if "mu" not in k}


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    dist.init_process_group("gloo", rank=0, world_size=1)
    a = build(moe_router_center=0.9)
    b = build(moe_router_center=0.9, moe_router_center_first=1)
    c = build()
    pa, pb, pc = params(a), params(b), params(c)
    assert pa.keys() == pb.keys() == pc.keys() and all(
        torch.equal(pa[k], pb[k]) and torch.equal(pb[k], pc[k]) for k in pa
    )
    assert centres(a) == [None] * 3 + [0.9] * 5, centres(a)
    assert centres(b) == [None] * 3 + [0.9] + [0.0] * 4, centres(b)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
