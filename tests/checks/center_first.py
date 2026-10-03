"""arch moe_router_center_first: with centring on, only the first n MoE layers centre their router input; the init draws
are unchanged (same parameters as all-layer and no centring). CPU: .venv/bin/python tests/checks/center_first.py"""
import os
import torch, torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29591")
dist.init_process_group("gloo", rank=0, world_size=1)
from allie.model.network import Config, create_model
from allie.model.moe import MoE
s16 = dict(moe=[256, 16], moe_shared_frac=0.25, moe_round=16, moe_update="quantile", moe_log_gates=True)
def build(arch):
    torch.manual_seed(5)
    cfg = Config(width=128, head_dim=64, layers=8, max_tokens=1024, scheduled_steps=8, arch=arch)
    return create_model(cfg, device="cpu")
a = build(s16 | dict(moe_dense_first=3, moe_router_center=0.9))
b = build(s16 | dict(moe_dense_first=3, moe_router_center=0.9, moe_router_center_first=1))
c = build(s16 | dict(moe_dense_first=3))
cen = lambda m: [blk.mlp.center if isinstance(blk.mlp, MoE) else None for blk in m.blocks]
print("all", cen(a)); print("first1", cen(b))
same = lambda x, y: x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
pa = {k: v for k, v in a.state_dict().items() if "mu" not in k}
pb = {k: v for k, v in b.state_dict().items() if "mu" not in k}
pc = c.state_dict()
print("params equal across centring choices (init RNG unchanged):", same(pa, pb) and same(pb, {k: v for k, v in pc.items()}))
ok = cen(a)[3:] == [0.9] * 5 and cen(b)[3:] == [0.9] + [0.0] * 4 and cen(a)[:3] == [None] * 3
print("PASS" if ok else "FAIL")
