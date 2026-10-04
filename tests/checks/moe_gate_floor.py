"""arch moe_gate_floor (model.moe.MoE gate_floor): off is bitwise the previous commit's MoE (forward, grads,
rebalance, state dict over 3 steps); moe_dims carries it and resolve refuses a nonfinite one. Tokens whose router
logits sit near -120 on every expert (scores 0: 0 / 0 forward), near -60 (sum ~1e-24; on the pinned GPU runtime the
compiled division backward overflows below ~1e-19) and just under / over the floor: off, the output is nonfinite;
floored, outputs and grads are finite, the all-zero tokens get the shared expert only and every token whose sum is at
least the floor is bitwise off's. CPU: E = 256 takes the torch top-k path, the routed experts run as moe_center's
torch reference.

    .venv/bin/python tests/checks/moe_gate_floor.py [REV]   (REV: the reference MoE, default f9baa37)
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.nn import functional as F

from allie.model import arch as model_arch
from allie.model import moe as mm
from allie.model import moe_kernels
from moe_center import DEV, D, E, H, K, reference, routed_ref, same, step

FLOOR = 1e-12
LOGITS = (
    -120,
    -60,
    -31.1,
    -29.7,
)  # 8 tokens each; the rest orthogonal to the router direction


def saturated(gen):
    """Router rows 30 u + noise; tokens 8i..8i+7 at logits ~LOGITS[i]."""
    u = torch.randn(D, generator=gen)
    u /= u.norm()
    router = 30 * u + 0.05 * torch.randn(E, D, generator=gen)
    x = torch.randn(1024, D, generator=gen)
    x -= (x @ u)[:, None] * u
    for i, z in enumerate(LOGITS):
        x[8 * i : 8 * i + 8] = x[8 * i : 8 * i + 8] * 0.001 + z / 30 * u
    return router, x[None].to(DEV, torch.bfloat16)


def build(router, floor):
    """A fresh MoE with this router and random (not zero-init) routed and shared down projections."""
    torch.manual_seed(1)
    m = mm.MoE(D, E, K, H, H, seq=1e-3, gate_floor=floor)
    m = m.to(DEV).train()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        m.router.copy_(router)
        m.down.copy_(0.3 * torch.randn(m.down.shape, generator=g))
        m.shared_down.copy_(0.3 * torch.randn(m.shared_down.shape, generator=g))
    return m


def run(router, x, dy, floor):
    m = build(router, floor)
    x = x.clone().requires_grad_()
    y = m(x)
    y.backward(dy)
    finite = lambda t: bool(torch.isfinite(t).all())
    grads = all(finite(p.grad) for p in m.parameters()) and finite(x.grad)
    return y.detach(), finite(y), grads


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "f9baa37"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29584")
    dist.init_process_group("gloo", rank=0, world_size=1)
    moe_kernels.routed = routed_ref
    gen = torch.Generator().manual_seed(0)
    xs = [
        (torch.randn(1, 1024, D, generator=gen) + 0.5).to(DEV, torch.bfloat16)
        for _ in range(3)
    ]
    dys = [
        torch.randn(1, 1024, D, generator=gen).to(DEV, torch.bfloat16) for _ in range(3)
    ]
    ok = True

    ref = reference(rev)
    kw = dict(seq=1e-3, center=0.9)
    torch.manual_seed(1)
    old = ref.MoE(D, E, K, H, H, update="quantile", **kw).to(DEV).train()
    torch.manual_seed(1)
    new = mm.MoE(D, E, K, H, H, **kw).to(DEV).train()
    got = [step(m, x, dy) for m in (old, new) for x, dy in zip(xs, dys)]
    off = all(same(a, b) for a, b in zip(got[:3], got[3:]))
    off &= same(old.state_dict(), new.state_dict())
    print(
        f"floor off: bitwise {rev}'s MoE over 3 steps (outputs, grads, buffers, state dict): {off}"
    )
    arch = dict(moe=[E, K], moe_update="quantile", moe_gate_floor=FLOOR)
    carried = mm.MoE(D, *model_arch.moe_dims(1536, arch)).gate_floor == FLOOR
    try:
        model_arch.resolve(dict(moe_gate_floor=float("inf")))
        refused = False
    except AssertionError:
        refused = True
    print(
        f"moe_dims carries the floor: {carried}; resolve refuses an infinite one: {refused}"
    )
    ok &= off and carried and refused

    router, x = saturated(gen)
    s = torch.sigmoid(x[0].float() @ router.T)
    total = s.topk(K, dim=-1).values.sum(-1)
    for i, z in enumerate(LOGITS):
        print(
            f"tokens at logits ~{z}: selected score sums {total[8 * i : 8 * i + 8].max():.2g} max"
        )
    dy = torch.randn(1, 1024, D, generator=gen).to(DEV, torch.bfloat16)
    y0, fy0, g0 = run(router, x, dy, 0.0)
    y1, fy1, g1 = run(router, x, dy, FLOOR)
    print(f"off: output finite {fy0}, grads finite {g0}")
    print(f"floor {FLOOR:g}: output finite {fy1}, grads finite {g1}")
    keep = total >= FLOOR
    rest = torch.equal(y0[0, keep], y1[0, keep])
    print(
        f"floor {FLOOR:g}: the {int(keep.sum())} tokens with sums >= the floor are bitwise off's: {rest}"
    )
    m = build(router, FLOOR)
    with torch.no_grad():
        h = x[0, :8]
        up, down = m.shared_up.type_as(h), m.shared_down.T.type_as(h)
        shared = F.linear(m.act(F.linear(h, up)), down)
    zero = torch.equal(y1[0, :8], shared)
    print(
        f"floor {FLOOR:g}: tokens whose scores are all 0 get the shared expert only: {zero}"
    )
    ok &= not fy0 and fy1 and g1 and rest and zero and int(keep.sum()) == 1024 - 24
    dist.destroy_process_group()
    print("PASS" if ok else "FAIL")
    sys.exit(not ok)


if __name__ == "__main__":
    main()
