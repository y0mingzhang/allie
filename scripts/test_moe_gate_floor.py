"""modded_moe.GATE_FLOOR (modded_train --moe-gate-floor): off is bitwise the previous commit's MoE (forward, grads,
rebalance, state dict over 3 steps). With router logits near -60 on every expert for some tokens (their sigmoid scores
sum to ~1e-24: the renormalisations' division backward overflows) and near -120 (the scores are 0: 0 / 0 forward),
off gives nonfinite outputs and grads; floored, both are finite, the -120 tokens get the shared expert only and every
token the floor leaves alone is bitwise off's. CPU: E = 256 takes the torch top-k path, the routed experts run as
test_moe_center's torch reference.

    .venv/bin/python scripts/test_moe_gate_floor.py [REV]   (REV: the reference MoE, default f9baa37)
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.nn import functional as F

import modded_moe as mm
import modded_smoe
from test_moe_center import DEV, D, E, H, K, reference, routed_ref, same, step

FLOOR = 1e-12


def saturated(gen):
    """Router rows 30 u + noise; tokens 0-7 at logits ~-60, 8-15 at ~-120, the rest orthogonal to u."""
    u = torch.randn(D, generator=gen)
    u /= u.norm()
    router = 30 * u + 0.05 * torch.randn(E, D, generator=gen)
    x = torch.randn(1024, D, generator=gen)
    x -= (x @ u)[:, None] * u
    x[:8] = x[:8] * 0.01 - 2 * u
    x[8:16] = x[8:16] * 0.01 - 4 * u
    return router, x[None].to(DEV, torch.bfloat16)


def build(router):
    """A fresh MoE with this router and random (not zero-init) routed and shared down projections."""
    torch.manual_seed(1)
    m = mm.MoE(D, E, K, H, H, update="quantile", seq=1e-3).to(DEV).train()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        m.router.copy_(router)
        m.down.copy_(0.3 * torch.randn(m.down.shape, generator=g))
        m.shared_down.copy_(0.3 * torch.randn(m.shared_down.shape, generator=g))
    return m


def run(router, x, dy, floor):
    mm.GATE_FLOOR = floor
    m = build(router)
    x = x.clone().requires_grad_()
    y = m(x)
    y.backward(dy)
    mm.GATE_FLOOR = 0.0
    finite = lambda t: bool(torch.isfinite(t).all())
    grads = all(finite(p.grad) for p in m.parameters()) and finite(x.grad)
    return y.detach(), finite(y), grads


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "f9baa37"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29584")
    dist.init_process_group("gloo", rank=0, world_size=1)
    modded_smoe.routed = routed_ref
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
    kw = dict(update="quantile", seq=1e-3, center=0.9)
    torch.manual_seed(1)
    old = ref.MoE(D, E, K, H, H, **kw).to(DEV).train()
    torch.manual_seed(1)
    new = mm.MoE(D, E, K, H, H, **kw).to(DEV).train()
    got = [step(m, x, dy) for m in (old, new) for x, dy in zip(xs, dys)]
    off = all(same(a, b) for a, b in zip(got[:3], got[3:]))
    off &= same(old.state_dict(), new.state_dict())
    print(
        f"floor off: bitwise {rev}'s MoE over 3 steps (outputs, grads, buffers, state dict): {off}"
    )
    ok &= off

    router, x = saturated(gen)
    z = x[0].float() @ router.T
    print(
        f"max logit per token: tokens 0-7 {z[:8].max():.1f}, 8-15 {z[8:16].max():.1f},"
        f" the rest >= {z[16:].max(1).values.min():.1f}"
    )
    dy = torch.randn(1, 1024, D, generator=gen).to(DEV, torch.bfloat16)
    y0, fy0, g0 = run(router, x, dy, 0.0)
    y1, fy1, g1 = run(router, x, dy, FLOOR)
    broken = not fy0 and not g0
    print(f"off: output finite {fy0}, grads finite {g0}")
    print(f"floor {FLOOR:g}: output finite {fy1}, grads finite {g1}")
    rest = torch.equal(y0[0, 16:], y1[0, 16:])
    print(f"floor {FLOOR:g}: the tokens it leaves alone are bitwise off's: {rest}")
    m = build(router)
    with torch.no_grad():
        h = x[0, 8:16]
        up, down = m.shared_up.type_as(h), m.shared_down.T.type_as(h)
        shared = F.linear(m.act(F.linear(h, up)), down)
    zero = torch.equal(y1[0, 8:16], shared)
    print(
        f"floor {FLOOR:g}: tokens whose scores are all 0 get the shared expert only: {zero}"
    )
    ok &= broken and fy1 and g1 and rest and zero
    dist.destroy_process_group()
    print("PASS" if ok else "FAIL")
    sys.exit(not ok)


if __name__ == "__main__":
    main()
