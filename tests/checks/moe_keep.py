"""arch moe_keep (model.moe.MoE keep). CPU, the routed experts as moe_center's torch reference.

off: bitwise REV's MoE over 3 training steps (outputs, grads, buffers, state dict), REV's extra_flops and moe_dims.
keep: the routed output is the top-k one with every expert outside a token's best keep (by score + bias) given gate
0, to FP32 rounding (forward and grads); the dropped experts get no gradient; the load, the quantile histogram, the
bias and the balance loss are the top-k's (buffers after a step and rebalance equal keep 0's); keep 16 of 16 is
refused. extra_flops counts keep routed experts.

    .venv/bin/python tests/checks/moe_keep.py [REV]   (REV: the commit before the switch, default a1663a0)
"""

import os
import sys

import torch
import torch.distributed as dist

from allie.model import arch as model_arch
from allie.model import moe as mm
from allie.model import moe_kernels
from moe_center import DEV, D, E, H, K, reference, routed_ref, same, step
from moe_log_gates import old_arch, refused

KEEP = 4


def masked_routed(keep_of):
    """routed_ref over all k experts, the gates outside each token's best keep set to 0."""

    def routed(x, up_t, down, k, se, order, offsets, gates, remat=True):
        return routed_ref(
            x, up_t, down, k, se, order, offsets, gates * keep_of(gates), remat
        )

    return routed


def build(keep, kw):
    torch.manual_seed(1)
    m = mm.MoE(D, E, K, H, H, keep=keep, **kw).to(DEV).train()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        m.down.copy_(0.3 * torch.randn(m.down.shape, generator=g))
        m.shared_down.copy_(0.3 * torch.randn(m.shared_down.shape, generator=g))
        m.bias.copy_(0.05 * torch.randn(E, generator=g))
    return m


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "a1663a0"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29587")
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
    kw = dict(update="quantile", seq=1e-3, center=0.9, gate_floor=1e-12)

    ref = reference(rev)
    torch.manual_seed(1)
    old = ref.MoE(D, E, K, H, H, **kw).to(DEV).train()
    torch.manual_seed(1)
    new = mm.MoE(D, E, K, H, H, **kw).to(DEV).train()
    got = [step(m, x, dy) for m in (old, new) for x, dy in zip(xs, dys)]
    off = all(same(a, b) for a, b in zip(got[:3], got[3:])) and same(
        old.state_dict(), new.state_dict()
    )
    ra, s16 = old_arch(rev), dict(moe=[256, 16], moe_shared_frac=0.25, moe_round=16)
    shapes = [(8, 512), (24, 1536), (32, 2048)]
    flops = all(
        ra.extra_flops(s16, d, L) == model_arch.extra_flops(s16, d, L)
        for L, d in shapes
    )
    dims = ra.moe_dims(1536, s16) + (0,) == model_arch.moe_dims(1536, s16)
    print(
        f"off: bitwise {rev}'s MoE over 3 steps: {off}; extra_flops, moe_dims equal: {flops}, {dims}"
    )

    # keep vs the top-k layer with the dropped experts' gates zeroed: same routing state, so one step each
    x, dy = xs[0], dys[0]
    a = build(KEEP, kw)
    ya, gxa, ga, ba = step(a, x, dy)
    b = build(0, kw)
    h = x.reshape(-1, D).float()
    s = (
        torch.sigmoid((h - b.mu) @ b.router.T.float())
        if b.center
        else torch.sigmoid(h @ b.router.T.float())
    )
    idx = torch.topk(s + b.bias, K, dim=-1).indices
    best = torch.sort(
        (s + b.bias).gather(1, idx), dim=-1, descending=True, stable=True
    ).indices[:, :KEEP]
    mask = torch.zeros_like(s, dtype=torch.bool).scatter_(1, idx.gather(1, best), True)
    kept = mask.gather(1, idx)  # [t, k] in the routed order of idx
    moe_kernels.routed = masked_routed(lambda gates: kept.to(gates.dtype))
    yb, gxb, gb, bb = step(b, x, dy)
    moe_kernels.routed = routed_ref
    err = max(
        [rel(ya, yb), rel(gxa, gxb)]
        + [rel(ga[n], gb[n]) for n in gb if gb[n].abs().sum() > 0]
    )
    used = mask.any(0)
    dropped = bool((ga["up"][~used] == 0).all() and (ga["down"][~used] == 0).all())
    stats = same(ba, bb)  # load, histogram, bias after rebalance, mu: all the top-k's
    print(
        f"keep {KEEP} vs top-{K} with the rest's gates 0: rel err {err:.1e}; unrouted experts no grad: {dropped}; buffers equal: {stats}"
    )

    k4 = s16 | dict(moe_keep=KEEP)
    _, k, routed, shared, *_ = model_arch.moe_dims(1536, s16)
    fewer = model_arch.extra_flops(s16, 1536, 24) - model_arch.extra_flops(
        k4, 1536, 24
    ) == 2 * 23 * 3 * 1536 * routed * (k - KEEP)
    carried = mm.MoE(D, *model_arch.moe_dims(1536, k4)).keep == KEEP
    guards = (
        refused(s16 | dict(moe_keep=16))
        and refused(dict(moe_keep=2))
        and refused(s16 | dict(moe_keep=2.0))
    )
    print(
        f"extra_flops drops {k - KEEP} routed experts per MoE layer: {fewer}; moe_dims carries keep: {carried}; guards: {guards}"
    )

    ok = (
        off
        and flops
        and dims
        and err < 1e-3
        and dropped
        and stats
        and fewer
        and carried
        and guards
    )
    dist.destroy_process_group()
    print("PASS" if ok else "FAIL")
    sys.exit(not ok)


if __name__ == "__main__":
    main()
