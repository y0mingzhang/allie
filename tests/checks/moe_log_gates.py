"""arch moe_log_gates and moe_dense_first. CPU, the routed experts as moe_center's torch reference.

off: bitwise REV's MoE over 3 steps (outputs, grads, buffers, state dict) and REV's extra_flops.
log gates: k**0.5 * softmax(logsigmoid(selected logits)) and the balance-loss affinities softmax(logsigmoid(all))
equal the renormalised sigmoids to FP32 rounding on ordinary tokens (outputs, grads); on tokens whose logits sit at
-120 / -60 / -31 / -30 on every expert (where the plain division is 0 / 0 or its backward overflows) outputs and
grads are finite. moe_dims carries the switch; resolve refuses it with sqrtsoftplus scores and refuses
moe_dense_first < 1; extra_flops moves (n - 1) MoE layers' FLOPs to the dense MLP.

    .venv/bin/python tests/checks/moe_log_gates.py [REV]   (REV: the commit before the switches, default 7baccfe)
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from allie.model import arch as model_arch
from allie.model import moe as mm
from allie.model import moe_kernels
from moe_center import DEV, D, E, H, K, reference, routed_ref, same, step
from moe_gate_floor import saturated


def old_arch(rev):
    src = subprocess.check_output(
        ["git", "-C", Path(__file__).parent, "show", f"{rev}:scripts/modded_arch.py"]
    )
    path = Path(tempfile.mkdtemp()) / "ref_arch.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("ref_arch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(router, x, dy, log_gates):
    torch.manual_seed(1)
    m = mm.MoE(D, E, K, H, H, update="quantile", seq=1e-3, log_gates=log_gates)
    m = m.to(DEV).train()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        m.router.copy_(router)
        m.down.copy_(0.3 * torch.randn(m.down.shape, generator=g))
        m.shared_down.copy_(0.3 * torch.randn(m.shared_down.shape, generator=g))
    x = x.clone().requires_grad_()
    y = m(x)
    y.float().pow(2).sum().backward() if dy is None else y.backward(dy)
    return y.detach(), x.grad, {n: p.grad for n, p in m.named_parameters()}


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


def refused(arch):
    try:
        model_arch.resolve(arch)
    except AssertionError:
        return True
    return False


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "7baccfe"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29586")
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

    ref = reference(rev)
    kw = dict(update="quantile", seq=1e-3, center=0.9, gate_floor=1e-12)
    torch.manual_seed(1)
    old = ref.MoE(D, E, K, H, H, **kw).to(DEV).train()
    torch.manual_seed(1)
    new = mm.MoE(D, E, K, H, H, **kw).to(DEV).train()
    got = [step(m, x, dy) for m in (old, new) for x, dy in zip(xs, dys)]
    off = all(same(a, b) for a, b in zip(got[:3], got[3:]))
    off &= same(old.state_dict(), new.state_dict())
    ra, shapes = old_arch(rev), [(8, 512), (24, 1536), (32, 2048)]
    s16 = dict(moe=[256, 16], moe_shared_frac=0.25, moe_round=16)
    flops = all(
        ra.extra_flops(s16, d, L) == model_arch.extra_flops(s16, d, L)
        for L, d in shapes
    )
    flops &= ra.extra_flops({}, 512, 8) == model_arch.extra_flops({}, 512, 8)
    print(f"off: bitwise {rev}'s MoE over 3 steps: {off}; extra_flops equal: {flops}")

    a3 = s16 | dict(moe_dense_first=3)
    dense = 2 * (3 * 2048 * model_arch.swiglu_hidden(2048) - 8 * 2048 * 2048)
    _, k, routed, shared, *_ = model_arch.moe_dims(2048, s16)
    layer = 2 * 2048 * 256 + 2 * (3 * 2048 * (shared + k * routed) - 8 * 2048 * 2048)
    moved = model_arch.extra_flops(a3, 2048, 32) - model_arch.extra_flops(
        s16, 2048, 32
    ) == 2 * (dense - layer)
    carried = mm.MoE(
        D, *model_arch.moe_dims(1536, s16 | dict(moe_log_gates=True))
    ).log_gates
    guards = refused(dict(moe_log_gates=True, moe_score="sqrtsoftplus"))
    guards &= refused(dict(moe_dense_first=0)) and refused(dict(moe_dense_first=2.0))
    print(
        f"dense_first 3 moves 2 layers' FLOPs: {moved}; moe_dims carries log_gates: {carried}; guards: {guards}"
    )

    router = 0.3 * torch.randn(E, D, generator=gen)
    x = (torch.randn(1, 1024, D, generator=gen) + 0.5).to(DEV, torch.bfloat16)
    y0, gx0, g0 = run(router, x, None, False)
    y1, gx1, g1 = run(router, x, None, True)
    err = max([rel(y1, y0), rel(gx1, gx0)] + [rel(g1[n], g0[n]) for n in g0])
    z = x[0].float() @ router.T
    idx = torch.sigmoid(z).topk(K, -1).indices
    s = torch.sigmoid(z).gather(1, idx)
    wp = s * (K**0.5 / s.sum(-1, keepdim=True))
    wl = K**0.5 * torch.softmax(torch.nn.functional.logsigmoid(z.gather(1, idx)), -1)
    gate = rel(wl, wp)
    print(
        f"log vs plain, ordinary tokens: FP32 gates {gate:.1e}, BF16 outputs / grads {err:.1e} (relative)"
    )

    router, xs = saturated(gen)
    dy = torch.randn(1, 1024, D, generator=gen).to(DEV, torch.bfloat16)
    fin = lambda *ts: all(bool(torch.isfinite(t).all()) for t in ts)
    y0, gx0, g0 = run(router, xs, dy, False)
    y1, gx1, g1 = run(router, xs, dy, True)
    plain, logs = fin(y0, gx0, *g0.values()), fin(y1, gx1, *g1.values())
    print(f"saturated tokens: plain finite {plain}, log gates finite {logs}")

    ok = (
        off
        and flops
        and moved
        and carried
        and guards
        and gate < 1e-6
        and err < 1e-3
        and not plain
        and logs
    )
    dist.destroy_process_group()
    print("PASS" if ok else "FAIL")
    sys.exit(not ok)


if __name__ == "__main__":
    main()
