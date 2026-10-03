"""moe_router_center (model.moe.MoE center=beta): off is bitwise the previous commit's MoE (forward, grads,
rebalance, state dict); on routes by W (h - mu), mu = the first step's global mean router input, then an EMA
with decay beta, booked once per real forward (replay_context's doubled forward and skipped recompute leave mu
bitwise unchanged over several micro-batches). CPU: E = 256 takes the torch top-k
path, and the routed experts (model.moe_kernels's Triton kernels, unchanged) run as a torch reference.

    .venv/bin/python tests/checks/moe_center.py [REV]   (REV: the reference MoE, default 90fd74c, the commit before the switch)
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

import legacy
from allie.model import moe as mm
from allie.model import moe_kernels

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, E, K, H = 32, 256, 16, 8


def routed_ref(x, up_t, down, k, se, order, offsets, gates, remat=True):
    """moe_kernels.routed in torch: sum over a token's k experts of gate * SwiGLU expert(x)."""
    tok = order // k
    u = torch.bmm(x[tok].float()[:, None], up_t[se].float())[:, 0]
    a, b = u.chunk(2, dim=-1)
    y = torch.bmm((torch.nn.functional.silu(a) * b)[:, None], down[se].float())[:, 0]
    y = y * gates.flatten()[order].float()[:, None]
    return torch.zeros(len(x), x.shape[1]).index_add(0, tok, y).to(x.dtype)


def reference(rev):
    src = subprocess.check_output(
        ["git", "-C", HERE, "show", f"{rev}:scripts/modded_moe.py"]
    )
    path = Path(tempfile.mkdtemp()) / "ref_moe.py"
    path.write_bytes(
        src.replace(b"allie_moe::", b"ref_moe::")
    )  # custom op names are global
    legacy.alias()
    spec = importlib.util.spec_from_file_location("ref_moe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step(m, x, dy):
    """One training forward and backward, then rebalance: output, input grad, param grads, buffers."""
    x = x.clone().requires_grad_()
    y = m(x)
    y.backward(dy)
    m.rebalance(1.0)
    grads = {n: p.grad.clone() for n, p in m.named_parameters()}
    return y.detach(), x.grad, grads, {n: b.clone() for n, b in m.named_buffers()}


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "90fd74c"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29583")
    dist.init_process_group("gloo", rank=0, world_size=1)
    gen = torch.Generator().manual_seed(0)
    xs = [
        (torch.randn(1, 1024, D, generator=gen) + 0.5).to(DEV, torch.bfloat16)
        for _ in range(3)
    ]
    dys = [
        torch.randn(1, 1024, D, generator=gen).to(DEV, torch.bfloat16) for _ in range(3)
    ]
    ok = True

    moe_kernels.routed = routed_ref
    ref = reference(rev)
    kw = dict(update="quantile", seq=1e-3)
    torch.manual_seed(1)
    old = ref.MoE(D, E, K, H, H, **kw).to(DEV).train()
    torch.manual_seed(1)
    new = mm.MoE(D, E, K, H, H, **kw).to(DEV).train()
    got = [step(m, x, dy) for m in (old, new) for x, dy in zip(xs, dys)]
    off = all(same(a, b) for a, b in zip(got[:3], got[3:]))
    off &= same(old.state_dict(), new.state_dict())
    print(
        f"center off: bitwise {rev}'s MoE over 3 steps (outputs, grads, buffers, state dict): {off}"
    )
    ok &= off

    beta = 0.9
    torch.manual_seed(1)
    m = mm.MoE(D, E, K, H, H, center=beta, **kw).to(DEV).train()
    keys = {"mu", "mu_steps"} <= set(m.state_dict()) and "mu_sum" not in m.state_dict()
    print(f"center on: mu, mu_steps persistent, mu_sum not: {keys}")
    ok &= keys
    mean = lambda x: x.reshape(-1, D).float().mean(0)
    want = mean(xs[0])
    step(m, xs[0], dys[0])
    first = torch.allclose(m.mu, want, atol=1e-6) and m.mu_steps.item() == 1
    print(f"step 1: mu = the batch mean router input {first}")
    want = beta * want + (1 - beta) * mean(xs[1])
    step(m, xs[1], dys[1])
    ema = torch.allclose(m.mu, want, atol=1e-6) and m.mu_steps.item() == 2
    print(f"step 2: mu = {beta} mu + {1 - beta} mean {ema}")
    ok &= first and ema

    m.load.zero_()
    with torch.no_grad():
        h = xs[2].reshape(-1, D)
        m(xs[2])
        s = torch.sigmoid((h.float() - m.mu) @ m.router.float().T)
        idx = torch.topk(s + m.bias, K, dim=-1).indices
        routed = torch.equal(
            m.load[:E], torch.bincount(idx.flatten(), minlength=E).float()
        )
    print(f"routing = top-k of sigmoid(W (h - mu)) + bias: {routed}")
    ok &= routed

    twice = mm.MoE(D, E, K, H, H, center=beta, **kw).to(DEV).train()
    twice.load_state_dict(m.state_dict())
    once = mm.MoE(D, E, K, H, H, center=beta, **kw).to(DEV).train()
    once.load_state_dict(m.state_dict())
    with torch.no_grad():
        for x in xs:  # three micro-batches
            once(x)
            with mm._reps(2):
                twice(x)
            with mm._reps(0, True):
                twice(x)
    once.rebalance(1.0)
    twice.rebalance(1.0)
    replay = torch.equal(once.mu, twice.mu)
    print(
        f"replay_context over 3 micro-batches (doubled forward, skipped recompute): mu bitwise equal {replay}"
    )
    ok &= replay
    dist.destroy_process_group()
    print("PASS" if ok else "FAIL")
    sys.exit(not ok)


if __name__ == "__main__":
    main()
