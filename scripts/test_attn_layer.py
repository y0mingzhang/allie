"""One compiled CausalSelfAttention of the reference (width 1536, 24 heads, BF16 weights, WSD windows)
on real rows, FlexAttention vs --attn-kernel's Triton kernels: the output and every grad, then CUDA-event
medians of forward and forward+backward per layer kind, so the fusions around attention (value-embedding
add, output gate, q/k norm, rotary) are inside the measurement. Ends with the reference's per-step sum:
24 layers (10 with value embeddings), each forward twice under eager block checkpoints.

    <runtime python> scripts/test_attn_layer.py --rows rows.npy (64 x 1025 tokens) [--profile]
"""

import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modded_medium as mm
from modded_medium import core

REPS, WIDTH, WINDOWS = 20, 1536, (11 * 128, 23 * 128)
KINDS = (  # name, layer index, value embeddings, key offset, count in the reference
    ("plain", 7, False, False, 14),
    ("ve", 2, True, False, 10),
    ("ve+key-offset", 0, True, True, 0),
)


def bench(fn):
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(REPS)]
    for _ in range(3):
        fn()
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return float(np.median([a.elapsed_time(b) for a, b in ev]))


def kernels(fn, top=12):
    """GPU kernels of one fn() call: total ms and count per kernel, largest first."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    rows = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    rows.sort(key=lambda e: -e.self_device_time_total)
    total = sum(e.self_device_time_total for e in rows) / 1e3
    out = [f"    {e.self_device_time_total / 1e3:7.3f} ms {e.count:3d}x {e.key[:100]}" for e in rows[:top]]
    return "\n".join([f"    {total:7.3f} ms total GPU", *out])


def err(a, b):
    d = (a.float() - b.float()).abs().max()
    return f"{d.item():.1e}/{(d / b.float().abs().max()).item():.1e}"


def main():
    rows = torch.as_tensor(
        np.load(sys.argv[sys.argv.index("--rows") + 1]).astype(np.int64)
    )
    x = rows[:, :-1].cuda()
    T = x.numel()
    dist.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:29531", rank=0, world_size=1
    )
    mm.configure(mm.Config(width=WIDTH, layers=24, max_tokens=T), "cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    yarn = core.Yarn(64, T)
    ctx = {b: mm.make_context(x, *WINDOWS, backend=b) for b in ("flex", "triton")}
    g = torch.Generator(device="cuda").manual_seed(0)
    h = F.rms_norm(
        torch.randn(1, T, WIDTH, generator=g, device="cuda"), (WIDTH,)
    ).bfloat16()
    ve = torch.randn(T, WIDTH, generator=g, device="cuda").bfloat16()
    dy = torch.randn(1, T, WIDTH, generator=g, device="cuda").bfloat16()
    sa = torch.tensor([0.5, 1.0], device="cuda")
    total = {"flex": 0.0, "triton": 0.0}
    print(f"{torch.cuda.get_device_name()}: {T} tokens, width {WIDTH}", flush=True)
    for name, i, use_ve, offset, count in KINDS:
        torch.manual_seed(i)
        attn = core.CausalSelfAttention(WIDTH, 64, WIDTH // 64, i).cuda()
        with torch.no_grad():
            attn.qkvo_w[3 * WIDTH :].normal_(
                0, 0.02
            )  # zero-init O would zero every attention grad
            for m in (attn.attn_gate, getattr(attn, "value_embed_gate", None)):
                if m is not None:
                    m.weight.normal_(0, 0.1)
        for p in attn.parameters():
            p.data = p.data.bfloat16()
        f = torch.compile(attn, dynamic=False, fullgraph=True)
        xin = h.detach().requires_grad_()
        vin = ve.detach().requires_grad_() if use_ve else None
        wrt = [xin, *([vin] if use_ve else []), *attn.parameters()]
        res, times = {}, {}
        for b in ("flex", "triton"):
            args = core.AttnArgs(
                vin, sa, ctx[b], WINDOWS[0], yarn.cos, yarn.sin, 0.1, offset
            )
            y = f(xin, args)
            res[b] = [y.detach(), *torch.autograd.grad(y, wrt, dy)]
            fw = bench(lambda: f(xin, args))
            fb = bench(lambda: torch.autograd.grad(f(xin, args), wrt, dy))
            times[b] = fw, fb
            if "--profile" in sys.argv:
                print(f"{name} {b} fwd+bwd kernels:")
                try:
                    print(kernels(lambda: torch.autograd.grad(f(xin, args), wrt, dy)))
                except Exception as e:  # diagnostics only
                    print(f"    profile failed: {type(e).__name__}: {e}")
            total[b] += count * (fw + fb)
        names = [
            "y",
            "dx",
            *(["dve"] if use_ve else []),
            *(n for n, _ in attn.named_parameters()),
        ]
        print(
            f"{name}: triton vs flex max abs/rel:",
            *(f"{n} {err(a, r)}" for n, a, r in zip(names, res["triton"], res["flex"])),
        )
        print(
            f"  ms fwd / fwd+bwd: flex {times['flex'][0]:.3f} / {times['flex'][1]:.3f}, triton {times['triton'][0]:.3f} / {times['triton'][1]:.3f}",
            flush=True,
        )
    print(
        f"reference step (2 fwd + bwd, 24 layers) attention modules: flex {total['flex']:.1f} ms, triton {total['triton']:.1f} ms, saved {total['flex'] - total['triton']:.1f} ms"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
