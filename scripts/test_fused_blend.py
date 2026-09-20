"""Gate of --fused-blend: modded_medium_core._fused_blend equals the eager residual
blend it replaces bit for bit (output and the x, x0, x02 and FP32 lambda gradients),
plain and under torch.utils.checkpoint, with its kernels eager and compiled; a tiny
--ckpt eager GPT gives identical logits and parameter gradients in every blend mode.

torch 2.8's CPU inductor drops emulate_precision_casts' rounding (CppVecOverrides
.to_dtype rejects use_compute_types; the dtype-convert CSE cache maps a rounded value's
upcast back to its FP32 source). round_like_triton() restores it, so the CPU compile
checks the lowering Triton codegens. With cuda, the blend checks (plus T 65536) run
on the GPU's Triton kernels instead and the GPT check is skipped.

    .venv/bin/python scripts/test_fused_blend.py [cuda]
"""

import itertools
import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch._dynamo.utils import counters
from torch._inductor.codegen import cpp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def round_like_triton():
    vec_to_dtype = cpp.CppVecOverrides.to_dtype

    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        return vec_to_dtype(x, dtype, src_dtype)

    cpp.CppVecOverrides.to_dtype = staticmethod(to_dtype)
    cpp.CppKernel.cache_dtype_convert = lambda self, *args: None


# _eager_block's run() at 7d25e76
def eager_blend(first, x, x0, x02, resid, w):
    if first:
        return (resid + w[0]) * x + w[1] * x02
    return resid * x + w[0] * x0 + w[1] * x02


def sample(shape, gen):
    """BF16 normals scaled by 2^-24..2^24; 1% at +-2^40..2^50, 1% at +-2^-139..2^-129
    (subnormal or zero in BF16), 1% signed zeros. Products and sums stay finite."""
    x = torch.randn(shape, generator=gen)
    x = x * 2.0 ** torch.randint(-24, 25, shape, generator=gen)
    u = torch.rand(shape, generator=gen)
    e = torch.randint(0, 11, shape, generator=gen).float()
    x = torch.where(u < 0.01, x.sign() * 2.0 ** (40 + e), x)
    x = torch.where((u >= 0.01) & (u < 0.02), x.sign() * 1.7 * 2.0 ** (-140 + e), x)
    x = torch.where((u >= 0.02) & (u < 0.03), x * 0, x)
    return x.bfloat16()


def bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


def blend_case(fn, first, ckpt, x, x0, x02, lam, g):
    xs = [t.detach().requires_grad_() for t in (x, x0, x02)]
    lam = lam.detach().requires_grad_()
    args = (first, *xs, lam[0], lam[1:])
    if ckpt:
        y = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    else:
        y = fn(*args)
    y.backward(g)
    return [y.detach()] + [t.grad for t in xs] + [lam.grad]


def test_blend(core, device):
    kernels = {"compiled": (core._blend_fwd, core._blend_bwd)}
    kernels["eager"] = tuple(f.__wrapped__ for f in kernels["compiled"])
    # no reliance on emulate_precision_casts: eager forward, fused backward
    kernels["bwd-only"] = (core._blend_fwd.__wrapped__, core._blend_bwd)
    lams = [
        torch.tensor(v, device=device)
        for v in (
            [1.0512345, 0.3121, -0.77777],
            [1.0 + 2**-8, 1.0 + 3 * 2**-8, -(1.0 + 2**-8)],  # BF16 rounding ties
            [0.9990001, 0.0, 0.0],  # the x0 lambdas' zero init
            [-1.9999, 1e-30, 1.3e-5],
        )
    ]
    gen = torch.Generator().manual_seed(0)
    graphs = counters["stats"]["unique_graphs"]
    shapes = list(itertools.product((1024, 4096), (512, 1536)))
    for T, d in shapes + [(65536, 1536)] * (device.type == "cuda"):
        x, x0, x02, g = (sample((1, T, d), gen).to(device) for _ in range(4))
        for lam, first, ckpt in itertools.product(lams, (True, False), (False, True)):
            want = blend_case(eager_blend, first, ckpt, x, x0, x02, lam, g)
            for name, pair in kernels.items():
                core._blend_fwd, core._blend_bwd = pair
                got = blend_case(core._fused_blend, first, ckpt, x, x0, x02, lam, g)
                ok = [same(a, b) for a, b in zip(want, got)]
                assert all(ok), (T, d, lam, first, ckpt, name, ok)
        # the backward's CUDA scale: eager g * l casts a CUDA 0-dim l to BF16 first
        s = lams[0][0].bfloat16()
        want = (g.float() * s.float()).bfloat16()
        assert same(want, kernels["compiled"][1](g, [s], [x])[0][0])
        assert g.is_cpu or same(want, g * lams[0][0])
        print(f"T {T} d {d}: output, x/x0/x02 and lambda grads bitwise eager")
    core._blend_fwd, core._blend_bwd = kernels["compiled"]
    assert counters["stats"]["unique_graphs"] - graphs >= 16, "kernels did not compile"


def gpt_case(mm, core, ckpt_blend, fused_blend):
    torch.manual_seed(0)
    cfg = mm.Config(
        width=64, head_dim=16, layers=8, max_tokens=1024, scheduled_steps=8,
        extension_steps=0, initial_batch_rows=8, ckpt="eager", ckpt_blend=ckpt_blend,
        fused_blend=fused_blend,
    )  # fmt: skip
    model = mm.create_model(cfg, device="cpu")
    # nonzero projections, embeddings and lambdas: every blend term carries signal
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2 and ("c_proj" in name or "embed" in name):
                p.normal_(0, 0.02)
        model.x0_lambdas.normal_(0, 0.3)
        model.scalars.add_(0.05 * torch.randn_like(model.scalars))
    net = torch.compile(model, dynamic=False, fullgraph=False)
    sched = core.ForwardScheduleConfig(mtp_weights=torch.ones(1), ws_short=1, ws_long=3)
    for step in range(2):
        model.zero_grad(set_to_none=True)
        gen = torch.Generator().manual_seed(step)
        rows = torch.randint(378, 2346, (1, 1025), generator=gen)
        x, y = rows[:, :-1], rows[:, 1:]
        ctx = mm.make_context(x, 128, 384, backend="dense")
        z = net(x.flatten(), y.flatten(), ctx, sched)
        (z.float() * torch.randn(z.shape, generator=gen)).sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    return z.detach(), grads


def test_gpt(mm, core):
    want, want_grads = gpt_case(mm, core, False, False)
    for ckpt_blend, fused_blend in ((False, True), (True, False), (True, True)):
        z, grads = gpt_case(mm, core, ckpt_blend, fused_blend)
        assert same(want, z), (ckpt_blend, fused_blend, "logits")
        assert grads.keys() == want_grads.keys()
        bad = [n for n in grads if not same(want_grads[n], grads[n])]
        assert not bad, (ckpt_blend, fused_blend, bad)
    print(f"GPT: every blend mode gives eager's logits and all {len(grads)} grads")


if __name__ == "__main__":
    device = torch.device(sys.argv[1] if sys.argv[1:] else "cpu")
    # a fresh inductor cache: every CPU kernel here compiles under round_like_triton()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp()
    round_like_triton()
    torch.set_num_threads(4)
    import modded_medium as mm

    test_blend(mm.core, device)
    if device.type == "cpu":
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29541", rank=0, world_size=1
        )
        test_gpt(mm, mm.core)
        dist.destroy_process_group()
    print(f"PASS --fused-blend bitwise eager on {device}")
