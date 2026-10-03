"""MoE layer tests. CUDA: the reference layer (d1536 SwiGLU, E96 top-4, 64K tokens, BF16 weights);
CPU: under TRITON_INTERPRET=1, whose BF16 dots run in FP32 and which has no libdevice, so the CPU
runs check indexing, masking, rounding points and autograd wiring, not MMA order.

layer: one MoE layer as --ckpt eager runs it (x + moe(norm(x)) compiled whole and checkpointed with
replay_context, deterministic; CPU: eager, d64 E16 top-4, 2048 tokens), 2 steps x 3 micro-batches
plus a no-grad training forward, the bias nonzero and one step per update rule, against a base
commit's layer (--base, with its shipped flags --set) bit for bit: outputs, input and parameter
gradients, the load buffer after every micro-batch, the rebalanced bias and the logged stats. Each
implementation runs in its own process.

kernels: moe_kernels.routed (output, input, weight and gate grads, and the no-grad path) against an
FP32 torch reference on ragged shapes with empty experts; its grads with expanded recomputed equal
those with it saved, bit for bit.

topk: moe_kernels.route against torch.topk (CUDA; CPU: aten_topk, an emulation of ATen's kernels)
on live, tie-heavy and special (+-0, +-inf, NaN, subnormal) scores.

counts: moe_layer.counts against the scatter_add_ histogram, eager and compiled.

seqraw: the sequence balance loss's router gradient on a row whose raw top-1 is always expert 0 but
whose bias balances the selection (the audit's example, both scores): about 0 from the biased
counts, the raw top-k formula's (DeepSeek-V3 Eq. 18) with moe_seq_raw.

    inhold.sh tests/checks/moe_nongemm.py [layer|kernels|topk|counts|seqraw ...] [--base COMMIT]
    Allie-v3.0's h192: layer --experts 256 --topk 16 --arch moe_shared_frac=0.25 --arch moe_round=16
    .venv/bin/python tests/checks/moe_nongemm.py    # CPU: reruns itself under TRITON_INTERPRET=1
"""

import argparse
import ast
import importlib
import itertools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "cc1ac00"
BASE_SET = [
    "modded_moe.TOPK_KERNEL=True",
    "modded_smoe_swiglu.EPILOGUE=True",
    "modded_smoe_aligned_linear.COMBINE=(((0,1),2),3)",
]


def bitdiff(a, b):
    """Max |difference| of the integer bit patterns (ULPs for same-sign floats); inf on a shape
    or dtype mismatch."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    if a.is_floating_point():
        ints = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        a, b = a.view(ints), b.view(ints)
    return (a.long() - b.long()).abs().max().item() if a.numel() else 0


def interpret():
    """Interpreter fixes: BF16 dots in FP32 (it multiplies the raw bits), FP32 -> BF16 to nearest
    even (it truncates), libdevice exp as numpy's (its libdevice is stubs)."""
    import numpy as np
    import triton.language as tl
    from triton.language.extra import libdevice
    from triton.runtime import interpreter as ip

    b = ip.InterpreterBuilder
    dot, cast = b.create_dot, b.cast_impl

    def f32(t):
        if t.dtype.scalar != tl.bfloat16:
            return t
        return ip.TensorHandle(
            (t.data.astype(np.uint32) << 16).view(np.float32), tl.float32
        )

    def to_bf16(self, src, ty):
        if src.dtype.scalar == tl.float32 and ty.scalar == tl.bfloat16:
            x = torch.from_numpy(np.ascontiguousarray(src.data)).bfloat16()
            return ip.TensorHandle(
                x.view(torch.int16).numpy().view(np.uint16), tl.bfloat16
            )
        return cast(self, src, ty)

    b.create_dot = lambda self, x, y, *rest: dot(self, f32(x), f32(y), *rest)
    b.cast_impl = to_bf16
    libdevice.exp = lambda x: tl.math.exp(x)


def base_scripts(base):
    out = tempfile.mkdtemp()
    repo = os.environ.get(
        "ALLIE_PROJECT_ROOT", HERE.parent
    )  # for a git-archived copy of scripts/
    git = ["git", "-C", str(repo), "archive", f"{base}:scripts"]
    tar = subprocess.run(git, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-f", "-", "-C", out], input=tar, check=True)
    return out


def run(a):
    """One implementation's layer record (subprocess): a base commit's flat scripts (--impl) or this package."""
    if a.impl:
        sys.path.insert(0, a.impl)
        import modded_moe as moe_layer
        from modded_arch import moe_dims
        from modded_moe import MoE
    else:
        from allie.model import moe as moe_layer
        from allie.model.arch import moe_dims
        from allie.model.moe import MoE

    for s in a.set:
        target, _, value = s.partition("=")
        mod, _, name = target.rpartition(".")
        assert hasattr(importlib.import_module(mod), name), target
        setattr(importlib.import_module(mod), name, ast.literal_eval(value))
    backend = "nccl" if DEV == "cuda" else "gloo"
    dist.init_process_group(
        backend, init_method=f"file://{a.dump}.store", rank=0, world_size=1
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    (e, d, t) = (96, 1536, 65536) if DEV == "cuda" else (16, 64, 2048)
    e = a.experts or e
    arch = dict(
        mlp="swiglu", moe=[e, a.topk], moe_kernel="scatter-dualgather", moe_init=0.006,
        moe_router_lr_mul=0.1, moe_gamma=1e-3, moe_seq=0.001, moe_update="sign",
        moe_score="sigmoid",
    )  # fmt: skip
    arch |= {k: ast.literal_eval(v) for k, _, v in (s.partition("=") for s in a.arch)}
    m = MoE(d, *moe_dims(d, arch)).to(DEV).train()
    with torch.no_grad():
        m.down.normal_(0, 0.02)
        m.shared_down.normal_(0, 0.02)
        m.bias.normal_(0, 0.01)  # routes that the bias moves (the moved stat)
    for n, p in m.named_parameters():
        if n != "router":
            p.data = p.data.bfloat16()

    def block(x):
        return x + m(F.rms_norm(x, (x.size(-1),)))

    fn = block if DEV == "cpu" else torch.compile(block, dynamic=False, fullgraph=True)

    def forward(x):
        return torch.utils.checkpoint.checkpoint(
            fn, x, use_reentrant=False, context_fn=moe_layer.replay_context
        )

    gen = torch.Generator(DEV).manual_seed(1)

    def rand():
        return torch.randn(t, d, device=DEV, generator=gen).bfloat16()

    rec = {}
    for i, rule in enumerate(("sign", "prop")):
        m.update = rule
        for j in range(3):
            x, dy = rand().requires_grad_(), rand()
            y = forward(x)
            y.backward(dy)
            rec[f"y{i}{j}"], rec[f"dx{i}{j}"] = y.detach(), x.grad
            rec[f"load{i}{j}"] = m.load.clone()
        with torch.no_grad():
            rec[f"y{i}-nograd"] = forward(rand())
            rec[f"load{i}-nograd"] = m.load.clone()
        rec |= {f"{n}.grad{i}": p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad(set_to_none=True)
        m.rebalance(0.7)
        rec[f"bias{i}"], rec[f"stats{i}"] = m.bias.clone(), m.stats.clone()
    torch.save({k: v.cpu() for k, v in rec.items()}, a.dump)
    dist.destroy_process_group()


def layer(a):
    tmp, out = Path(a.dump or tempfile.mkdtemp()), {}
    tmp.mkdir(parents=True, exist_ok=True)
    shape = [f"--experts={a.experts}"] * bool(a.experts) + [f"--topk={a.topk}"]
    shape += [f"--arch={s}" for s in a.arch]
    for name, impl, sets in (
        ("base", base_scripts(a.base), a.set or BASE_SET * (a.base == BASE)),
        ("new", None, []),
    ):
        cmd = [sys.executable, __file__, "run", "--dump", str(tmp / name)]
        cmd += ["--impl", str(impl)] * bool(impl)
        subprocess.run(cmd + shape + [f"--set={s}" for s in sets], check=True)
        out[name] = torch.load(tmp / name)
    base, new = out["base"], out["new"]
    assert base.keys() == new.keys()
    diff = {k: bitdiff(base[k], new[k]) for k in base}
    bad = {k: v for k, v in diff.items() if v}
    print(
        f"{'FAIL' if bad else 'PASS'} layer vs {a.base} ({DEV}): {len(diff)} tensors {bad or ''}"
    )
    return not bad


def reference(x, up, down, k, flat, gates):
    """FP32 routed SwiGLU experts: x [T, D], up [E, 2H, D], down [E, H, D], flat [T * k]."""
    xs = x.repeat_interleave(k, 0)
    out = xs.new_zeros(len(xs), down.shape[-1])
    for e in range(len(up)):
        rows = (flat == e).nonzero()[:, 0]
        a, b = (xs[rows] @ up[e].T).chunk(2, -1)
        out[rows] = (F.silu(a) * b) @ down[e]
    return (out.view(len(x), k, -1) * gates[..., None]).sum(1)


def kernels(a):
    from allie.model import moe as moe_layer
    from allie.model import moe_kernels

    gen = torch.Generator(DEV).manual_seed(2)
    ok = True
    shapes = (
        ((96, 1536, 512, 4, 8192), (128, 768, 256, 4, 4096)) if DEV == "cuda" else ()
    )
    for e, d, h, k, t in shapes + ((16, 64, 22, 4, 300), (8, 48, 40, 2, 129)):
        s = torch.rand(t, e, device=DEV, generator=gen)
        s[:, : e // 4] -= 2  # a quarter of the experts get no routes
        idx = s.topk(k, -1).indices
        gates = torch.rand(t, k, device=DEV, generator=gen).bfloat16()
        x = torch.randn(t, d, device=DEV, generator=gen).bfloat16()
        up = (torch.randn(e, 2 * h, d, device=DEV, generator=gen) / d**0.5).bfloat16()
        down = (torch.randn(e, h, d, device=DEV, generator=gen) / h**0.5).bfloat16()
        dy = torch.randn(t, d, device=DEV, generator=gen).bfloat16()
        flat = idx.flatten()
        order = flat.argsort(stable=True)
        offsets = moe_layer.counts(flat[order], e).cumsum(0)
        leaves = [v.clone().requires_grad_() for v in (x, up, down, gates)]
        y = moe_kernels.routed(
            leaves[0],
            leaves[1].transpose(1, 2),
            leaves[2],
            k,
            flat[order],
            order,
            offsets,
            leaves[3],
        )
        y.backward(dy)
        kept = [v.clone().requires_grad_() for v in (x, up, down, gates)]
        moe_kernels.routed(
            kept[0],
            kept[1].transpose(1, 2),
            kept[2],
            k,
            flat[order],
            order,
            offsets,
            kept[3],
            False,
        ).backward(dy)
        remat = all(torch.equal(u.grad, v.grad) for u, v in zip(leaves, kept))
        with torch.no_grad():
            y0 = moe_kernels.routed(
                x, up.transpose(1, 2), down, k, flat[order], order, offsets, gates
            )
        refs = [v.float().requires_grad_() for v in (x, up, down, gates)]
        r = reference(*refs[:3], k, flat, refs[3])
        r.backward(dy.float())
        errs = [
            ((u.float() - v).norm() / v.norm()).item()
            for u, v in zip(
                (y, *(p.grad for p in leaves)), (r, *(p.grad for p in refs))
            )
        ]
        exact = torch.equal(y0, y.detach())
        good = max(errs) < 2e-2 and exact and remat
        print(
            f"  E{e} d{d} h{h} k{k} T{t}: rel err out/dx/dup/ddown/dgates {errs}, no-grad == grad {exact},"
            f" remat == saved {remat}"
        )
        ok &= good
    print(f"{'PASS' if ok else 'FAIL'} routed kernels vs FP32 reference ({DEV})")
    return ok


def aten_row(row, key, k):
    """One row of torch.topk(x, k) as ATen's CUDA kernels compute it: gatherTopK in the radix
    order (key), then SortUtils.cuh's 32-slot bitonicSortKVInPlace (16 threads) with GTOp."""
    kth = sorted(key, reverse=True)[k - 1]
    above = [i for i, v in enumerate(key) if v > kth]
    slot = above + [i for i, v in enumerate(key) if v == kth][: k - len(above)]
    slot += [None] * (32 - k)
    size = 2
    while size <= 32:
        stride = size // 2
        while stride:
            for t in range(16):
                p, up = 2 * t - (t & (stride - 1)), size < 32 and t & (size // 2) != 0
                a, b = slot[p], slot[p + stride]
                swap = (
                    b is None
                    or a is not None
                    and (row[a] != row[a] and row[b] == row[b] or row[a] > row[b])
                )
                if swap == up:
                    slot[p], slot[p + stride] = b, a
            stride //= 2
        size *= 2
    return slot[:k]


def aten_topk(x, k):
    bits = x.view(torch.int32)
    radix = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    radix = torch.where(x.isnan(), 2**31 - 1, radix)
    rows = [aten_row(r, key, k) for r, key in zip(x.tolist(), radix.tolist())]
    idx = torch.tensor(rows, dtype=torch.long)
    return x.gather(1, idx), idx


def topk(a):
    """Router top-ks against torch.topk: routing indices of s + bias for k, values and indices of
    s for k + 1, and zeros without stats, at E 96 and 128 (the kernel's largest)."""
    t, k = (65536, 4) if DEV == "cuda" else (256, 4)
    gen = torch.Generator(DEV).manual_seed(4)
    return all([topk_at(t, e, k, gen) for e in (96, 128)])


def topk_at(t, e, k, gen):
    from allie.model import moe_kernels

    ref = (lambda x, k: tuple(torch.topk(x, k, dim=-1))) if DEV == "cuda" else aten_topk

    def rand(*shape):
        return torch.rand(*shape, device=DEV, generator=gen)

    live = torch.sigmoid(8 * rand(t, e) - 4), 0.02 * rand(e) - 0.01
    grid = (torch.randint(0, 6, (t, e), device=DEV, generator=gen) / 8).float()
    ties = grid, torch.randint(-2, 3, (e,), device=DEV, generator=gen) / 8
    odd = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 0.5, 2**-149]
    )
    pick = torch.randint(0, len(odd), (t, e), device=DEV, generator=gen)
    special = (
        torch.where(rand(t, e) < 0.3, odd.to(DEV)[pick], grid),
        torch.zeros(e, device=DEV),
    )
    ok = True
    for name, (s, bias) in {"live": live, "ties": ties, "special": special}.items():
        s, bias = s.contiguous(), bias.float()
        got = moe_kernels.route(s, bias, k, True)
        want = ref(s + bias, k)[1], *ref(s, k + 1)
        diff = [bitdiff(w.cpu(), g.cpu()) for w, g in zip(want, got)]
        zeros = moe_kernels.route(s, bias, k, False)
        diff.append(bitdiff(want[0].cpu(), zeros[0].cpu()))
        diff.append(max(z.abs().max().item() for z in zeros[1:]))
        print(
            f"  {name}: routing idx, stats values, stats idx, no-stats idx, no-stats zeros: {diff}"
        )
        ok &= not any(diff)
    print(
        f"{'PASS' if ok else 'FAIL'} router top-k at {t} x {e}, k {k}: bitwise torch.topk"
    )
    return ok


def counts(a):
    from allie.model.moe import counts as new

    def old(flat, e):
        return flat.new_zeros(e).scatter_add_(0, flat, torch.ones_like(flat))

    g = torch.Generator().manual_seed(0)
    ok = True
    for e, k, t in itertools.product((64, 96, 128), (4, 8), (1024, 65536)):
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda f, e=e: new(f[f.argsort(stable=True)], e), fullgraph=True
        )
        s = torch.rand(t, e, generator=g)
        few = torch.randperm(e, generator=g) < 2 * k
        routings = (
            s.topk(k, -1).indices,
            (s + torch.linspace(8, 0, e)).topk(k, -1).indices,
            (s + 2 * few).topk(k, -1).indices,  # e - 2k experts empty
            torch.full((t, k), e - 1),  # every route to one expert
        )
        for idx in routings:
            flat = idx.flatten().to(DEV)
            want = old(flat, e)
            for c in (new(flat[flat.argsort(stable=True)], e), compiled(flat)):
                ok &= c.dtype == want.dtype and torch.equal(c, want)
    print(
        f"{'PASS' if ok else 'FAIL'} counts vs scatter_add_ ({DEV}), eager and compiled"
    )
    return ok


def seqraw(a):
    from allie.model.moe import MoE

    if not dist.is_initialized():
        store = f"file://{tempfile.mkdtemp()}/store"
        dist.init_process_group("gloo", init_method=store, rank=0, world_size=1)
    t, e, d = 1024, 4, 64
    want = torch.tensor([0.9, 0.2, 0.2, 0.2]).repeat(t, 1)
    want[torch.arange(t), torch.arange(t) % e] += 0.01
    inverse = dict(sigmoid=torch.logit, sqrtsoftplus=lambda p: p.square().expm1().log())
    ok = True
    for score, inv in inverse.items():
        h = F.pad(inv(want.double()), (0, d - e)).bfloat16().to(DEV)
        norms = []
        for raw in (False, True):
            m = MoE(d, e, 1, 64, 0, seq=1.0, score=score, seq_raw=raw).to(DEV).train()
            with torch.no_grad():
                m.router.zero_()[:, :e] = torch.eye(e)
                m.bias.copy_(torch.tensor([-0.7, 0, 0, 0]))
            for n, p in m.named_parameters():
                if n != "router":
                    p.data = p.data.bfloat16()
            m(h).backward(torch.zeros(t, d, dtype=h.dtype, device=DEV))
            r = m.router.detach().clone().requires_grad_()
            z = F.linear(h.float(), r)
            s = z.sigmoid() if score == "sigmoid" else F.softplus(z).sqrt()
            top = s.topk(1).indices if raw else (s + m.bias).topk(1).indices
            sel = torch.zeros_like(s).scatter_(1, top, 1.0).mean(0)
            loss = t / 8 * (sel * e * (s / s.sum(-1, keepdim=True)).mean(0)).sum()
            ref = torch.autograd.grad(loss, r)[0]
            loads = m.load[:e].tolist(), (sel * t).tolist()
            ok &= loads[0] == [t / e] * e and loads[1] == (
                [t, 0, 0, 0] if raw else loads[0]
            )
            ok &= torch.allclose(m.router.grad, ref, rtol=1e-4, atol=1e-9)
            norms.append(m.router.grad.norm().item())
        ok &= norms[0] < 1e-6 * norms[1]
        print(f"  {score}: router grad norm biased {norms[0]:.3g}, raw {norms[1]:.3g}")
    print(
        f"{'PASS' if ok else 'FAIL'} sequence balance loss from the raw top-k ({DEV})"
    )
    return ok


def main():
    if (
        DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1"
    ):  # before triton's import
        env = os.environ | {"TRITON_INTERPRET": "1"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    if DEV == "cpu":
        interpret()
    p = argparse.ArgumentParser()
    tests = dict(layer=layer, kernels=kernels, topk=topk, counts=counts, seqraw=seqraw)
    p.add_argument("cmd", nargs="*", choices=(*tests, "run"))
    p.add_argument("--impl")
    p.add_argument("--dump", help="layer: keep the two records in this directory")
    p.add_argument("--base", default=BASE, help="layer: the reference commit")
    p.add_argument(
        "--set", action="append", default=[], help="layer: MOD.NAME=VALUE (base side)"
    )
    p.add_argument(
        "--experts", type=int, default=0, help="layer: E (default 96, CPU 16)"
    )
    p.add_argument("--topk", type=int, default=4, help="layer: k")
    p.add_argument(
        "--arch",
        action="append",
        default=[],
        help="layer: KEY=VALUE arch override (both sides)",
    )
    a = p.parse_args()
    if a.cmd == ["run"]:
        return run(a)
    torch.use_deterministic_algorithms(True)
    results = [tests[c](a) for c in a.cmd or tests]
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
