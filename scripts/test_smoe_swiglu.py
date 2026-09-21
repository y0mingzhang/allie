"""--moe-swiglu-epilogue (modded_smoe_swiglu) bit for bit against scatter-dualgather + compiled SwiGLU.

cpu (TRITON_INTERPRET=1, small shapes, ragged dims, empty experts): the interpreter's BF16 dots
run in FP32 and it has no libdevice (numpy exp), so this checks indexing, masking, where BF16
rounds and the autograd wiring, not MMA order or libdevice. kernels: fused up (pre, y) and fused
down dgrad + SwiGLU backward against the unfused kernels followed by inductor's generated
elementwise kernels (_inductor_fwd/bwd: torch 2.10's code for MoE.act, verbatim); layer: MoE
output, input and parameter grads (2 micro-batches, eager checkpoints, eager and compiled) and a
no-grad forward, flag on against flag off with MoE.act run as those inductor kernels.
cuda (reference layer: E96 top-4, d1536, h512, 64K tokens): kernels against the real unfused
kernels and torch.compile'd MoE.act and its autograd backward, for every tile config, timed;
the fused kernels against the copied inductor kernels; then test_moe_nongemm.py layer (compiled,
eager checkpoints, deterministic, all grads, loads, bias, stats) against perf-int4 with the flag.
sm89 (no GPU): offline compile of the fused kernels per config, registers, spills, shared memory.

    inhold.sh /abs/scripts/test_smoe_swiglu.py [--base 211415c]
    .venv/bin/python scripts/test_smoe_swiglu.py --device cpu|sm89
"""

import argparse
import copy
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

HERE = Path(__file__).resolve().parent
SHAPE = 96, 65536, 4, 1536, 512  # E, T, k, d, h
UP = (
    (128, 128, 64, 8, 3),
    (128, 64, 32, 4, 4),
    (128, 128, 32, 8, 4),
    (64, 128, 64, 4, 3),
)
DX = (64, 128, 64, 8, 3), (128, 128, 64, 8, 3), (128, 64, 64, 8, 3), (64, 256, 64, 8, 3)


@triton.jit
def _inductor_fwd(in_ptr0, out_ptr0, xnumel, H: tl.constexpr, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % H
    x1 = xindex // H
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2 * H * x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (H + x0 + 2 * H * x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)


@triton.jit
def _inductor_bwd(
    in_ptr0, in_ptr1, out_ptr0, xnumel, H: tl.constexpr, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % (2 * H)
    x1 = xindex // (2 * H)
    x2 = xindex
    tmp0 = x0
    tmp3 = tl.full([1], H, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (H * x1 + (x0)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (H + 2 * H * x1 + (x0)), tmp4 & xmask, other=0.0).to(
        tl.float32
    )
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (2 * H * x1 + (x0)), tmp4 & xmask, other=0.0).to(
        tl.float32
    )
    tmp10 = tmp9.to(tl.float32)
    tmp11 = -tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = tmp16 * tmp13
    tmp18 = tmp8 * tmp17
    tmp19 = tmp13 - tmp17
    tmp20 = tmp10 * tmp19
    tmp21 = tmp20 + tmp13
    tmp22 = tmp18 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp4, tmp23, tmp24)
    tmp26 = tmp0 >= tmp3
    tmp29 = tl.load(in_ptr0 + (H * x1 + ((-H) + x0)), tmp26 & xmask, other=0.0).to(
        tl.float32
    )
    tmp30 = tl.load(in_ptr1 + (2 * H * x1 + ((-H) + x0)), tmp26 & xmask, other=0.0).to(
        tl.float32
    )
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.sigmoid(tmp31)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp29 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp25, tmp37)
    tl.store(out_ptr0 + (x2), tmp38, xmask)


@torch.library.custom_op("test_swiglu::fwd", mutates_args={"y"})
def inductor_fwd(pre: torch.Tensor, y: torch.Tensor) -> None:
    _inductor_fwd[(triton.cdiv(y.numel(), 1024),)](pre, y, y.numel(), y.shape[1], 1024)


@torch.library.custom_op("test_swiglu::bwd", mutates_args={"dpre"})
def inductor_bwd(dy: torch.Tensor, pre: torch.Tensor, dpre: torch.Tensor) -> None:
    grid = (triton.cdiv(pre.numel(), 1024),)
    _inductor_bwd[grid](dy, pre, dpre, pre.numel(), dy.shape[1], 1024)


class InductorAct(torch.autograd.Function):
    """MoE.act (swiglu) as the compiled layer runs it, through the copied inductor kernels."""

    @staticmethod
    def forward(ctx, pre):
        y = pre.new_empty(pre.shape[0], pre.shape[1] // 2)
        inductor_fwd(pre, y)
        ctx.save_for_backward(pre)
        return y

    @staticmethod
    def backward(ctx, dy):
        (pre,) = ctx.saved_tensors
        dpre = torch.empty_like(pre)
        inductor_bwd(dy.contiguous(), pre, dpre)
        return dpre


def inductor_act(pre):
    return InductorAct.apply(pre.contiguous())


def compiled_act(pre):  # MoE.act, as the layer's compiled graph fuses it
    a, b = pre.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


def interpret():
    """Interpreter fixes: BF16 dots in FP32 (it multiplies the raw bits), FP32 -> BF16 to nearest
    even (it truncates), libdevice exp as numpy's (its libdevice is stubs)."""
    import numpy as np
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


def bits(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    i = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return int((a.view(i).long() - b.view(i).long()).abs().max()) if a.numel() else 0


def routes(e, t, k, scenario, dev):
    import modded_moe

    g = torch.Generator(dev).manual_seed(5)
    s = torch.rand(t, e, device=dev, generator=g)
    if scenario == "skew":  # half the experts empty, one holding most rows
        s[:, : e // 2] -= 2
        s[:, e - 1] += 0.5
    flat = s.topk(k, -1).indices.flatten()
    order = flat.argsort(stable=True)
    se = flat[order]
    return se, order, modded_moe.counts(se, e).cumsum(0)  # as MoE.forward


def data(e, t, k, d, h, dev, exact):
    """x, up^T, down, gates, output grad; exact: dyadic values whose dots are exact in FP32."""
    g = torch.Generator(dev).manual_seed(6)

    def rnd(*shape, scale):
        if exact:
            return torch.randint(-8, 9, shape, device=dev, generator=g) * scale
        return torch.randn(*shape, device=dev, generator=g) * scale

    x = rnd(t, d, scale=1 / 8 if exact else 1).bfloat16()
    up = rnd(e, 2 * h, d, scale=2**-7 if exact else 0.03).bfloat16()
    down = rnd(e, h, d, scale=2**-7 if exact else 0.03).bfloat16()
    gates = torch.rand(t, k, device=dev, generator=g).bfloat16()
    grad = rnd(t, d, scale=1 / 8 if exact else 1).bfloat16()
    return x, up.transpose(1, 2), down, gates, grad


def bench(f, *args, **kwargs):
    return triton.testing.do_bench(lambda: f(*args, **kwargs))


def unfused_up(x, up_t, se, order, offs, k, act):
    """scatter-dualgather's pre (the up GEMM) and y."""
    import modded_smoe_aligned_linear as aligned

    pre = aligned.matmul(x, up_t, se, order, offs, k, False, True)
    p = pre.detach().requires_grad_()
    return pre, p, act(p)


def unfused_dx(grad, down_t, gates, order, offs, p, y):
    """scatter-dualgather's d pre: the gated down dgrad, then the act's backward."""
    import modded_smoe_gated as gated

    dy = gated.input_grad(grad, down_t, gates, order, offs)
    return torch.autograd.grad(y, p, dy, retain_graph=True)[0]


def kernels(dev, e, t, k, d, h, act, configs, exact=False, time=False):
    """Fused against unfused for every (up, dx) config: bit diffs of pre, y, y without pre, d pre;
    time: ms of each fused kernel against its unfused pair (GEMM + elementwise kernel)."""
    import modded_smoe_swiglu as swiglu

    worst, out = 0, []
    for scenario in ("random", "skew"):
        se, order, offs = routes(e, t, k, scenario, dev)
        x, up_t, down, gates, grad = data(e, t, k, d, h, dev, exact)
        up_args = x, up_t, order, offs, k
        pre, p, y = unfused_up(x, up_t, se, order, offs, k, act)
        dx_args = grad, down.permute(0, 2, 1), gates, order, offs
        dpre = unfused_dx(*dx_args, p, y)
        if time:
            ms = (
                bench(unfused_up, x, up_t, se, order, offs, k, act),
                bench(unfused_dx, *dx_args, p, y),
            )
            out.append(
                f"  {scenario} unfused: up GEMM + act {ms[0]:.3f} ms, down dgrad + act backward {ms[1]:.3f} ms"
            )
            best = {}
        for cu, cd in configs:
            fp, fy = swiglu.up(*up_args, config=cu)
            fy2 = swiglu.up(*up_args, save=False, config=cu)[1]
            fd = swiglu.dx(*dx_args, pre, config=cd)
            diff = [bits(fp, pre), bits(fy, y), bits(fy2, y), bits(fd, dpre)]
            worst = max(worst, *diff)
            row = f"  {scenario} up {cu} dx {cd}: bit diff pre, y, y (no pre), d pre {diff}"
            if time:
                ms = (
                    bench(swiglu.up, *up_args, config=cu),
                    bench(swiglu.dx, *dx_args, pre, config=cd),
                )
                row += f"; fused up {ms[0]:.3f} ms, dx {ms[1]:.3f} ms"
                best = {
                    n: min(best.get(n, (1e9,)), (v, c))
                    for n, v, c in (("up", ms[0], cu), ("dx", ms[1], cd))
                }
            out.append(row)
        if time:
            out.append(
                f"  {scenario} fastest: "
                + ", ".join(f"{n} {c} {v:.3f} ms" for n, (v, c) in best.items())
            )
    return worst, out


def layer_cpu():
    """MoE with the flag on against off (MoE.act as inductor's kernels), eager and compiled."""
    import modded_moe
    import modded_smoe_swiglu as swiglu
    import torch._dynamo
    import torch.utils.checkpoint

    modded_moe.MoE.act = lambda self, x: inductor_act(x)
    modded_moe.BLOCK_RECOMPUTE = True
    worst = 0
    for scenario in ("random", "skew"):
        torch.manual_seed(701)
        m = modded_moe.MoE(
            64,
            8,
            2,
            24,
            48,
            kind="swiglu",
            kernel="scatter-dualgather",
            init=0.006,
            seq=0,
        )
        with torch.no_grad():
            m.down.normal_(0, 0.02)
            m.shared_down.normal_(0, 0.02)
            if scenario == "skew":
                m.bias[:4] = -1e4
        for n, p in m.named_parameters():
            if n != "router":
                p.data = p.data.bfloat16()
        g = torch.Generator().manual_seed(340)
        xs = [torch.randn(256, 64, generator=g).bfloat16() for _ in range(3)]
        rs = [torch.randn(256, 64, generator=g) for _ in range(2)]
        for compiled in (False, True):
            res = []
            for flag in (False, True):
                swiglu.EPILOGUE = flag
                mm = copy.deepcopy(m).train()
                f = torch.compile(mm, fullgraph=True, dynamic=False) if compiled else mm
                rec = {}
                for i, r in enumerate(rs):
                    x = xs[i].clone().requires_grad_()
                    y = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
                    (y.float() * r).sum().backward()
                    rec |= {f"y{i}": y.detach(), f"dx{i}": x.grad}
                with torch.no_grad():
                    rec["nograd"] = f(xs[2])
                rec |= {n: p.grad for n, p in mm.named_parameters()}
                res.append(rec)
                torch._dynamo.reset()
            diff = {n: bits(v, res[1][n]) for n, v in res[0].items()}
            worst = max(worst, *diff.values())
            bad = {n: v for n, v in diff.items() if v} or "none"
            how = "compiled" if compiled else "eager"
            print(
                f"  layer {scenario} {how}: {len(diff)} tensors, mismatches {bad}",
                flush=True,
            )
    swiglu.EPILOGUE = False
    return worst


def cpu():
    import modded_smoe_swiglu as swiglu

    interpret()
    torch.use_deterministic_algorithms(True)
    worst = 0
    # the interpreter sums each BLOCK_K's products apart: random data needs the reference's BLOCK_K
    exact = [
        ((32, 16, 16, 4, 2), (16, 32, 32, 4, 2)),
        ((16, 32, 64, 4, 2), (128, 256, 64, 8, 3)),
    ]
    rand = [
        ((32, 16, 32, 4, 2), (16, 32, 64, 4, 2)),
        ((128, 128, 32, 8, 3), (128, 128, 64, 8, 3)),
    ]
    for name, configs in (("exact-sum", exact), ("random", rand)):
        w, out = kernels(
            "cpu", 7, 45, 3, 80, 40, inductor_act, configs, name == "exact-sum"
        )
        worst = max(worst, w)
        print(f"kernels, {name} data", *out, sep="\n", flush=True)
    swiglu.UP, swiglu.DX = rand[0]
    return max(worst, layer_cpu())


def cuda(base):
    import modded_smoe_swiglu as swiglu

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    act = torch.compile(compiled_act, fullgraph=True, dynamic=False)
    worst, out = kernels("cuda", *SHAPE, act, list(zip(UP, DX)), time=True)
    print("kernels at the reference shape", *out, sep="\n", flush=True)
    w, out = kernels("cuda", *SHAPE, inductor_act, [(swiglu.UP, swiglu.DX)])
    print(
        f"fused against the GEMMs + copied inductor kernels: max bit diff {w}",
        *out,
        sep="\n",
        flush=True,
    )
    worst = max(worst, w)
    cmd = [sys.executable, str(HERE / "test_moe_nongemm.py"), "layer", "--base", base]
    cmd += ["--set", "modded_smoe_swiglu.EPILOGUE=True"]
    layer = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(
        layer.stdout,
        layer.stderr[-3000:] if layer.returncode else "",
        sep="",
        flush=True,
    )
    verdict = [s for s in layer.stdout.splitlines() if s.startswith(("PASS", "FAIL"))]
    return worst if verdict and verdict[-1].startswith("PASS") else max(worst, 1)


def sm89():
    """ptxas -v of the fused kernels (and the gated dgrad they replace) at the reference shape."""
    import modded_smoe_gated as gated
    import modded_smoe_swiglu as swiglu
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    e, _, k, d, h = SHAPE
    ptxas = f"{triton.__path__[0]}/backends/nvidia/bin/ptxas"
    ptrs = {"X", "W", "PRE", "Y", "DY", "GATES", "OUT"}
    idx = {"ORDER", "OFF", "TILES"}

    def usage(fn, config, **const):
        bm, bn, bk, warps, stages = config
        sig = {
            n: "*bf16" if n in ptrs else "*i64" if n in idx else "constexpr"
            for n in fn.arg_names
        }
        const |= {
            "E": e,
            "D": d,
            "H": h,
            "BM": bm,
            "BN": bn,
            "BK": bk,
            "SEARCH": e.bit_length() + 1,
        }
        attrs = {
            (i,): [["tt.divisibility", 16]]
            for i, t in enumerate(sig.values())
            if t[0] == "*"
        }
        src = ASTSource(fn, sig, {n: v for n, v in const.items() if n in sig}, attrs)
        options = {"num_warps": warps, "num_stages": stages}
        kernel = triton.compile(src, target=GPUTarget("cuda", 89, 32), options=options)
        with tempfile.NamedTemporaryFile("w", suffix=".ptx") as f:
            f.write(kernel.asm["ptx"])
            f.flush()
            cmd = [ptxas, "-arch=sm_89", "-v", f.name, "-o", os.devnull]
            log = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
        regs = re.search(r"Used (\d+) registers", log).group(1)
        spill = re.search(
            r"(\d+) bytes spill stores, (\d+) bytes spill loads", log
        ).groups()
        return f"{regs} registers, spill {spill[0]}/{spill[1]} B, shared {kernel.metadata.shared} B"

    w = {"SW1": 1, "SW2": d}
    for c in UP:
        info = usage(swiglu._up, c, SX0=d, SX1=1, SW0=2 * h * d, TOPK=k, SAVE=True, **w)
        print("up", c, info, flush=True)
    for c in (*DX, (128, 256, 64, 8, 3)):
        print(
            "dx",
            c,
            usage(swiglu._dx, c, SY0=d, SY1=1, SW0=h * d, FAN=k, **w),
            flush=True,
        )
    c = (128, 256, 64, 8, 3)
    print("gated dgrad", c, usage(gated._dx, c, SY0=d, SY1=1, SW0=h * d, FAN=k, **w))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu", "sm89"))
    p.add_argument("--base", default="211415c", help="perf-int4")
    a = p.parse_args()
    if (
        a.device == "cpu" and os.environ.get("TRITON_INTERPRET") != "1"
    ):  # before triton's import
        env = os.environ | {"TRITON_INTERPRET": "1"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(HERE))
    worst = (
        cuda(a.base) if a.device == "cuda" else {"cpu": cpu, "sm89": sm89}[a.device]()
    )
    print(
        f"{'FAIL' if worst else 'PASS'} swiglu epilogue ({a.device}): max bit diff {worst}",
        flush=True,
    )
    raise SystemExit(1 if worst else 0)


if __name__ == "__main__":
    main()
