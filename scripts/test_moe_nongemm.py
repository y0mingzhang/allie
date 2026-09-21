"""Gates of the MoE layer's non-GEMM speedups against perf-int (a522ee6), bit for bit.

layer: one MoE layer as --ckpt eager runs it (x + moe(norm(x)) compiled whole by _eager_block,
checkpointed, deterministic), 2 steps x 3 micro-batches plus a no-grad training forward, the
bias nonzero and one step per update rule. Each implementation runs in its own process; outputs,
input and parameter gradients, the load buffer after every micro-batch, the rebalanced bias and
the logged stats must match perf-int's bitwise. --set MOD.NAME=VALUE switches (new side only).
CUDA: the candidate layer (d1536 SwiGLU, E96 top-4, 64K tokens, BF16 weights), fwd+bwd ms and
the kernels whose time changed. CPU: 2048 tokens of a d64 E16 top-4 layer (pad kernel).

orders (CUDA): which FP32 add order over the top-k (any of the 15 add trees for k = 4) equals
cuBLAS's gates @ expanded on 64K x 4 x 1536 BF16 inputs of wide dynamic range, eagerly and
compiled with the shared-expert add and residual as the layer fuses them.

topk: modded_moe_topk's router top-ks against torch.topk (CUDA; CPU: aten_topk, an emulation of
ATen's kernels, under TRITON_INTERPRET=1) on live, tie-heavy and special (+-0, +-inf, NaN,
subnormal) scores, and their times.

    inhold.sh scripts/test_moe_nongemm.py layer [--eager] [--no-ckpt] [--base COMMIT] [--set ...]
    inhold.sh scripts/test_moe_nongemm.py orders
    inhold.sh scripts/test_moe_nongemm.py layer --set 'modded_smoe_aligned_linear.COMBINE=(0,1,2,3)'
    inhold.sh scripts/test_moe_nongemm.py topk
    inhold.sh scripts/test_moe_nongemm.py layer --set modded_moe.TOPK_KERNEL=True
"""

import argparse
import ast
import functools
import importlib
import itertools
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton.testing

BASE = "a522ee6"
HERE = Path(__file__).resolve().parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def bitdiff(a, b):
    """Max |difference| of the integer bit patterns (ULPs for same-sign floats); inf on a shape
    or dtype mismatch."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    if a.is_floating_point():
        ints = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        a, b = a.view(ints), b.view(ints)
    return (a.long() - b.long()).abs().max().item() if a.numel() else 0


def base_scripts(base=BASE):
    out = tempfile.mkdtemp()
    git = ["git", "-C", str(HERE.parent), "archive", f"{base}:scripts"]
    tar = subprocess.run(git, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-f", "-", "-C", out], input=tar, check=True)
    assert os.path.exists(f"{out}/modded_moe.py"), out
    return out


def build(dev, kernel):
    from modded_arch import moe_dims
    from modded_moe import MoE

    torch.manual_seed(0)
    if dev == "cuda":
        arch = {"mlp": "swiglu", "moe_init": 0.006, "moe_seq": 0.001}
        arch |= {"moe": [96, 4], "moe_kernel": kernel or "scatter-dualgather"}
        m, t = MoE(1536, *moe_dims(1536, arch)), 65536
    else:
        m = MoE(64, 16, 4, 32, 64, kind="swiglu", seq=0.001, kernel=kernel or "pad")
        t = 2048
    m = m.to(dev).train()
    with torch.no_grad():
        m.down.normal_(0, 0.02)
        m.shared_down.normal_(0, 0.02)
        m.bias.normal_(0, 0.01)  # routes that the bias moves (the moved stat)
    for n, p in m.named_parameters():
        if n != "router":
            p.data = p.data.bfloat16()
    return m, t


class Block(torch.nn.Module):
    layer_idx = 1

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def _forward(self, x, attn_args):
        import modded_medium_core as core

        return x + self.mlp(core.norm(x))


def run(a):
    sys.path.insert(0, a.impl)
    import modded_medium_core as core
    import modded_moe

    for s in a.set:
        target, _, value = s.partition("=")
        mod, _, name = target.rpartition(".")
        assert hasattr(importlib.import_module(mod), name), target
        setattr(importlib.import_module(mod), name, ast.literal_eval(value))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    modded_moe.BLOCK_RECOMPUTE = True
    m, t = build(dev, a.kernel)
    blk = Block(m)
    if a.eager:
        blk._compiled = blk._forward
    gen = torch.Generator(dev).manual_seed(1)

    def rand():
        return torch.randn(t, m.router.shape[1], device=dev, generator=gen).bfloat16()

    def step(x, dy):
        y = core._eager_block(blk, x, None, ckpt=a.ckpt)
        y.backward(dy)
        return y

    rec = {}
    for i, rule in enumerate(("sign", "prop")):
        m.update = rule
        for j in range(3):
            x, dy = rand().requires_grad_(), rand()
            rec[f"y{i}{j}"] = step(x, dy).detach()
            rec[f"dx{i}{j}"], rec[f"load{i}{j}"] = x.grad, m.load.clone()
        with torch.no_grad():
            rec[f"y{i}-nograd"] = core._eager_block(blk, rand(), None, ckpt=a.ckpt)
            rec[f"load{i}-nograd"] = m.load.clone()
        rec |= {f"{n}.grad{i}": p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad(set_to_none=True)
        m.rebalance(0.7)
        rec[f"bias{i}"], rec[f"stats{i}"] = m.bias.clone(), m.stats.clone()
    acts = [torch.profiler.ProfilerActivity.CPU]
    acts += [torch.profiler.ProfilerActivity.CUDA] * (dev == "cuda")
    with torch.profiler.profile(activities=acts) as prof:
        step(rand().requires_grad_(), rand())
        if dev == "cuda":
            torch.cuda.synchronize()
    events = prof.key_averages()
    topk = sum(e.count for e in events if e.key == "aten::topk")
    kernels = {}
    for e in events:
        if e.device_type == torch.autograd.DeviceType.CUDA:
            name = re.sub(r"_\d+$", "", e.key)  # inductor's kernel numbering
            n, ms = kernels.get(name, (0, 0.0))
            kernels[name] = n + e.count, ms + e.device_time_total / 1e3
    ms = []
    if dev == "cuda":
        x, dy = rand().requires_grad_(), rand()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(8):
            start.record()
            for _ in range(4):
                step(x, dy)
            end.record()
            end.synchronize()
            ms.append(start.elapsed_time(end) / 4)
    rec = {k: v.cpu() for k, v in rec.items()}
    torch.save({"rec": rec, "ms": ms, "topk": topk, "kernels": kernels}, a.dump)


def layer(a):
    flags = ["--eager"] * a.eager + ["--no-ckpt"] * (not a.ckpt)
    flags += ["--kernel", a.kernel] * bool(a.kernel)
    out = {}
    tmp = Path(tempfile.mkdtemp())
    for name, impl, extra in (
        ("base", base_scripts(a.base), []),
        ("new", HERE, [f"--set={s}" for s in a.set]),
    ):
        cmd = [sys.executable, __file__, "run", "--impl", str(impl)]
        cmd += ["--dump", str(tmp / f"{name}.pt"), *flags, *extra]
        subprocess.run(cmd, check=True)
        out[name] = torch.load(tmp / f"{name}.pt")
    base, new = out["base"]["rec"], out["new"]["rec"]
    assert base.keys() == new.keys()
    diff = {k: bitdiff(base[k], new[k]) for k in base}
    bad = {k: v for k, v in diff.items() if v}
    kb, kn = out["base"]["kernels"], out["new"]["kernels"]
    for name in sorted(kb.keys() | kn.keys()):
        (cb, tb), (cn, tn) = kb.get(name, (0, 0)), kn.get(name, (0, 0))
        if abs(tn - tb) > 0.05:
            print(f"  {tb:7.3f} ms {cb:3d}x -> {tn:7.3f} ms {cn:3d}x  {name[:80]}")
    what = f"layer {' '.join(flags + a.set) or 'default'}: {len(diff)} tensors"
    what += (
        f"; topk calls per micro-batch {out['base']['topk']} -> {out['new']['topk']}"
    )
    if out["new"]["ms"]:
        ms = [statistics.median(out[n]["ms"]) for n in ("base", "new")]
        what += f"; fwd+bwd ms {ms[0]:.3f} -> {ms[1]:.3f}"
    verdict = "FAIL" if bad else "PASS"
    print(
        f"{verdict} {what}; max bit diff {max(diff.values())} {bad or ''}", flush=True
    )
    return not bad


def trees(leaves):
    """Every add tree over leaves up to commutativity (FP adds commute but do not associate)."""
    if len(leaves) == 1:
        yield leaves[0]
        return
    first, rest = leaves[0], leaves[1:]
    for r in range(len(rest)):
        for left in itertools.combinations(rest, r):
            right = tuple(x for x in rest if x not in left)
            yield from itertools.product(trees((first, *left)), trees(right))


def orders(a):
    from modded_smoe_aligned_linear import combine

    t, k, d, dev = a.tokens, a.topk, 1536, "cuda"
    torch.use_deterministic_algorithms(True)
    gen = torch.Generator(dev).manual_seed(3)

    def normal(*shape):
        return torch.randn(*shape, device=dev, generator=gen)

    # expert outputs over 2^-20..2^20: the top-k sums round, so add orders differ
    scale = 2.0 ** torch.randint(-20, 21, (t, k, d), device=dev, generator=gen)
    e = (normal(t, k, d) * scale).bfloat16()
    w = torch.sigmoid(normal(t, k))
    # FP32, cast to BF16 as MoE.experts_out does
    w = w * (k**0.5 / w.sum(-1, keepdim=True))
    s, x = (normal(t, d) * 0.1).bfloat16(), normal(t, d).bfloat16()
    ref = (w.bfloat16().unsqueeze(1) @ e).squeeze(1)
    found = []
    for tree in trees(tuple(range(k))):
        got = combine(e, w.bfloat16(), tree)
        n = (got.view(torch.int16) != ref.view(torch.int16)).sum().item()
        print(
            f"  {tree}: {n} of {ref.numel()} differ, max bit diff {bitdiff(ref, got)}"
        )
        found += [tree] * (n == 0)
    ok = bool(found)
    for tree in found:  # compiled as the layer fuses it: + shared expert, + residual
        old = torch.compile(
            lambda e, w, s, x: x + ((w.bfloat16().unsqueeze(1) @ e).squeeze(1) + s)
        )
        new = torch.compile(
            lambda e, w, s, x, tree=tree: x + (combine(e, w.bfloat16(), tree) + s)
        )
        n = bitdiff(old(e, w, s, x), new(e, w, s, x))
        print(f"  {tree} compiled with the shared add and residual: max bit diff {n}")
        ok &= n == 0
    verdict = "PASS" if ok else "FAIL"
    print(
        f"{verdict} orders equal to cuBLAS at {t} x {k} x {d}: {found or 'none'}",
        flush=True,
    )
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
    """Router top-ks (modded_moe_topk.route) against torch.topk (CUDA) or aten_topk (CPU, under
    TRITON_INTERPRET=1): routing indices of s + bias for k, values and indices of s for k + 1."""
    import modded_moe_topk

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, e, k = (a.tokens, 96, a.topk) if dev == "cuda" else (256, 96, a.topk)
    gen = torch.Generator(dev).manual_seed(4)
    ref = (lambda x, k: tuple(torch.topk(x, k, dim=-1))) if dev == "cuda" else aten_topk

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=gen)

    live = torch.sigmoid(8 * rand(t, e) - 4), 0.02 * rand(e) - 0.01
    grid = (torch.randint(0, 6, (t, e), device=dev, generator=gen) / 8).float()
    ties = grid, torch.randint(-2, 3, (e,), device=dev, generator=gen) / 8
    odd = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 0.5, 2**-149]
    )
    pick = torch.randint(0, len(odd), (t, e), device=dev, generator=gen)
    special = (
        torch.where(rand(t, e) < 0.3, odd.to(dev)[pick], grid),
        torch.zeros(e, device=dev),
    )
    ok = True
    for name, (s, bias) in {"live": live, "ties": ties, "special": special}.items():
        s, bias = s.contiguous(), bias.float()
        idx, val, top = modded_moe_topk.route(s, bias, k, True)
        want = ref(s + bias, k)[1], *ref(s, k + 1)
        got = (idx, val, top)
        diff = [bitdiff(w.cpu(), g.cpu()) for w, g in zip(want, got)]
        zeros = modded_moe_topk.route(s, bias, k, False)
        diff.append(bitdiff(want[0].cpu(), zeros[0].cpu()))
        diff.append(max(z.abs().max().item() for z in zeros[1:]))
        print(
            f"  {name}: routing idx, stats values, stats idx, no-stats idx, no-stats zeros: {diff}"
        )
        ok &= not any(diff)
    if dev == "cuda":
        s, bias = live
        bench = triton.testing.do_bench
        routing = bench(lambda: torch.topk(s + bias, k))
        both = bench(lambda: (torch.topk(s + bias, k), torch.topk(s, k + 1)))
        print(f"  torch.topk ms: routing {routing:.3f}, both {both:.3f}")
        for rows, warps in ((8, 4), (16, 4), (16, 8), (32, 8)):
            run = functools.partial(modded_moe_topk.route, s, bias, k)
            ms = [
                bench(functools.partial(run, st, rows, warps)) for st in (False, True)
            ]
            print(
                f"  route ms, {rows} rows {warps} warps: routing {ms[0]:.3f}, both {ms[1]:.3f}"
            )
    verdict = "PASS" if ok else "FAIL"
    print(
        f"{verdict} router top-k at {t} x {e}, k {k}: bitwise torch.topk incl. ties",
        flush=True,
    )
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("layer", "orders", "topk", "run"))
    p.add_argument("--impl")
    p.add_argument("--base", default=BASE, help="layer: the reference commit")
    p.add_argument("--dump")
    p.add_argument("--kernel")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--tokens", type=int, default=65536)
    p.add_argument("--topk", type=int, default=4)
    a = p.parse_args()
    if a.cmd == "run":
        return run(a)
    raise SystemExit(
        0 if {"layer": layer, "orders": orders, "topk": topk}[a.cmd](a) else 1
    )


if __name__ == "__main__":
    main()
