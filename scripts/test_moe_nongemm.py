"""Gate of the MoE layer's non-GEMM speedups against perf-int (a522ee6), bit for bit.

layer: one MoE layer as --ckpt eager runs it (x + moe(norm(x)) compiled whole by _eager_block,
checkpointed, deterministic), 2 steps x 3 micro-batches plus a no-grad training forward, the
bias nonzero and one step per update rule. Each implementation runs in its own process; outputs,
input and parameter gradients, the load buffer after every micro-batch, the rebalanced bias and
the logged stats must match perf-int's bitwise. --set MOD.NAME=VALUE switches (new side only).
CUDA: the candidate layer (d1536 SwiGLU, E96 top-4, 64K tokens, BF16 weights), fwd+bwd ms and
the kernels whose time changed. CPU: 2048 tokens of a d64 E16 top-4 layer (pad kernel).

    inhold.sh scripts/test_moe_nongemm.py layer [--eager] [--no-ckpt] [--set ...]
"""

import argparse
import ast
import importlib
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

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


def base_scripts():
    out = tempfile.mkdtemp()
    git = ["git", "-C", str(HERE.parent), "archive", f"{BASE}:scripts"]
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
        ("base", base_scripts(), []),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("layer", "run"))
    p.add_argument("--impl")
    p.add_argument("--dump")
    p.add_argument("--kernel")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    if a.cmd == "run":
        return run(a)
    raise SystemExit(0 if layer(a) else 1)


if __name__ == "__main__":
    main()
