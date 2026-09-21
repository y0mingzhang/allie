"""Expert weight grads without zero-fills or AccumulateGrad copies, bitwise against a base commit.

Runs this directory's MoE and the base commit's (git archive) in separate processes on identical
seeds, for every ScatterMoE kernel, eager and compiled (fullgraph, eager checkpoint), under
deterministic mode: random routes, half the experts without rows, and top-1 with every row on one
expert. CUDA adds the reference layer (E96 top-4, d1536, h512 swiglu, 64K tokens) and the tuned-tile
shape (E96 top-6, d2048, h455, 16K tokens). Compares the output and every grad bit for bit and
counts AccumulateGrad copies of expert grads. usage: test_expert_fills.py [--base 8123133]
[--device cuda|cpu] (cpu: Triton interpreter, small shapes only)."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = ("scatter", "scatter-tuned", "scatter-gather", "scatter-dualgather")
# E, k, d, h, shared, tokens
SMALL = {"cpu": (8, 2, 64, 24, 48, 256), "cuda": (16, 4, 256, 40, 176, 2048)}
CASES = [
    (k, s, SMALL, c)
    for k in KERNELS
    for s in ("random", "skew", "one")
    for c in (False, True)
]
REF, TILES = (96, 4, 1536, 512, 2048, 65536), (96, 6, 2048, 455, 2728, 16384)
CUDA_CASES = [
    ("scatter-dualgather", s, REF, c) for s in ("random", "skew") for c in (False, True)
]
CUDA_CASES += [(k, "random", TILES, True) for k in ("scatter-tuned", "scatter-gather")]


def run(kernel, scenario, shape, compiled, dev):
    import modded_moe
    import torch
    import torch.utils.checkpoint

    e, k, d, h, shared, t = shape
    k = 1 if scenario == "one" else k
    torch.manual_seed(701)
    m = (
        modded_moe.MoE(
            d, e, k, h, shared, kind="swiglu", kernel=kernel, init=0.006, seq=0
        )
        .to(dev)
        .train()
    )
    with torch.no_grad():
        m.down.normal_(0, 0.02)
        m.shared_down.normal_(0, 0.02)
        if scenario == "skew":
            m.bias[: e // 2] = -1e4  # half the experts get no rows
        if scenario == "one":
            m.bias[3] = 1e4

    def take(p):
        p.g, p.grad = p.grad, None

    for n, p in m.named_parameters():
        if n != "router":
            p.data = p.data.bfloat16()
        p.register_post_accumulate_grad_hook(take)
    g = torch.Generator().manual_seed(340)
    x = torch.randn(t, d, generator=g).bfloat16().to(dev).requires_grad_()
    r = torch.randn(t, d, generator=g).to(dev)
    f = torch.compile(m, fullgraph=True, dynamic=False) if compiled else m

    def step():
        y = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
        (y.float() * r).sum().backward()
        return y

    # cached free blocks hold NaN: an unwritten element cannot pass as a zero
    if dev == "cuda":
        torch.full((1 << 28,), float("nan"), device=dev)
    y = step()
    out = {"y": y.detach(), "dx": x.grad} | {n: p.g for n, p in m.named_parameters()}
    x.grad = None
    experts = {tuple(m.up.shape), tuple(m.down.shape)}
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as prof:
        step()
    copies = sum(
        ev.name == "aten::copy_"
        and "AccumulateGrad" in getattr(ev.cpu_parent, "name", "")
        and tuple(ev.input_shapes[0]) in experts
        for ev in prof.events()
    )
    out = {n: v.detach().cpu().clone() for n, v in out.items()}
    out["accumulate_copies"] = torch.tensor(copies)
    return out


def worker(src, dev, path):
    sys.path.insert(0, src)
    import modded_moe
    import torch
    import torch._dynamo

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 256
    modded_moe.BLOCK_RECOMPUTE = True
    res = {}
    for kernel, scenario, shape, compiled in CASES + (
        CUDA_CASES if dev == "cuda" else []
    ):
        shape = shape[dev] if isinstance(shape, dict) else shape
        name = f"{kernel}/{scenario}/{shape[-1]}/{'compiled' if compiled else 'eager'}"
        res |= {
            f"{name}/{n}": v
            for n, v in run(kernel, scenario, shape, compiled, dev).items()
        }
        torch._dynamo.reset()
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(
            name,
            "expert AccumulateGrad copies",
            int(res[f"{name}/accumulate_copies"]),
            flush=True,
        )
    torch.save(res, path)


def bits(a, b):
    import torch

    i = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return int((a.view(i).long() - b.view(i).long()).abs().max()) if a.numel() else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="8123133")
    p.add_argument("--device", default="cuda")
    p.add_argument("--worker", nargs=2, metavar=("SRC", "OUT"))
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if a.device == "cpu":
        os.environ["TRITON_INTERPRET"] = "1"
    if a.worker:
        return worker(a.worker[0], a.device, a.worker[1])
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        tar = subprocess.run(
            ["git", "-C", str(HERE.parent), "archive", a.base, "scripts"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["tar", "-x", "-C", tmp], input=tar.stdout, check=True)
        res = {}
        for tag, src in (("base", f"{tmp}/scripts"), ("new", str(HERE))):
            print(f"== {tag} {src}", flush=True)
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=f"{tmp}/inductor-{tag}")
            cmd = [
                sys.executable,
                __file__,
                "--device",
                a.device,
                "--worker",
                src,
                f"{tmp}/{tag}.pt",
            ]
            subprocess.run(cmd, check=True, env=env)
            res[tag] = torch.load(f"{tmp}/{tag}.pt")
    base, new = res["base"], res["new"]
    bad, worst, compared = [], 0, 0
    for key, b in base.items():
        if key.endswith("accumulate_copies"):
            continue
        n, compared = new[key], compared + 1
        diff = bits(n, b) if n.shape == b.shape and n.dtype == b.dtype else -1
        _, scenario, *_, name = key.split("/")
        if (
            scenario == "skew"
            and name in ("up", "down")
            and n[: n.shape[0] // 2].view(torch.int16).any()
        ):
            diff = -1  # experts without rows must hold +0.0
        worst = max(worst, diff)
        if diff:
            bad.append((key, diff))
    copies = {
        tag: sum(int(v) for k, v in r.items() if k.endswith("accumulate_copies"))
        for tag, r in res.items()
    }
    print(
        f"compared {compared} tensors; expert AccumulateGrad copies per profiled step, all cases: {copies}"
    )
    for key, diff in bad:
        print("MISMATCH", key, diff)
    print(
        f"{'PASS' if not bad and copies['new'] == 0 else 'FAIL'} max bit diff {worst}"
    )


if __name__ == "__main__":
    main()
