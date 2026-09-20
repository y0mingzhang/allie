"""modded_moe.counts vs the scatter_add_ histogram it replaced, under deterministic algorithms (CUDA if
present, else CPU): equal values, dtype, device and shape over random and adversarial routings, eager
and compiled; then a pad-kernel MoE layer whose output, gradients and load buffer must match bitwise."""

import itertools
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import modded_moe
import torch
from modded_moe import MoE, counts

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def old(flat, e):
    return flat.new_zeros(e).scatter_add_(0, flat, torch.ones_like(flat))


def new(flat, e):
    return counts(flat[flat.argsort(stable=True)], e)


def routings(t, e, k, g):
    s = torch.rand(t, e, generator=g)
    few = torch.randperm(e, generator=g) < 2 * k
    yield "uniform", s.topk(k, -1).indices
    yield "skew", (s + torch.linspace(8, 0, e)).topk(k, -1).indices
    yield "subset", (s + 2 * few).topk(k, -1).indices  # e - 2k experts empty
    yield "first", torch.arange(k).expand(t, k)
    yield "last", torch.arange(e - k, e).expand(t, k)
    yield "one", torch.full((t, k), e - 1)  # every route to one expert


def check_counts(mode):
    g = torch.Generator().manual_seed(0)
    for e, k, t in itertools.product((64, 96, 128, 256), (4, 6, 8), (1024, 65536)):
        torch._dynamo.reset()
        both = torch.compile(
            lambda f, e=e: (old(f, e), new(f, e)), fullgraph=True, dynamic=False
        )
        for name, idx in routings(t, e, k, g):
            flat, key = idx.flatten().to(DEV), (e, k, t, name)
            want = old(flat, e)
            for c in both(flat) if mode == "compiled" else (new(flat, e),):
                assert (c.dtype, c.device) == (want.dtype, want.device), key
                assert c.shape == want.shape and torch.equal(c, want), key
            assert int(want.sum()) == t * k
        print(f"counts {mode} e={e} k={k} t={t} ok", flush=True)


def layer(mode, count):
    modded_moe.counts = count
    torch._dynamo.reset()
    torch.manual_seed(0)
    m = MoE(64, 96, 4, 32, 64, init=0.006, seq=0.001, kind="swiglu", kernel="pad")
    m = m.to(DEV).train()
    with torch.no_grad():
        m.down.normal_(0, 0.02)
        m.shared_down.normal_(0, 0.02)
        m.bias[:8] = 1.0  # overloaded experts: the training capacity drops routes
    for n, p in m.named_parameters():
        if n != "router":
            p.data = p.data.bfloat16()
    x = torch.randn(2048, 64, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    f = torch.compile(m, fullgraph=True, dynamic=False) if mode == "compiled" else m
    y = f(x)
    (y.float() * torch.randn(y.shape, device=DEV)).sum().backward()
    return [y, x.grad, m.load, *(p.grad for p in m.parameters())]


def check_layer(mode):
    a = layer(mode, old)
    b = layer(mode, counts)
    modded_moe.counts = counts
    assert a[2][-4] > 0, "no dropped routes"
    for u, v in zip(a, b, strict=True):
        assert u.dtype == v.dtype and torch.equal(u, v)
    print(f"pad layer {mode} bitwise ok", flush=True)


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    for mode in sys.argv[1:] or ("eager", "compiled"):
        check_counts(mode)
        check_layer(mode)
