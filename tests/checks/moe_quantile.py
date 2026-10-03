"""Quantile Balancing (model.moe update="quantile") on synthetic skewed sigmoid scores: the histogram
shift against the exact per-expert quantile, the realised top-k load after 1 and 3 updates, on the same
tokens and on fresh ones (the next step's batch), invariance to the doubled counts of replay_context, and
MoE.rebalance's wiring (gloo, world 1), and a two-expert case that cycles under per-expert cutoffs.
CPU or CUDA.

    .venv/bin/python tests/checks/moe_quantile.py
"""

import os
import sys

import torch
import torch.distributed as dist

from allie.model import moe as mm
from allie.model import moe_kernels


def scores(t, e, gen, device):
    shift = torch.zeros(e)
    shift[:4], shift[4:8] = 3.0, -3.0  # four hot experts, four cold
    return torch.sigmoid(2 * torch.randn(t, e, generator=gen) + shift).to(device)


def load(s, bias, k):
    idx = torch.topk(s + bias, k, dim=-1).indices
    return torch.bincount(idx.flatten(), minlength=s.shape[1]).double(), idx


def update(s, bias, k):
    _, idx = load(s, bias, k)
    b = bias + mm.qb_shift(mm.qb_counts(s, bias, idx), k)
    return (b - b.mean()).float()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    t, e, k = 1 << 16, 64, 4
    s, fresh = scores(t, e, gen, device), scores(t, e, gen, device)
    target = t * k / e
    bias = torch.zeros(e, device=device)
    ok = True

    _, idx = load(s, bias, k)
    alpha = torch.topk(s + bias, k + 1, dim=-1).values[:, k:]
    same = torch.equal(mm.qb_margins(s, bias, idx), alpha - (s + bias))
    print(f"margins = (k+1)-th biased score - biased score: {same}")
    ok &= same
    edges = mm.qb_edges(device).float()
    mid, n = edges[1:-2], moe_kernels.QB_LEVELS
    key = torch.arange(
        1, len(edges) - 2, device=device
    )  # x at an edge: [lo, hi) above 0, (lo, hi] below
    want = torch.where(mid < 0, key - 1, key)
    extra = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 32.0, -32.0], device=device)
    same = torch.equal(mm.qb_bin(mid), want) and mm.qb_bin(extra).tolist() == [
        n,
        n,
        n,
        n - 1,
        2 * n - 1,
        0,
    ]
    print(f"bin edges: {same}")
    ok &= same
    exact = mm.qb_margins(s, bias, idx).kthvalue(round(target), dim=0).values.double()
    got = mm.qb_shift(mm.qb_counts(s, bias, idx), k)
    width = exact.abs().clamp(min=2**-24) * 2**-moe_kernels.QB_MANT
    err = ((got - exact).abs() / width).max().item()
    print(f"shift vs exact quantile: max error {err:.3f} bins")
    ok &= err <= 1

    if device == "cuda" or os.environ.get("TRITON_INTERPRET"):
        for z, kz in (
            (s.clone(), k), (scores(4099, 96, gen, device), 6),
            (scores(t // 4, 2 * e, gen, device), 2 * k), (scores(t // 8, 4 * e, gen, device), 12),
        ):  # fmt: skip
            z[: len(z) // 8, : 3 * kz] = 1.0  # saturated ties
            bz = torch.randn(z.shape[1], generator=gen).to(device) * 0.1
            iz = load(z, bz, kz)[1]
            hist = torch.zeros(z.shape[1], moe_kernels.QB_BINS, device=device).int()
            moe_kernels.qb_hist(z, bz, iz, hist, 2)
            same = torch.equal(hist.long(), 2 * mm.qb_counts(z, bz, iz))
            print(f"qb_hist E{z.shape[1]} top-{kz} = 2 x qb_counts: {same}")
            ok &= same
    doubled = mm.qb_shift(2 * mm.qb_counts(s, bias, idx), k)
    print(f"doubled counts: shift equal {torch.equal(doubled, got)}")
    ok &= torch.equal(doubled, got)

    # excluding each expert from its own cutoff cycles [4, 0] <-> [0, 4] here (codex's repro)
    two, b2 = (
        torch.tensor([[0.6, 0.5], [0.7, 0.5], [0.8, 0.5], [0.9, 0.5]], device=device),
        bias[:2],
    )
    for _ in range(3):
        b2 = update(two, b2, 1)
    l2 = load(two, b2, 1)[0].tolist()
    print(f"E2 top-1 period-2 case after 3 updates: load {l2}")
    ok &= l2 == [2.0, 2.0]

    l0 = load(s, bias, k)[0] / target
    print(f"update 0: load / target {l0.min():.3f} .. {l0.max():.3f}")
    tol = {1: 0.5, 3: 0.1}  # max |load / target - 1| on the same tokens
    noise = 6 / target**0.5  # fresh tokens: + 6 sigma of binomial sampling
    for n in range(1, 6):
        bias = update(s, bias, k)
        if n == 4:
            continue
        dev = [(load(z, bias, k)[0] / target - 1).abs().tolist() for z in (s, fresh)]
        same, new = ((max(v), sum(v) / e) for v in dev)
        print(f"update {n}: |load / target - 1| max (mean) same tokens {same[0]:.4f} "
              f"({same[1]:.4f}), fresh {new[0]:.4f} ({new[1]:.4f})")  # fmt: skip
        if n in tol:
            ok &= max(dev[0]) <= tol[n] and max(dev[1]) <= tol[n] + noise

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    dist.init_process_group("gloo", rank=0, world_size=1)
    m = mm.MoE(16, e, k, 8, 8, update="quantile").to(device)
    m.bias.fill_(0.3)
    _, idx = load(s, m.bias, k)
    mm.qb_book(m.hist, s, m.bias, idx)
    want = m.bias + mm.qb_shift(mm.qb_counts(s, m.bias, idx), k)
    m.load[:e] = torch.bincount(idx.flatten(), minlength=e).float()
    m.rebalance(0.0)
    wired = torch.allclose(m.bias, (want - want.mean()).float()) and not m.hist.any()
    print(f"MoE.rebalance: bias = centred shift {wired}, stats {m.stats.tolist()}")
    ok &= wired
    dist.destroy_process_group()
    print("PASS" if ok else "FAIL")
    sys.exit(not ok)


if __name__ == "__main__":
    main()
