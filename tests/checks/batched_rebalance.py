"""model.moe.rebalance_all vs per-layer rebalance (CPU, gloo world 1 and 2): quantile updates on random loads and
histograms, the first layer centring: biases, mu and the integer-valued stats equal bitwise, the logged margin within
float rounding.

    python tests/checks/batched_rebalance.py
"""

import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from allie.model import moe as moe_layer
from allie.model import moe_kernels
from allie.model.moe import MoE

E, K, D = 64, 4, 32


def layers(seed, n=5):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        m = MoE(
            D,
            E,
            K,
            8,
            8,
            center=0.9 if i == 0 else 0.0,
            log_gates=True,
        )
        m.load.copy_(
            torch.cat(
                (
                    torch.randint(0, 500, (E,), generator=g).float(),
                    torch.tensor([0.0, 7, 4096, 3.3]),
                )
            )
        )
        m.hist.copy_(
            torch.randint(
                0, 40, (E, moe_kernels.QB_BINS), generator=g, dtype=torch.int32
            )
        )
        m.bias.copy_(torch.randn(E, generator=g) * 0.1)
        if m.center:
            m.mu_sum.copy_(torch.randn(D + 1, generator=g).abs())
            m.mu_steps.fill_(float(i))
        out.append(m)
    return out


def run(rank, world):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", "29613"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    a = layers(100 + rank)
    b = copy.deepcopy(a)
    for m in a:
        m.rebalance()
    moe_layer.rebalance_all(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x.bias, y.bias), f"layer {i} bias"
        assert torch.equal(x.load, y.load) and torch.equal(x.hist, y.hist), (
            f"layer {i} zeroed"
        )
        if x.center:
            assert torch.equal(x.mu, y.mu) and torch.equal(x.mu_steps, y.mu_steps), (
                f"layer {i} mu"
            )
        same = torch.equal(x.stats[:-1], y.stats[:-1])
        assert same, f"layer {i} stats {x.stats} vs {y.stats}"
        assert torch.allclose(x.stats[-1], y.stats[-1], rtol=1e-6, atol=0), (
            f"layer {i} margin"
        )
    if rank == 0:
        print(
            f"world {world}: {len(a)} layers equal (bias, mu, stats; margin within float rounding)"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    for world in (1, 2):
        mp.spawn(run, args=(world,), nprocs=world, join=True)
    print("PASS batched_rebalance")
