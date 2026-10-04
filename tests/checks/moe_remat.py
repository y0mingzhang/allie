"""--moe-remat --moe-chunks 4 (the routed down output rerun in checkpointed blocks, the [T*k, D] products in 4 token
chunks) trains bitwise as the default on CPU, gloo world 2: the GPT (board CNN, dense attention, 256 experts top-4,
shared expert, quantile balancing, router centring, adam_every, gate floor, FP32 small masters), blocks 0-3 under eager
checkpoints and 4-7 not, one micro-batch on even steps and two on odd, real validation rows; every step's losses and
parameters equal. Collectives are deterministic (FP32 sums in rank order); model.moe_kernels's Triton kernels and polar
express run as torch references (on GPU the gates' grad, a batched matmul per chunk, may round otherwise).

    TORCH_COMPILE_DISABLE=1 python tests/checks/moe_remat.py
"""

import functools
import os
import socket

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from allie import paths
from allie.model import moe_kernels
from allie.data.packed import Packed
from allie.model.network import (
    Config,
    TrainingManager,
    core,
    cpu_copy,
    create_model,
    make_context,
    move_losses,
)
from allie.train.schedule import Schedule

DATA = str(paths.DATA / "lichess_tokens_v2")
ARCH = dict(
    moe=[256, 4], moe_shared_frac=0.25, moe_update="quantile", moe_router_center=0.9,
    moe_router_lr_mul=0.05, adam_every=True, moe_gate_floor=1e-12, fp32_small_masters=True,
)  # fmt: skip
STEPS = 6
SCHEDULE = Schedule(warmup_steps=2, mtp_steps=0, split_step=3, batch_rows=8)


class Done:
    def wait(self):
        pass


def patch():
    gather, world, rank = dist.all_gather, dist.get_world_size(), dist.get_rank()

    def every(t):
        b = t.detach().contiguous().reshape(-1).view(torch.uint8)
        parts = [torch.empty_like(b) for _ in range(world)]
        gather(parts, b)
        return [p.view(t.dtype).view(t.shape) for p in parts]

    def total(t, op):
        s = functools.reduce(torch.add, [x.double() for x in every(t)])
        return s / world if op == dist.ReduceOp.AVG else s

    def done(async_op):
        return Done() if async_op else None

    def all_reduce(t, op=dist.ReduceOp.SUM, group=None, async_op=False):
        t.copy_(total(t, op))
        return done(async_op)

    def reduce(t, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        s = total(t, op)
        if rank == dst:
            t.copy_(s)
        return done(async_op)

    def reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM, group=None, async_op=False):
        out.copy_(total(x, op).chunk(world)[rank])
        return done(async_op)

    def all_gather_into_tensor(out, x, group=None, async_op=False):
        out.copy_(torch.cat(every(x)))
        return done(async_op)

    def broadcast(t, src, group=None, async_op=False):
        t.copy_(every(t)[src])
        return done(async_op)

    dist.all_reduce, dist.reduce, dist.broadcast = all_reduce, reduce, broadcast
    dist.reduce_scatter_tensor = reduce_scatter_tensor
    dist.all_gather_into_tensor = all_gather_into_tensor

    def experts(offsets):
        return torch.repeat_interleave(
            torch.arange(len(offsets)), offsets.diff(prepend=offsets.new_zeros(1))
        )

    def mm(a, w):
        return torch.bmm(a.float()[:, None], w.float())[:, 0]

    def gated(dy, gates, order):
        g = gates.flatten()[order].float()[:, None]
        return (dy[order // gates.shape[1]].float() * g).to(dy.dtype)

    def up(x, w, order, offsets, k, save=True):
        pre = mm(x[order // k], w[experts(offsets)]).to(x.dtype)
        a, b = pre.float().chunk(2, -1)
        return pre if save else pre[:0], (F.silu(a) * b).to(x.dtype)

    def dx(dy, w, gates, order, offsets, pre):
        acc = mm(gated(dy, gates, order), w[experts(offsets)]).to(dy.dtype).float()
        a, b = pre.float().chunk(2, -1)
        s = torch.sigmoid(a)
        return torch.cat((acc * b * s * (a * (1 - s) + 1), acc * a * s), -1).to(
            pre.dtype
        )

    def scatter(x, w, se, order, h):
        out = x.new_empty(order.numel(), w.shape[-1])
        out[order] = mm(x, w[se]).to(x.dtype)
        return out

    def down_wgrad(dy, y, gates, order, offsets):
        g = gated(dy, gates, order).float()
        out = torch.zeros(len(offsets), y.shape[1], dy.shape[1])
        out.index_add_(0, experts(offsets), y.float()[:, :, None] * g[:, None])
        return out.to(dy.dtype)

    def up_wgrad(dpre, x, order, offsets, k):
        out = torch.zeros(len(offsets), dpre.shape[1], x.shape[1])
        rows = dpre.float()[:, :, None] * x[order // k].float()[:, None]
        return out.index_add_(0, experts(offsets), rows).to(dpre.dtype).transpose(1, 2)

    for f in (up, dx, scatter, down_wgrad, up_wgrad):
        setattr(moe_kernels, f.__name__, f)

    def polar_express(G, split_baddbmm=False):
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
        for a, b, c in core.polar_express_coeffs:
            A = X @ X.mT
            X = a * X + (b * A + c * A @ A) @ X
        return X.mT if G.size(-2) > G.size(-1) else X

    core.polar_express = polar_express


def build():
    torch.manual_seed(0)
    cfg = Config(
        width=64, head_dim=16, layers=8, max_tokens=1024, scheduled_steps=STEPS,
        ckpt="eager", ckpt_frac=0.5, arch=ARCH,
    )  # fmt: skip
    model = create_model(cfg, device="cpu")
    return model, TrainingManager(model, cfg, SCHEDULE)


def train(val, model, manager):
    """Every step's losses (this rank's micro-batches) and the parameters after it."""
    world, rank, out = dist.get_world_size(), dist.get_rank(), []
    for step in range(STEPS):
        manager.advance_schedule(step)
        losses = []
        accum = 1 + step % 2
        for micro in range(accum):
            if micro == accum - 1:
                manager.activate_hooks(step)
            rows = torch.as_tensor(val.rows(np.array([100 * step + 10 * micro + rank])))
            x, y = rows[:, :-1], rows[:, 1:]
            ctx = make_context(x, 1408, 2944, backend="dense")
            logits = model(x.flatten(), ctx, manager.get_forward_args())
            loss, primary, count = move_losses(logits, x, y, ctx, manager.mtp_weights)
            (loss * (world / 8)).backward()
            losses.append((primary / count).item())
        manager.step_optimizers(step)
        core.sync_params()
        out.append((losses, cpu_copy(model.state_dict())))
    return out


def same(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def worker(rank, world, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    patch()
    val = Packed(DATA, "val")
    ref = train(val, *build())
    moe_kernels.SAVE_EXPANDED, moe_kernels.CHUNKS = False, 4
    got = train(val, *build())
    ok = torch.tensor(float(same(ref, got)))
    dist.all_reduce(ok)  # the patched (summing) collective
    assert ok.item() == world, "--moe-remat --moe-chunks 4 must train bitwise as the default"
    if rank == 0:
        print(f"world {world}: --moe-remat --moe-chunks 4 bitwise over {STEPS} steps", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(2, port), nprocs=2)
    print("PASS moe_remat")
