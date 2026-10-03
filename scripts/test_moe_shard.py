"""arch moe_shard (modded_shard) on CPU, gloo worlds 2 and 4: the GPT (board CNN, dense attention, v2's MoE recipe:
256 experts top-4, shared expert, quantile balancing, router centring, adam_every, gate floor, FP32 small masters),
blocks 0-3 under eager checkpoints and 4-7 not, two micro-batches a step, real validation rows. Collectives are
deterministic (FP32 sums in rank order); modded_smoe's Triton kernels and polar express run as torch references.

replicated vs sharded: every step's losses and every parameter (the sharded experts gathered whole) are equal, or
the step's largest difference is printed. resume: a sharded run saved after 3 steps as modded_train saves it
(model.pt with whole experts, each rank's state), rebuilt, loaded and continued through the embed split equals the
uninterrupted run bit for bit: losses, model.pt, every rank's optimizer state. --moe-remat (modded_smoe.SAVE_EXPANDED
off) is bitwise the sharded run. eval: an unsharded model loading the final model.pt gives the sharded model's eval
logits.

    TORCH_COMPILE_DISABLE=1 python scripts/test_moe_shard.py
"""

import functools
import os
import socket
import sys
import tempfile

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modded_shard
import modded_smoe
from lm_data import Packed
from modded_medium import (
    Config,
    TrainingManager,
    core,
    cpu_copy,
    create_model,
    make_context,
    move_losses,
)
from modded_wsd import Schedule

DATA = "/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2"
ARCH = dict(
    moe=[256, 4], moe_shared_frac=0.25, moe_update="quantile", moe_router_center=0.9,
    moe_router_lr_mul=0.05, adam_every=True, moe_gate_floor=1e-12, fp32_small_masters=True,
)  # fmt: skip
STEPS, SAVE = 6, 3
SCHEDULE = Schedule(warmup_steps=2, mtp_steps=0, split_step=SAVE, batch_rows=8)


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
        setattr(modded_smoe, f.__name__, f)

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


def build(shard):
    torch.manual_seed(0)
    cfg = Config(
        width=64, head_dim=16, layers=8, max_tokens=1024, scheduled_steps=STEPS,
        ckpt="eager", ckpt_frac=0.5, arch=ARCH | dict(moe_shard=shard),
    )  # fmt: skip
    model = create_model(cfg, device="cpu")
    assert all(
        (m.pos is not None) == shard for m in model.modules() if isinstance(m, core.MoE)
    )
    return model, TrainingManager(model, cfg, SCHEDULE)


def train(val, model, manager, steps, accum=2):
    """Every step's losses (this rank's micro-batches) and the whole parameters after it (rank 0)."""
    world, rank, out = dist.get_world_size(), dist.get_rank(), []
    for step in steps:
        manager.advance_schedule(step)
        losses = []
        for micro in range(accum):
            if micro == accum - 1:
                manager.activate_hooks(step)
            rows = torch.as_tensor(val.rows(np.array([100 * step + 10 * micro + rank])))
            x, y = rows[:, :-1], rows[:, 1:]
            ctx = make_context(x, 1408, 2944, backend="dense")
            logits = model(x.flatten(), y.flatten(), ctx, manager.get_forward_args())
            loss, primary, count = move_losses(logits, x, y, ctx, manager.mtp_weights)
            (loss * (world / 8)).backward()
            losses.append((primary / count).item())
        manager.step_optimizers(step)
        core.sync_params()
        out.append((losses, modded_shard.state_dict(model, cpu_copy)))
    return out


def same(a, b):
    """Largest absolute difference of two nested states (0.0: bitwise equal)."""
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        return (
            0.0 if torch.equal(a, b) else (a.double() - b.double()).abs().max().item()
        )
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([same(a[k], b[k]) for k in a], default=0.0)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max([same(x, y) for x, y in zip(a, b)], default=0.0)
    assert a == b, (a, b)
    return 0.0


def check_masters(manager):
    """Every owned FP32 master rounds to its BF16 param."""
    for opt in manager.optimizers[2:]:
        for g in opt.param_groups:
            lo = 0 if opt.local else dist.get_rank() * g["chunk_size"]
            for i, m in enumerate(g.get("master", ())):
                assert torch.equal(m.bfloat16(), g["params"][lo + i]), g["params"][
                    0
                ].label


def worker(rank, world, port, tmp):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    patch()
    val = Packed(DATA, "val")
    say = lambda *a: rank == 0 and print(f"world {world}:", *a, flush=True)

    ref = train(val, *build(False), range(STEPS))
    model, manager = build(True)
    got = train(val, model, manager, range(STEPS))
    check_masters(manager)
    for step, ((la, wa), (lb, wb)) in enumerate(zip(ref, got)):
        diff = torch.tensor(
            [abs(x - y) for x, y in zip(la, lb)] + [same(wa, wb) if rank == 0 else 0.0]
        )
        dist.all_reduce(diff)
        say(
            f"step {step + 1}: replicated vs sharded, loss diff {diff[:-1].max():.3g}, param diff {diff[-1]:.3g}"
        )
        assert step or not diff.any(), "step 1 must be bitwise"
    final = got[-1][1], manager.rank_state_dict()
    del model, manager
    modded_smoe.SAVE_EXPANDED = False
    assert same(train(val, *build(True), range(STEPS)), got) == 0.0
    modded_smoe.SAVE_EXPANDED = True
    say("--moe-remat (expanded rerun in checkpointed blocks) bitwise")

    model, manager = build(True)
    head = train(val, model, manager, range(SAVE))
    assert same(head, got[:SAVE]) == 0.0
    whole = modded_shard.state_dict(model, cpu_copy)
    if rank == 0:
        torch.save(whole, f"{tmp}/model.pt")
    torch.save(manager.rank_state_dict(), f"{tmp}/rank{rank}.pt")
    dist.barrier()
    del model, manager
    model, manager = build(True)
    model.load_state_dict(torch.load(f"{tmp}/model.pt", mmap=True))
    manager.load_rank_state_dict(torch.load(f"{tmp}/rank{rank}.pt", weights_only=False))
    tail = train(val, model, manager, range(SAVE, STEPS))
    assert model.split_embed
    assert same([x[0] for x in tail], [x[0] for x in got[SAVE:]]) == 0.0
    if rank == 0:
        assert same(tail[-1][1], final[0]) == 0.0
    assert same(manager.rank_state_dict(), final[1]) == 0.0
    say(f"resume at {SAVE} -> {STEPS} bitwise: losses, model.pt, rank state")

    if rank == 0:
        torch.save(tail[-1][1], f"{tmp}/model.pt")
    dist.barrier()
    plain = build(False)[0]
    plain.load_state_dict(torch.load(f"{tmp}/model.pt"))
    plain.split_embed = model.split_embed  # model.pt's inference state
    rows = torch.as_tensor(val.rows(np.array([7])))
    ctx = make_context(rows[:, :-1], 1408, 2944, backend="dense")
    args = (
        rows[:, :-1].flatten(),
        rows[:, 1:].flatten(),
        ctx,
        manager.get_forward_args(),
    )
    with torch.inference_mode():
        a, b = model.eval()(*args), plain.eval()(*args)
    assert torch.equal(a, b)
    say("eval: unsharded model on the sharded run's model.pt gives its logits")
    dist.destroy_process_group()


if __name__ == "__main__":
    for world in (2, 4):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(world, port, tmp), nprocs=world)
    print("PASS moe_shard")
