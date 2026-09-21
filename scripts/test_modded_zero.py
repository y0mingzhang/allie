"""CPU tests of the optimizers' BF16 weights (FP32 master shards, ZeRO-2 grad reduction to the
NorMuon owner, deferred owner broadcasts) on gloo worlds 1/2/4, MoE experts included. NCCL-only
collectives (AVG reduce, reduce-scatter, all-gather-into-tensor) and the Triton Newton-Schulz
kernels are replaced by torch equivalents; the model is never run forward (BF16-representable fake
grads per rank).

Checked after every step, on every rank: each owned master's BF16 rounding is its param, every
rank holds the same params; and a run saved at step 4 (model state and rank_state_dict), rebuilt
and restored continues bit-identical to the uninterrupted run through the embed split.

    .venv/bin/python scripts/test_modded_zero.py
"""

import os
import socket
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modded_medium as mm
from modded_medium import Config, TrainingManager, create_model, core
from modded_wsd import Schedule

ARCH = dict(moe=[8, 2], moe_seq=1e-3)
# 10 layers: the attn group pads at world 4; split mid-run
LAYERS, STEPS, SPLIT = 10, 8, 5
SCHEDULE = Schedule(warmup_steps=2, mtp_steps=0, split_step=SPLIT, batch_rows=8)


class Done:
    def wait(self):
        pass


def patch():
    reduce = dist.all_reduce

    def all_reduce(t, op=dist.ReduceOp.SUM, group=None, async_op=False):
        buf = t.float().clone()
        reduce(buf)
        t.copy_(buf / dist.get_world_size() if op == dist.ReduceOp.AVG else buf)
        return Done() if async_op else None

    def reduce_scatter_tensor(
        out, inp, op=dist.ReduceOp.SUM, group=None, async_op=False
    ):
        buf = inp.float().clone()
        all_reduce(buf, op)
        out.copy_(buf.chunk(dist.get_world_size())[dist.get_rank()])
        return Done() if async_op else None

    def reduce_to(t, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        buf = t.float().clone()
        all_reduce(buf, op)
        if dist.get_rank() == dst:
            t.copy_(buf)
        return Done() if async_op else None

    def all_gather_into_tensor(out, inp, group=None, async_op=False):
        bits = inp.contiguous().view(torch.uint8)  # bit-exact transport, any dtype
        parts = [torch.empty_like(bits) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, bits)
        whole = out.view(torch.uint8)
        whole.copy_(torch.cat(parts).view(whole.shape))
        return Done() if async_op else None

    def polar_express(G, split_baddbmm=False):
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
        for a, b, c in core.polar_express_coeffs:
            A = X @ X.mT
            X = a * X + (b * A + c * A @ A) @ X
        return X.mT if G.size(-2) > G.size(-1) else X

    bcast = dist.broadcast
    dist.all_reduce, dist.reduce_scatter_tensor = all_reduce, reduce_scatter_tensor
    dist.all_gather_into_tensor, dist.reduce = all_gather_into_tensor, reduce_to
    dist.broadcast = lambda t, src, group=None, async_op=False: (
        bcast(t.view(torch.uint8), src) or Done()
    )
    core.polar_express = polar_express


def build():
    torch.manual_seed(0)
    cfg = Config(
        width=64, head_dim=16, layers=LAYERS, max_tokens=1024, scheduled_steps=STEPS,
        ckpt="eager", arch=ARCH,
    )  # fmt: skip
    model = create_model(cfg, device="cpu")
    return model, TrainingManager(model, cfg, SCHEDULE)


def train(model, manager, steps, accum=2):
    """Per-rank BF16-representable fake gradients, identical for every model."""
    for step in steps:
        manager.advance_schedule(step)
        for micro in range(accum):
            if micro == accum - 1:
                manager.activate_hooks(step)
            g = torch.Generator().manual_seed(
                1000 * step + 10 * micro + dist.get_rank()
            )
            loss = 0
            for p in model.parameters():
                x = (1e-2 * torch.randn(p.shape, generator=g)).bfloat16()
                loss = loss + (p * x.to(p.dtype)).sum()
            loss.backward()
        manager.step_optimizers(step)
        core.sync_params()
        check(manager)


def check(manager):
    rank = dist.get_rank()
    for g in manager.muon_opt.param_groups:
        lo = rank * g["chunk_size"]
        for i, m in enumerate(g.get("master", ())):
            assert (
                m.dtype == torch.float32 and g["params"][lo + i].dtype == torch.bfloat16
            )
            assert torch.equal(m.bfloat16(), g["params"][lo + i]), g["params"][0].label
    for p in manager.model.parameters():
        every = [torch.empty_like(p) for _ in range(dist.get_world_size())]
        dist.all_gather(every, p.detach().contiguous())
        assert all(torch.equal(e, p) for e in every), f"ranks disagree on {p.label}"


def test_resume():
    full, full_mgr = build()
    assert {n for n, p in full.named_parameters() if getattr(p, "master", False)} == {
        n for n, p in full.named_parameters() if p.label in mm.MASTER_LABELS
    }
    assert not any(hasattr(p, "fp32") for p in full.parameters())  # init stash gone
    train(full, full_mgr, range(STEPS))
    init = build()[0]
    moved = {
        p.label
        for p, q in zip(full.parameters(), init.parameters())
        if not torch.equal(p, q)
    }
    assert set(mm.MASTER_LABELS) <= moved, moved
    part, part_mgr = build()
    train(part, part_mgr, range(4))
    shared, local = mm.cpu_copy(part.state_dict()), part_mgr.rank_state_dict()
    model, manager = build()
    model.load_state_dict(shared)
    manager.load_rank_state_dict(local)
    train(model, manager, range(4, STEPS))
    assert model.split_embed
    for p, q in zip(model.parameters(), full.parameters()):
        assert p.dtype == q.dtype and torch.equal(p, q), p.label
    for g, h in zip(manager.muon_opt.param_groups, full_mgr.muon_opt.param_groups):
        for k in ("master", "momentum_buffer", "second_momentum_buffer"):
            if k in g:
                assert g[k].dtype == h[k].dtype and torch.equal(g[k], h[k]), k


def worker(rank, world, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    patch()
    test_resume()
    if rank == 0:
        print(f"world {world}: ok ({STEPS} steps, resume at 4, split at {SPLIT})")
    dist.destroy_process_group()


if __name__ == "__main__":
    for world in (1, 2, 4):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mp.spawn(worker, args=(world, port), nprocs=world)
