"""CPU tests of --bf16-weights (BF16 matrices + FP32 master shards) on gloo worlds 1/2/4. NCCL-only
collectives (AVG reduce-scatter, all-gather-into-tensor) and the Triton Newton-Schulz kernels are
replaced by torch equivalents.

The BF16-weight path must equal the FP32-weight path bit for bit (FP32 init masters, FP32 grad
accumulation and reduce-scatter); fake grads are BF16-representable so both see the same dW.

    TORCH_COMPILE_DISABLE=1 .venv/bin/python scripts/test_modded_zero.py
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
from modded_wsd import Schedule, install

LAYERS, STEPS, SPLIT = (
    10,
    8,
    5,
)  # 10 layers: the attn group pads at world 4; split mid-run


class Done:
    def get_future(self):
        f = torch.futures.Future()
        f.set_result(None)
        return f


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
        mean = buf.chunk(dist.get_world_size())[dist.get_rank()]
        out.copy_(mean)
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

    dist.all_reduce, dist.reduce_scatter_tensor = all_reduce, reduce_scatter_tensor
    dist.all_gather_into_tensor = all_gather_into_tensor
    core.polar_express = polar_express


def matrices(model):
    return [
        p for p in model.parameters() if getattr(p, "label", None) in mm.MASTER_LABELS
    ]


def build(bf16):
    torch.manual_seed(0)
    cfg = Config(
        width=64, head_dim=16, layers=LAYERS, max_tokens=1024, scheduled_steps=STEPS,
        extension_steps=0, initial_batch_rows=8, bf16_weights=bf16,
    )  # fmt: skip
    model = create_model(cfg, device="cpu")
    manager = TrainingManager(model, cfg)
    manager.split_step = SPLIT
    return model, manager


def train(model, manager, steps, accum):
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


def shards(manager):
    """(master, group params, first owned index) for every NorMuon group holding masters."""
    return [
        (g["master"], g["params"], dist.get_rank() * g["chunk_size"])
        for g in manager.muon_opt.param_groups
        if "master" in g
    ]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def test_reference(accum):
    """--bf16-weights is the FP32-weight optimizer bit for bit: masters equal the FP32 params and the
    BF16 params equal BF16(FP32 params) after every step, on every rank; nothing switched group."""
    ref, ref_mgr = build(False)
    bf, bf_mgr = build(True)
    ref_of = {id(p): q for p, q in zip(bf.parameters(), ref.parameters())}
    keys = (
        "lr",
        "initial_lr",
        "weight_decay",
        "momentum",
        "beta2",
        "betas",
        "eps",
        "chunk_size",
    )
    for o, r in zip(bf_mgr.optimizers, ref_mgr.optimizers):
        assert type(o) is type(r) and len(o.param_groups) == len(r.param_groups)
        for g, h in zip(o.param_groups, r.param_groups):
            assert [id(ref_of[id(p)]) for p in g["params"]] == [
                id(q) for q in h["params"]
            ]
            assert {k: g.get(k) for k in keys} == {k: h.get(k) for k in keys}
    assert {n for n, p in bf.named_parameters() if getattr(p, "master", False)} == {
        n for n, p in bf.named_parameters() if p.label in mm.MASTER_LABELS
    }
    assert not any(
        hasattr(p, "fp32") for p in bf.parameters()
    )  # init stash handed over
    for step in range(STEPS):
        train(ref, ref_mgr, [step], accum)
        train(bf, bf_mgr, [step], accum)
        found = shards(bf_mgr)
        assert len(found) == 2, found  # attn and mlp masters
        for master, params, lo in found:
            assert master.dtype == torch.float32 and params[0].dtype == torch.bfloat16
            for i, m in enumerate(master):
                r = ref_of[id(params[lo + i])].detach()
                assert torch.equal(m, r), f"step {step}: master != FP32 param"
        for p, q in zip(bf.parameters(), ref.parameters()):
            assert torch.equal(p, q.to(p.dtype)), f"step {step}: {p.label}"
        for g, h in zip(bf_mgr.muon_opt.param_groups, ref_mgr.muon_opt.param_groups):
            for k in ("momentum_buffer", "second_momentum_buffer"):
                assert g[k].dtype == h[k].dtype and torch.equal(g[k], h[k]), k
    for p in bf.parameters():
        every = [torch.empty_like(p) for _ in range(dist.get_world_size())]
        dist.all_gather(every, p.detach().contiguous())
        assert all(torch.equal(e, p) for e in every), f"ranks disagree on {p.label}"


def test_resume():
    """save at step 4, rebuild and load, continue: bit-identical to an uninterrupted run."""
    full, full_mgr = build(True)
    train(full, full_mgr, range(STEPS), 2)
    part, part_mgr = build(True)
    train(part, part_mgr, range(4), 2)
    shared, local = mm.cpu_copy(part.state_dict()), part_mgr.rank_state_dict()
    model, manager = build(True)
    model.load_state_dict(shared)
    manager.load_rank_state_dict(local)
    assert all(m.dtype == torch.float32 for m, *_ in shards(manager))
    train(model, manager, range(4, STEPS), 2)
    for p, q in zip(model.parameters(), full.parameters()):
        assert p.dtype == q.dtype and torch.equal(p, q), p.label
    for g, h in zip(manager.muon_opt.param_groups, full_mgr.muon_opt.param_groups):
        for k in ("master", "momentum_buffer", "second_momentum_buffer"):
            if k in g:
                assert g[k].dtype == h[k].dtype and torch.equal(g[k], h[k]), k


def test_distadam_master():
    """DistAdam with BF16 master params (sharded and small): BF16 slice == BF16(master), master near
    FP32 params fed the same grads, bit-identical resume from its state dict."""
    world, rank = dist.get_world_size(), dist.get_rank()
    torch.manual_seed(1)
    init = [torch.randn(s).bfloat16().float() for s in ((8 * world, 16), (4, 8))]

    def make(bf16):
        ps = [
            torch.nn.Parameter(x.clone().to(torch.bfloat16 if bf16 else x.dtype))
            for x in init
        ]
        for p in ps:
            p.label, p.master = "attn", bf16
        return ps, core.DistAdam(
            ps, ["attn"], lr=0.01, betas=(0.9, 0.95), weight_decay=0.1
        )

    def run(ps, opt, steps):
        for step in steps:
            opt.should_sync = True
            g = torch.Generator().manual_seed(7 * step + rank)
            xs = [(1e-2 * torch.randn(p.shape, generator=g)).bfloat16() for p in ps]
            sum((p * x.to(p.dtype)).sum() for p, x in zip(ps, xs)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    ref, ref_opt = make(False)
    bf, bf_opt = make(True)
    run(ref, ref_opt, range(6))
    run(bf, bf_opt, range(6))
    for p, q in zip(bf, ref):
        m = bf_opt.state[p]["master"]
        lo = 0 if p.numel() < 1024 else rank * len(m)
        assert m.dtype == torch.float32 and torch.equal(
            p[lo : lo + len(m)], m.bfloat16()
        )
        r = q.detach()[lo : lo + len(m)]
        assert rel(m, r) < 1e-3, rel(
            m, r
        )  # the reference feeds FP32 grads to BF16 moments
    state = mm.cpu_copy(bf_opt.state_dict())
    again, again_opt = make(True)
    with torch.no_grad():
        for p, q in zip(again, bf):
            p.copy_(q)
    again_opt.load_state_dict(state)
    for p, index in zip(again, state["param_groups"][0]["params"]):
        again_opt.state[p] = mm.cpu_copy(
            state["state"][index]
        )  # as load_rank_state_dict
    run(bf, bf_opt, range(6, 9))
    run(again, again_opt, range(6, 9))
    for p, q in zip(again, bf):
        assert torch.equal(p, q)
        assert torch.equal(again_opt.state[p]["master"], bf_opt.state[q]["master"])


def worker(rank, world, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    patch()
    install(
        core,
        Schedule(warmup_steps=2, mtp_steps=0, split_step=SPLIT, batch_rows=8),
        -1,
        STEPS,
    )
    for accum in (1, 3):
        test_reference(accum)
    test_resume()
    test_distadam_master()
    if rank == 0:
        print(
            f"world {world}: ok; --bf16-weights bit-identical to FP32 weights ({STEPS} steps, 1 and 3 "
            f"micro-batches, split at {SPLIT})"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    for world in (1, 2, 4):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mp.spawn(worker, args=(world, port), nprocs=world)
