"""Compiled GPT (board CNN, SwiGLU MoE E16 top-2 with the shared expert, BF16
weights with ZeRO-2 masters, --ckpt eager) on real validation rows; torchrun on one or two GPUs.

recompute: a training step's output and every gradient with the eager block checkpoints equal
those of the same per-block compiled graphs run without them.
resume: 8 steps crossing the multi-token switch (4) and the embed split (5), saved after 3 steps
(model state and rank_state_dict, in memory), rebuilt, restored and continued: losses and every
model and optimizer tensor equal the uninterrupted run's. Not covered: world-size resharding.
ARCH='{"moe_shard": true}' (JSON, merged into the arch): the same with the experts sharded (the
model state with whole experts, as model.pt keeps them).

    torchrun --standalone --nproc_per_node=2 tests/checks/checkpoint_resume.py
"""

import gc
import io
import json
import os

import numpy as np
import torch
import torch.distributed as dist

from allie import paths
from allie.model import shard as expert_shard
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
SCHEDULE = Schedule(warmup_steps=2, mtp_steps=4, split_step=5, batch_rows=2)
ORIGINAL = core._eager_block
ARCH = json.loads(os.environ.get("ARCH", "{}"))


def build():
    torch.manual_seed(701)
    cfg = Config(
        width=128, head_dim=64, layers=8, max_tokens=1024, scheduled_steps=8, ckpt="eager",
        arch=dict(moe=[16, 2], moe_seq=0.001, moe_update="quantile") | ARCH,
    )  # fmt: skip
    m = create_model(cfg, device=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    with torch.no_grad():  # nonzero expert outputs: every expert grad carries signal
        for name, p in m.named_parameters():
            if name.endswith(("down", "c_proj")):
                p.normal_(0, 0.01)
    mgr = TrainingManager(m, cfg, SCHEDULE)
    return m, mgr, torch.compile(m, dynamic=False, fullgraph=False)


def batch(val, step, micro):
    rows = torch.as_tensor(
        val.rows(np.array([100 * step + 10 * micro + dist.get_rank()]))
    )
    return rows[:, :-1].cuda(), rows[:, 1:].cuda()


def advance(val, m, mgr, net, step):
    mgr.advance_schedule(step)
    primary = []
    for micro in range(2):
        if micro == 1:
            mgr.activate_hooks(step)
        x, y = batch(val, step, micro)
        ctx = make_context(x, mgr.ws_short * 128, mgr.ws_long * 128)
        logits = net(x.flatten(), ctx, mgr.get_forward_args())
        loss, move, count = move_losses(logits, x, y, ctx, mgr.mtp_weights)
        (loss * (dist.get_world_size() / 8)).backward()
        primary.append(float(move.detach() / count))
    mgr.step_optimizers(step)
    return primary


def snapshot(m, mgr):
    """Every rank's copy of the model state (whole experts) and its own rank state, once the owners'
    updates of the last step have arrived."""
    core.sync_params()
    whole = [expert_shard.state_dict(m, cpu_copy)]
    dist.broadcast_object_list(whole, 0)
    return dict(model=whole[0], manager=mgr.rank_state_dict())


def equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b), path
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(equal(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(equal(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    assert a == b, (path, a, b)
    return 0


def recompute(val, checkpointed):
    core._eager_block = (
        ORIGINAL
        if checkpointed
        else torch.compiler.disable(
            lambda module, x, attn_args, blend, ckpt=True: ORIGINAL(
                module, x, attn_args, blend, ckpt=False
            )
        )
    )
    m, mgr, net = build()
    mgr.advance_schedule(0)
    for _ in range(3):
        for p in m.parameters():
            p.grad, p.fresh = None, True
        x, y = batch(val, 0, 0)
        ctx = make_context(x, mgr.ws_short * 128, mgr.ws_long * 128)
        z = net(x.flatten(), ctx, mgr.get_forward_args())
        z.float().square().mean().backward()
    proof = {"output": z.detach().cpu()}
    for name, p in m.named_parameters():
        grads = (p.grad, getattr(p, "main_grad", None), getattr(p, "part", None))
        grad = next((g for g in grads if g is not None), None)
        if grad is not None:
            proof[name] = grad.detach().cpu().clone()
    shards = {n for n, p in m.named_parameters() if getattr(p, "local", False)}
    assert shards <= proof.keys(), shards - proof.keys()
    core._eager_block = ORIGINAL
    del m, mgr, net
    gc.collect()
    torch.cuda.empty_cache()
    return proof


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    )
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    val = Packed(DATA, "val")

    ref, got = recompute(val, False), recompute(val, True)
    assert ref.keys() == got.keys()
    bad = [k for k in ref if not torch.equal(ref[k], got[k])]
    assert not bad, bad
    print(json.dumps(dict(rank=dist.get_rank(), recompute_exact=len(ref))), flush=True)

    m, mgr, net = build()
    for s in range(3):
        advance(val, m, mgr, net, s)
    buf = io.BytesIO()
    torch.save(snapshot(m, mgr), buf)
    losses = [advance(val, m, mgr, net, s) for s in range(3, 8)]
    expected = snapshot(m, mgr)
    assert m.split_embed
    del m, mgr, net
    gc.collect()
    torch.cuda.empty_cache()
    m, mgr, net = build()
    buf.seek(0)
    saved = torch.load(buf, weights_only=False)
    m.load_state_dict(saved["model"])
    mgr.load_rank_state_dict(saved["manager"])
    resumed = [advance(val, m, mgr, net, s) for s in range(3, 8)]
    assert losses == resumed, (losses, resumed)
    now = snapshot(m, mgr)
    count = equal(expected, now)
    print(json.dumps(dict(rank=dist.get_rank(), resume_exact=count, losses=losses)))
    dist.destroy_process_group()
    print("PASS checkpoint recompute and resume", flush=True)


if __name__ == "__main__":
    main()
