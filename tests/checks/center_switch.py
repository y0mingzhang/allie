"""trainer --resume-new-source from arch moe_router_center_first 1 to 4 (migratable, centre_more) on CPU: the
moe_shard check's model (arch moe_shard, world 2, gloo; quantile balancing, centring 0.9) at 8 layers, block 0 dense, so
7 MoE layers, blocks 0-3 under eager checkpoints; once with --moe-batched-rebalance --moe-remat --moe-chunks 4, once
with neither.

A (center_first 1) trained 3 steps and saved as the trainer saves it (model.pt with whole experts, each rank's state)
loads into B (center_first 4) as the trainer loads it: MoE layers 1-3 (blocks 2-4) take fresh mu and mu_steps (0),
every saved tensor (layer 0's mu and mu_steps, the biases, the weights) and the rank state (optimizer, grads, schedule)
load bitwise. B's first step, whose new layers subtract mu = 0, is A's: the same losses and every saved tensor but layer
0's mu equal, and each new layer's mu is that step's mean router input over both ranks' micro-batches (layer 0's its EMA
step), mu_steps 1. B saved after 6 steps and resumed plainly on B is bitwise B's straight run to 8: losses, model.pt,
rank state. Refused: a decrease (4 -> 1, 0 -> 4), another arch change alongside, B's load without centre_more, and
with it a parent missing layer 0's mu (only the newly centred layers start fresh).

    TORCH_COMPILE_DISABLE=1 python tests/checks/center_switch.py
"""

import json
import os
import socket
import tempfile

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import moe_shard as T
from allie.data.packed import Packed
from allie.model import moe_kernels
from allie.model import shard as expert_shard
from allie.model.network import Config, TrainingManager, core, create_model
from allie.train.trainer import centre_more, migratable

ARCH = T.ARCH | dict(moe_shard=True)
SWITCH, SAVE, STEPS = 3, 6, 8
NEW = (1, 2, 3)  # the MoE layers B centres and A does not


def build(first):
    torch.manual_seed(0)
    cfg = Config(
        width=64, head_dim=16, layers=8, max_tokens=1024, scheduled_steps=STEPS,
        ckpt="eager", ckpt_frac=0.5, arch=ARCH | dict(moe_router_center_first=first),
    )  # fmt: skip
    model = create_model(cfg, device="cpu")
    return model, TrainingManager(model, cfg, T.SCHEDULE)


def save(whole, manager, path):
    os.makedirs(path, exist_ok=True)
    if dist.get_rank() == 0:
        torch.save(whole, f"{path}/model.pt")
    torch.save(manager.rank_state_dict(), f"{path}/rank{dist.get_rank()}.pt")
    dist.barrier()


def load(first, path, switch=False):
    """The trainer's resume: under switch its frozen-source branch (the rank state's arch rewritten, centre_more)."""
    model, manager = build(first)
    weights = torch.load(f"{path}/model.pt", mmap=True)
    state = torch.load(f"{path}/rank{dist.get_rank()}.pt", weights_only=False)
    if switch:
        weights = centre_more(model, weights, 1)
        state["config"] = dict(state["config"], arch=manager.cfg.arch)
    model.load_state_dict(weights)
    manager.load_rank_state_dict(state)
    return model, manager, state


def means(val, model, manager, step):
    """Train one step; per MoE layer its real forwards' mean router input over the ranks (backward recomputes skipped)."""
    sums, on = {}, [False]
    hooks = [
        model.register_forward_pre_hook(lambda *_: on.__setitem__(0, True)),
        model.register_forward_hook(lambda *_: on.__setitem__(0, False)),
    ]

    def book(m, args):
        if on[0]:
            h = args[0].reshape(-1, args[0].shape[-1])
            u = torch.cat(
                (h.float().sum(0), h.new_full((1,), len(h), dtype=torch.float32))
            )
            sums[m] = sums[m] + u if m in sums else u

    hooks += [m.register_forward_pre_hook(book) for m in manager.moe]
    out = T.train(val, model, manager, [step])
    for h in hooks:
        h.remove()
    for t in sums.values():
        dist.all_reduce(t)
    return out, [sums[m][:-1] / sums[m][-1].clamp(min=1) for m in manager.moe]


def worker(rank, world, port, tmp, batched):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    T.patch()
    core.BATCHED_REBALANCE = batched
    moe_kernels.SAVE_EXPANDED, moe_kernels.CHUNKS = not batched, 4 if batched else 1
    val = Packed(T.DATA, "val")
    tag = "batched rebalance, remat, chunks 4" if batched else "per-layer rebalance"
    say = lambda *a: rank == 0 and print(f"{tag}:", *a, flush=True)

    model, manager = build(1)
    saved = T.train(val, model, manager, range(SWITCH))[-1][1]
    save(saved, manager, f"{tmp}/a")
    mu0 = manager.moe[0].mu.clone()
    ref = T.train(val, model, manager, [SWITCH])
    expert_shard.reset()
    del model, manager

    model, manager, state = load(4, f"{tmp}/a", switch=True)
    assert [bool(m.center) for m in manager.moe] == [True] * 4 + [False] * 3
    fresh = {f"blocks.{i + 1}.mlp.{b}" for i in NEW for b in ("mu", "mu_steps")}
    whole = expert_shard.state_dict(model, T.cpu_copy)
    if rank == 0:
        assert whole.keys() - saved.keys() == fresh and saved.keys() <= whole.keys()
        assert T.same({k: whole[k] for k in saved}, saved) == 0.0
        assert not any(whole[k].any() for k in fresh)
        assert (
            whole["blocks.1.mlp.mu_steps"] == SWITCH and whole["blocks.1.mlp.mu"].any()
        )
    assert T.same(manager.rank_state_dict(), state) == 0.0
    say(
        f"switch 1 -> 4 at step {SWITCH}: new {sorted(fresh)} fresh; every saved tensor and the rank state bitwise"
    )

    (first,), mean = means(val, model, manager, SWITCH)
    assert T.same(first[0], ref[0][0]) == 0.0
    if rank == 0:
        old = lambda w: {k: w[k] for k in saved if k != "blocks.1.mlp.mu"}
        assert T.same(old(first[1]), old(ref[0][1])) == 0.0
    for i, m in enumerate(manager.moe[:4]):
        ema = mu0.lerp(mean[i], 1 - torch.tensor(0.9))
        mu, n = (mean[i], 1) if i in NEW else (ema, SWITCH + 1)
        assert torch.equal(m.mu, mu) and m.mu_steps == n, i
    say(f"step {SWITCH} is A's (losses, weights, biases); new layers' mu = its mean router input "
        f"(|mu| {[round(manager.moe[i].mu.norm().item(), 3) for i in NEW]}), mu_steps 1")  # fmt: skip

    head = T.train(val, model, manager, range(SWITCH + 1, SAVE))
    save(head[-1][1], manager, f"{tmp}/b")
    tail = T.train(val, model, manager, range(SAVE, STEPS))
    final = tail[-1][1], manager.rank_state_dict()
    expert_shard.reset()
    del model, manager

    model, manager, _ = load(4, f"{tmp}/b")
    again = T.train(val, model, manager, range(SAVE, STEPS))
    assert T.same([x[0] for x in again], [x[0] for x in tail]) == 0.0
    if rank == 0:
        assert T.same(again[-1][1], final[0]) == 0.0
    assert T.same(manager.rank_state_dict(), final[1]) == 0.0
    expert_shard.reset()
    del model, manager
    say(f"plain resume on B at {SAVE} -> {STEPS} bitwise: losses, model.pt, rank state")

    model = build(4)[0]
    weights = torch.load(f"{tmp}/a/model.pt", mmap=True)
    lost = {k: v for k, v in weights.items() if k != "blocks.1.mlp.mu"}
    for w in (weights, centre_more(model, lost, 1)):
        try:
            model.load_state_dict(w)
            refused = False
        except RuntimeError:
            refused = True
        assert refused, (
            "B loaded A's weights without centre_more, or without layer 0's mu"
        )
    expert_shard.reset()
    dist.destroy_process_group()


def refusals():
    arch = lambda first, **k: json.dumps(ARCH | dict(moe_router_center_first=first) | k)
    for old, new in ((1, 4), (1, 0), (4, 0), (1, 1), (0, 0)):
        assert migratable("arch", arch(old), arch(new)), (old, new)
    assert migratable("arch", arch(1), arch(4, moe_gate_floor=1e-10))  # + RESUMABLE
    for old, new in ((4, 1), (0, 4), (0, 1)):
        assert not migratable("arch", arch(old), arch(new)), (old, new)
    for k in (
        dict(moe_seq=0.002),
        dict(moe_router_center=0.8),
        dict(moe_dense_first=2),
    ):
        assert not migratable("arch", arch(1), arch(4, **k)), k
    print(
        "migratable: 1 -> 4, 1 -> 0, 4 -> 0 admitted; 4 -> 1, 0 -> 4 and another arch change refused",
        flush=True,
    )


if __name__ == "__main__":
    refusals()
    for batched in (True, False):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(2, port, tmp, batched), nprocs=2)
    print("PASS center switch")
