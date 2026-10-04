"""train.trainer's --resume-retune switches on CPU: another --lr-scale (TrainingManager.retune) and another --mix
(data.mix.same_until), each bitwise where it must be.

lr_scale: the moe_shard check's model (arch moe_shard, world 2, gloo) trained 6 steps at lr_scale 1, saved after 3 as
train.trainer saves it (model.pt with whole experts, each rank's state). Resumed unchanged, the run is bitwise the straight
one (losses, model.pt, rank state). Resumed at lr_scale 0.5 through retune, every optimizer group's initial_lr is half the
saved one and, after the first step, its lr is initial_lr x the schedule's factor, bitwise half the unchanged resume's; the
step's change of the FP32 masters is half the unchanged resume's: Adam's to 1e-3 (but the embed's, which the split at
this step replaces by the head's), NorMuon's a little under (its lr^2 weight decay quarters). Without retune the load
refuses the config.

mix: a control sampler on an old (2017-05) and a recent (2025-01) Lichess month, 24 rows, its state saved after 6.
recent50(2024-01):control loaded from that state, as train.trainer loads it, draws bitwise the rows of its own straight
run, which equal control's until the mark at row 12 and differ after it; the state after 24 rows is refused.
data.mix.same_until admits a cool/recent tail of the same inner policy up to its mark, also inside a composed policy,
and refuses a passed mark, another inner policy and a tail whose '+' factors differ from the start.

    TORCH_COMPILE_DISABLE=1 python tests/checks/resume_retune.py
"""

import os
import socket
import tempfile

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from allie import paths
from allie.data import mix as cm
from allie.data.packed import Packed
from allie.model import shard as expert_shard
from allie.model.network import Config, TrainingManager, create_model
import moe_shard as T

OLD = str(paths.DATA / "data-v1-hist2/2017-05")
NEW = str(paths.DATA / "data-v1/2025-01")
INNER, TAIL = "control", "recent50(2024-01):control"
ROWS, K = 24, 6  # rows drawn; the state saved after K (the tail's mark: row 12)


def build(lr_scale=1.0):
    torch.manual_seed(0)
    cfg = Config(
        width=64, head_dim=16, layers=8, max_tokens=1024, scheduled_steps=T.STEPS,
        lr_scale=lr_scale, ckpt="eager", ckpt_frac=0.5, arch=T.ARCH | dict(moe_shard=True),
    )  # fmt: skip
    model = create_model(cfg, device="cpu")
    return model, TrainingManager(model, cfg, T.SCHEDULE)


def groups(manager):
    return [g for opt in manager.optimizers for g in opt.param_groups]


def masters(manager):
    """Per optimizer with FP32 masters (NorMuon's per group, Adam's per parameter): this rank's, flattened."""
    out = []
    for opt in manager.optimizers:
        m = [g["master"] for g in opt.param_groups if "master" in g]
        m += [opt.state[p]["master"] for g in opt.param_groups for p in g["params"]
              if p.label != "embed" and "master" in opt.state.get(p, {})]  # fmt: skip
        if m:
            out.append(torch.cat([x.double().flatten() for x in m]))
    return out


def worker(rank, world, port, tmp):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    T.patch()
    val = Packed(T.DATA, "val")
    say = lambda *a: rank == 0 and print(*a, flush=True)

    model, manager = build()
    whole = T.train(val, model, manager, range(T.SAVE))[-1][1]
    if rank == 0:
        torch.save(whole, f"{tmp}/model.pt")
    torch.save(manager.rank_state_dict(), f"{tmp}/rank{rank}.pt")
    dist.barrier()
    straight = T.train(val, model, manager, range(T.SAVE, T.STEPS))
    final = straight[-1][1], manager.rank_state_dict()
    expert_shard.reset()
    del model, manager

    def resume(lr_scale, retune):
        model, manager = build(lr_scale)
        model.load_state_dict(torch.load(f"{tmp}/model.pt", mmap=True))
        state = torch.load(f"{tmp}/rank{rank}.pt", weights_only=False)
        if retune:
            manager.retune(state)
        manager.load_rank_state_dict(state)
        return model, manager

    saved = torch.load(f"{tmp}/rank{rank}.pt", weights_only=False)
    model, manager = resume(1.0, False)
    before = masters(manager)
    tail = T.train(val, model, manager, range(T.SAVE, T.SAVE + 1))
    lr1 = [g["lr"] for g in groups(manager)]
    moved1 = [(m - b).norm() for m, b in zip(masters(manager), before)]
    tail += T.train(val, model, manager, range(T.SAVE + 1, T.STEPS))
    assert T.same([x[0] for x in tail], [x[0] for x in straight]) == 0.0
    if rank == 0:
        assert T.same(tail[-1][1], final[0]) == 0.0
    assert T.same(manager.rank_state_dict(), final[1]) == 0.0
    expert_shard.reset()
    del model, manager
    say(
        f"unchanged resume at {T.SAVE} -> {T.STEPS} bitwise: losses, model.pt, rank state"
    )

    model, manager = resume(0.5, True)
    assert T.same(masters(manager), before) == 0.0
    old = [g for o in saved["optimizers"] for g in o["param_groups"]]
    assert all(
        g["initial_lr"] == 0.5 * o["initial_lr"] for g, o in zip(groups(manager), old)
    )
    assert manager.rank_state_dict()["config"]["lr_scale"] == 0.5
    T.train(val, model, manager, range(T.SAVE, T.SAVE + 1))
    factor = manager.schedule.lr(T.SAVE, manager.decay_start, manager.end_step)
    lr = [g["lr"] for g in groups(manager)]
    assert all(x == g["initial_lr"] * factor for x, g in zip(lr, groups(manager)))
    assert all(x == 0.5 * y for x, y in zip(lr, lr1))
    moved = [(m - b).norm() for m, b in zip(masters(manager), before)]
    ratio = torch.tensor([float(a / b) for a, b in zip(moved, moved1)])
    adam, muon = ratio[0], ratio[1:]
    assert abs(adam - 0.5) < 1e-3 and (muon > 0.49).all() and (muon <= 0.5).all(), ratio
    expert_shard.reset()
    del model, manager
    say(f"retune to lr_scale 0.5: {len(lr)} groups' lr = initial_lr x {factor:.3g}, half the unchanged; first step's "
        f"master change ratio {ratio.tolist()} (Adam, NorMuon x{len(muon)})")  # fmt: skip

    manager = build(0.5)[1]
    try:
        manager.load_rank_state_dict(
            torch.load(f"{tmp}/rank{rank}.pt", weights_only=False)
        )
        refused = False
    except AssertionError:
        refused = True
    assert refused, "lr_scale 0.5 loaded a lr_scale 1 state without retune"
    say("without retune the lr_scale 1 state is refused")
    dist.destroy_process_group()


def draw(sampler, n):
    return [sampler.batch(1).copy() for _ in range(n)]


def mix():
    kw = dict(months=[OLD, NEW], chunk=32)
    a = cm.Sampler(INNER, 42, total_rows=ROWS, **kw)
    head = draw(a, K)
    state = a.state_dict()
    inner = head + draw(a, ROWS - K)
    late = a.state_dict()  # past the mark
    tail = draw(cm.Sampler(TAIL, 42, total_rows=ROWS, **kw), ROWS)
    c = cm.Sampler(TAIL, 42, total_rows=ROWS, **kw)
    c.load_state_dict(state)  # as train.trainer does under --resume-retune
    resumed = draw(c, ROWS - K)
    assert all(np.array_equal(x, y) for x, y in zip(resumed, tail[K:])), (
        "the switch is not the tail's own run"
    )
    diff = [i for i in range(ROWS) if not np.array_equal(inner[i], tail[i])]
    assert diff and diff[0] >= ROWS // 2, diff
    frac = state["seen"] / a.total_rows
    assert cm.same_until(INNER, INNER, 1.0) and cm.same_until(INNER, TAIL, frac)
    assert cm.same_until(INNER, TAIL, 0.5) and not cm.same_until(INNER, TAIL, 0.6)
    assert not cm.same_until(
        INNER, "recent50(2024-01):up4", frac
    ) and not cm.same_until(INNER, "up4", frac)
    assert cm.same_until(INNER, "cool50:up4", frac) and cm.same_until(
        "up4", "cool50(up4):up8", frac
    )
    assert not cm.same_until("up4", "cool50:up8", frac)
    # a tail inside a composed policy: compose applies otb_x2 from the start
    assert not cm.same_until(INNER, "cool90:natural+otb_x2", frac)
    assert cm.same_until("control+otb_x2", "cool90:natural+otb_x2", frac)
    try:
        cm.Sampler(TAIL, 42, total_rows=ROWS, **kw).load_state_dict(late)
        refused = False
    except ValueError:
        refused = True
    assert refused
    print(f"mix: {INNER} -> {TAIL} at row {K} (progress {frac:.2f}) bitwise the tail's own run; equal to {INNER} "
          f"through row {diff[0] - 1}, differs from row {diff[0]} (mark at row {ROWS // 2}); refusals hold", flush=True)  # fmt: skip


if __name__ == "__main__":
    mix()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(2, port, tmp), nprocs=2)
    print("PASS resume retune")
