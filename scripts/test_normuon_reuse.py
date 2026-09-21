"""CPU proof of NorMuon buffer reuse against the unchanged ZeRO-2 optimizer.

Uses test_modded_zero's CPU collective/kernel stand-ins. Tests optimizer updates,
padding/idle owners, 3D expert groups, gradient overwrite and cross-path resume.
The queued full eight-GPU training comparison supplies the NCCL/compiled gate.
"""
import os
import socket
import argparse

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import test_modded_zero as z


def exact(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8)), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            exact(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f"{path}.{index}")
    else:
        assert a == b, (path, a, b)


def build(reuse):
    os.environ["NORMUON_REUSE_BUFFERS"] = str(int(reuse))
    model, manager = z.build(True)
    assert manager.muon_opt._reuse_buffers == reuse
    return model, manager


def snapshot(model, manager):
    return z.mm.cpu_copy(model.state_dict()), manager.rank_state_dict()


def poison_grad_workspace(manager):
    # The next first micro must overwrite every owned row, not read last step's
    # Nesterov-transformed workspace. Padding is deliberately not poisoned.
    for group in manager.muon_opt.param_groups:
        for p in group["params"]:
            if not hasattr(p, "main_grad"):
                continue  # FP32 gate parameters use the ordinary gradient path.
            assert p.fresh
            if p.main_grad is not None:
                p.main_grad.fill_(float("nan"))


def check(moe, accum, defer):
    z.ZERO2, z.DEFER = True, defer
    z.ARCH = dict(mlp="swiglu", moe=[8, 2], moe_kernel="pad") if moe else {}
    ref, ref_mgr = build(False)
    candidate, candidate_mgr = build(True)
    for step in range(z.STEPS):
        z.train(ref, ref_mgr, [step], accum)
        z.train(candidate, candidate_mgr, [step], accum)
        exact(snapshot(candidate, candidate_mgr), snapshot(ref, ref_mgr))
        if step == 3:
            # Resume the candidate from the stock optimizer's checkpoint, then
            # continue across the embedding split with its existing flat views.
            model_state, optimizer_state = snapshot(ref, ref_mgr)
            candidate, candidate_mgr = build(True)
            candidate.load_state_dict(model_state)
            candidate_mgr.load_rank_state_dict(optimizer_state)
            exact(snapshot(candidate, candidate_mgr), snapshot(ref, ref_mgr))
        if dist.get_world_size() > 1:
            poison_grad_workspace(candidate_mgr)
    if dist.get_rank() == 0:
        print(f"PASS world={dist.get_world_size()} moe={moe} accum={accum} defer={defer}", flush=True)


def worker(rank, world, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    z.patch()
    z.install(z.core, z.Schedule(warmup_steps=2, mtp_steps=0, split_step=z.SPLIT, batch_rows=8), -1, z.STEPS)
    for moe in (False, True):
        for accum in (1, 3):
            check(moe, accum, True)
    check(True, 3, False)  # reuse gradients with ordinary all-gather publication
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", nargs="+", type=int, default=[1, 2, 4])
    for world in parser.parse_args().worlds:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(worker, args=(world, port), nprocs=world)
