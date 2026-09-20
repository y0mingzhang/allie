"""CPU proof of the --async-checkpoint writer (modded_checkpoints.AsyncSaver) on two gloo ranks.

save() mirrors modded_train.save() for both paths on nested rank/model/data state (bf16/fp32/fp64,
non-contiguous and 0-dim tensors, None grads, RNG state, Python scalars). Checked: every async file,
pointer and log row equals its synchronous twin byte for byte although the sources change in place
right after each snapshot; buffers are reused and reallocated on a shape change; the writer thread
blocks the trainer's signals; an OSError(512) on the first fsync is retried; pointers move only once
every rank's files are durable, the next save waits for the in-flight one, and a failed or killed
write never moves them. Not covered: CUDA pinned copies and modded_train's own call sites.
Run: python test_async_checkpoint.py
"""

import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path


def worker(mode, tmp, rank):
    import numpy as np
    import torch
    import torch.distributed as dist

    import modded_checkpoints as mc
    from lm_checkpoint import atomic_save, rng_state
    from modded_medium import cpu_copy

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp}/pg-{mode}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    group = dist.new_group(backend="gloo")
    real_atomic, real_fsync = mc.atomic_save, os.fsync

    def content(k):
        g = torch.Generator().manual_seed(10 * k + rank)

        def r(*shape, dtype=torch.float32):
            return torch.randn(*shape, generator=g).to(dtype)

        n = 8 if k < 3 else 9  # shape change at save 3
        torch.manual_seed(k)
        random.seed(k)
        np.random.seed(k)
        manager = {
            "config": {"width": 128, "arch": {"moe": [16, 2]}},
            "rank": rank,
            "world": 2,
            "optimizers": [
                {
                    "state": {
                        0: {
                            "step": torch.tensor(float(k)),
                            "exp_avg": r(n, 4, dtype=torch.bfloat16),
                            "exp_avg_sq": r(n, 4),
                        },
                        1: {"momentum_buffer": r(4, n).t()},
                    },
                    "param_groups": [
                        {
                            "lr": 0.01 * k,
                            "betas": (0.9, 0.95),
                            "params": [0, 1],
                            "lr_cpu": r(1, dtype=torch.float64),
                        }
                    ],
                }
            ],
            "optimizer_flags": [{"freeze_timer": k, "should_sync": False}],
            "gradients": OrderedDict(
                w=r(n, 4, dtype=torch.bfloat16) if k % 2 else None, b=r(4)
            ),
            "split_embed": k > 1,
            "schedule_step": 128 * k,
            "yarn": {
                "cos": r(32, 16, dtype=torch.bfloat16),
                "angular_freq": r(16),
                "attn_scale": 0.1 * k,
            },
        }
        model = OrderedDict(
            [
                ("embed.weight", r(n, 4, dtype=torch.bfloat16)),
                ("blocks.0.w", r(4, 4)),
                ("lm_head.weight", r(4, n).t()),
            ]
        )
        data = {
            "policy": "p",
            "rng": np.random.default_rng(k).bit_generator.state,
            "cursor": {"a": [k, 2]},
            "seen": 1000 * k,
            "pool_ids": list(range(50 * k)),
        }
        return manager, model, data, rng_state()

    def tensors(x):
        if isinstance(x, torch.Tensor):
            yield x
        elif isinstance(x, (dict, list, tuple)):
            for v in x.values() if isinstance(x, dict) else x:
                yield from tensors(v)

    def save(out, k, names, saver=None):
        step = 128 * k
        directory = out / "checkpoints" / f"step-{step:08d}-{step:08x}"
        if rank == 0:
            directory.mkdir(parents=True)
        dist.barrier()
        manager, model, data, rng = content(k)
        host, write = (saver.snapshot, saver.add) if saver else (cpu_copy, atomic_save)
        write(
            {"manager": host(manager), "rng": rng, "data": data, "flops": step},
            directory / f"rank{rank}.pt",
        )
        if rank == 0:
            state = {
                "model": host(model),
                "step": step,
                "metrics": {"move_ce": 1 / k},
                "inference": host({"yarn": manager["yarn"]}),
            }
            write(state, directory / "model.pt")

        def commit(save=mc.durable_save):
            assert all((directory / f"rank{i}.pt").exists() for i in range(2))
            mc.publish(out, directory, step, 2, names, save)
            removed = mc.prune(out, 2, directory)
            row = {"step": step, "directory": str(directory.relative_to(out))}
            with (out / "checkpoints.jsonl").open("a") as f:
                f.write(json.dumps(row | {"removed": removed}) + "\n")

        if saver:
            saver.start(commit if rank == 0 else None)
        else:
            dist.barrier()
            if rank == 0:
                commit(atomic_save)
            dist.barrier()
        return manager, model

    def pointer(out):
        return torch.load(out / "last.pt", weights_only=False)["step"]

    def logged(out):
        rows = (out / "checkpoints.jsonl").read_text().splitlines()
        return [json.loads(x)["step"] for x in rows]

    if mode == "crash":
        out = Path(tmp) / "crash"
        saver = mc.AsyncSaver(group)
        save(out, 1, ["last.pt"], saver)
        saver.flush()
        if rank == 1:
            mc.atomic_save = lambda *a: (time.sleep(30), real_atomic(*a))
        save(out, 2, ["last.pt"], saver)
        if rank == 0:
            saver.thread.join()  # rank 0's files are durable
        for _ in range(5):
            saver.poll()
            time.sleep(0.1)
        assert rank == 1 or pointer(out) == 128
        os._exit(0)  # the job dies with rank 1's files unwritten

    if rank == 0:
        out = Path(tmp) / "crash"
        assert pointer(out) == 128
        d = out / "checkpoints" / f"step-{256:08d}-{256:08x}"
        assert (d / "model.pt").exists() and not (d / "rank1.pt").exists()
        assert logged(out) == [128]

    sync, done = Path(tmp) / "sync", Path(tmp) / "async"
    saver = mc.AsyncSaver(group)
    names = {
        1: ["last.pt", "best.pt"],
        2: ["last.pt", "fork-256.pt", "best.pt"],
        3: ["last.pt"],
        4: ["last.pt"],
    }
    masks, buffers = [], []

    def spy(*a):
        main = threading.current_thread() is threading.main_thread()
        masks.append((main, signal.pthread_sigmask(signal.SIG_BLOCK, [])))
        time.sleep(0.2)  # the in-place updates below land before torch.save runs
        real_atomic(*a)

    mc.atomic_save = spy
    for k, dest in names.items():
        save(sync, k, dest)
        saver.flush()
        manager, model = save(done, k, dest, saver)
        for t in tensors((manager, model)):
            t.add_(1)
        manager["optimizers"][0]["param_groups"][0]["lr"] = -1.0
        manager["config"]["arch"]["moe"].append(0)
        # odd saves commit from poll, even ones from the next flush
        while k % 2 and saver.pending:
            saver.poll()
            time.sleep(0.05)
        for _ in range(3):
            saver.poll()
        buffers.append({p: b.data_ptr() for p, (_, b) in saver.buffers.items()})
    saver.flush()
    mc.atomic_save = real_atomic
    sigs = {signal.SIGINT, signal.SIGTERM, signal.SIGUSR1}
    assert {main for main, _ in masks} == {False, rank == 0}
    assert all(sigs <= m if not main else not sigs & m for main, m in masks)
    exp = (1, "optimizers", 0, "state", 0, "exp_avg")
    assert all(buffers[1][p] == v for p, v in buffers[0].items())
    assert buffers[2][exp] != buffers[1][exp]
    assert buffers[2][(1, "yarn", "cos")] == buffers[1][(1, "yarn", "cos")]

    calls = []

    def flaky(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise OSError(512, "Unknown error 512")
        real_fsync(fd)

    save(sync, 5, ["last.pt"])
    os.fsync = flaky
    save(done, 5, ["last.pt"], saver)
    saver.flush()
    os.fsync = real_fsync
    # rank file twice, then rank 0's model.pt and last.pt
    assert len(calls) == (4 if rank == 0 else 2), calls

    def slow(*a):
        time.sleep(2)
        real_atomic(*a)

    save(sync, 6, ["last.pt"])
    if rank == 1:
        mc.atomic_save = slow
    save(done, 6, ["last.pt"], saver)
    for _ in range(5):
        saver.poll()
        time.sleep(0.1)
    assert rank == 1 or pointer(done) == 640, "published before rank 1 was durable"
    waited = saver.flush()
    mc.atomic_save = real_atomic
    assert waited > 0.5 and (rank == 1 or pointer(done) == 768), waited

    dist.barrier()
    if rank == 0:

        def files(out):
            return sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())

        assert files(sync) == files(done), (files(sync), files(done))
        assert all(
            (sync / f).read_bytes() == (done / f).read_bytes() for f in files(sync)
        )
        kept = sorted(p.name[:13] for p in (done / "checkpoints").iterdir())
        assert kept == ["step-00000256", "step-00000640", "step-00000768"], kept
    dist.barrier()

    def broken(fd):
        raise OSError(512, "Unknown error 512")

    if rank == 1:
        os.fsync = broken
    save(done, 7, ["last.pt"], saver)
    try:
        saver.flush()
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed write not reported")
    os.fsync = real_fsync
    assert rank == 1 or (pointer(done) == 768 and 896 not in logged(done))
    print(f"rank {rank}: passed (torch {torch.__version__})", flush=True)
    dist.destroy_process_group()


def main():
    if len(sys.argv) == 4:
        return worker(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    tmp = tempfile.mkdtemp(prefix="async-ckpt-")
    for mode in ("crash", "check"):
        procs = [
            subprocess.Popen([sys.executable, __file__, mode, tmp, str(r)])
            for r in range(2)
        ]
        assert [p.wait() for p in procs] == [0, 0], mode
    print("passed", tmp)


if __name__ == "__main__":
    main()
