"""train.tail.TailEMA on CPU, gloo worlds 1, 2 and 4.

A toy model of replicated tensors and rank-local expert shards (dim 0 of a whole [8, 3, 5] tensor, rank r's rows
r * 8 / w ..) follows a known trajectory over 30 steps (every rank publishes the same values; a shard is its rows of
the whole tensor). With every 3, horizons 0.1 and 0.25 (window from step 7):
- the written tail-ema-H.pt equals a reference that keeps whole FP32 EMAs on one process with the same schedule
  (initialised at step 7, then every 3rd step and the last, decay (1 - 1/h)^gap), to FP32 rounding; dtypes are the
  parameters'; ema_from == window_from;
- a state from a 27-step run (the same window start, another long horizon) is refused, and one rank's mismatch
  restarts every rank;
- a run saved at step 17 (inside the window) and continued in a fresh TailEMA from the state writes the same files
  bitwise; a state under other settings (every 2) is refused and the EMAs restart, recorded as ema_from 19 (the first update after 17).

    python tests/checks/tail_ema.py
"""

import os
import socket
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from allie.train.tail import TailEMA

TOTAL, FRACS, EVERY = 30, [0.1, 0.25], 3
EXPERTS = 8


def trajectory(step):
    g = torch.Generator().manual_seed(1000 + step)
    return dict(
        a=torch.randn(6, 4, generator=g),
        b=torch.randn(16, generator=g).bfloat16(),
        c=torch.randn(3, 2, 2, generator=g),
        experts=torch.randn(EXPERTS, 3, 5, generator=g).bfloat16(),
    )


class Toy(nn.Module):
    def __init__(self, rank, world):
        super().__init__()
        t, n = trajectory(0), EXPERTS // world
        self.a, self.c = nn.Parameter(t["a"]), nn.Parameter(t["c"])
        self.b = nn.Parameter(t["b"])
        self.experts = nn.Parameter(t["experts"][rank * n : (rank + 1) * n].clone())
        self.experts.local = True
        self.rank, self.n = rank, n

    @torch.no_grad()
    def publish(self, step):
        t = trajectory(step)
        for k in ("a", "b", "c"):
            getattr(self, k).copy_(t[k])
        self.experts.copy_(t["experts"][self.rank * self.n : (self.rank + 1) * self.n])


def run(rank, world, out, steps, state=None, every=EVERY, total=TOTAL):
    """Steps 1..steps; returns the TailEMA state at the end (as saved: a copy)."""
    model = Toy(rank, world)
    group = dist.new_group(backend="gloo")
    tail = TailEMA(model, FRACS, total, every, group)
    tail.load_state_dict(state)
    first = 1 if state is None or tail.last is None else steps_done(state)
    for step in range(first, steps + 1):
        model.publish(step)
        if tail.due(step):
            tail.update(step)
    if steps == total:
        tail.write(out, total)
    s = tail.state_dict()
    return (
        None if s is None else s | dict(ema=[[t.clone() for t in e] for e in s["ema"]])
    )


def steps_done(state):
    return state["saved_at"] + 1


def reference():
    horizons = [max(1, round(f * TOTAL)) for f in FRACS]
    start = TOTAL - 3 * max(horizons)
    ema, last = None, None
    for step in range(1, TOTAL + 1):
        if not (step > start and ((step - start - 1) % EVERY == 0 or step == TOTAL)):
            continue
        x = {k: v.float() for k, v in trajectory(step).items()}
        if ema is None:
            ema, last = [{k: v.clone() for k, v in x.items()} for _ in horizons], step
            continue
        gap, last = step - last, step
        for e, h in zip(ema, horizons):
            for k in e:
                e[k].lerp_(x[k], 1 - (1 - 1 / h) ** gap)
    return ema, start + 1


def worker(rank, world, port, out):
    os.environ |= dict(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    out = Path(out)
    straight = out / "straight"
    resumed = out / "resumed"
    restarted = out / "restarted"
    if rank == 0:
        for d in (straight, resumed, restarted):
            d.mkdir()
    dist.barrier()
    run(rank, world, straight, TOTAL)
    saved = run(rank, world, None, 17)
    saved["saved_at"] = 17
    run(rank, world, resumed, TOTAL, saved)
    other = run(rank, world, None, 17, every=2)  # another cadence's state: refused
    other["saved_at"] = 17
    model = Toy(rank, world)
    tail = TailEMA(model, FRACS, TOTAL, EVERY, dist.new_group(backend="gloo"))
    tail.load_state_dict(other)
    assert tail.ema is None
    # refused too: 27 steps have the same window start (6) but another long horizon (7, not 8)
    short = run(rank, world, None, 17, total=27)
    short["saved_at"] = 17
    probe = TailEMA(Toy(rank, world), FRACS, TOTAL, EVERY, dist.new_group(backend="gloo"))
    assert probe.start == TailEMA(Toy(rank, world), FRACS, 27, EVERY, dist.new_group(backend="gloo")).start
    probe.load_state_dict(short)
    assert probe.ema is None
    # unanimous: one rank's mismatched state restarts every rank
    if world > 1:
        probe = TailEMA(Toy(rank, world), FRACS, TOTAL, EVERY, dist.new_group(backend="gloo"))
        probe.load_state_dict(other if rank == 1 else saved)
        assert probe.ema is None, rank
    for step in range(18, TOTAL + 1):
        model.publish(step)
        if tail.due(step):
            tail.update(step)
    tail.write(restarted, TOTAL)
    dist.barrier()
    if rank == 0:
        ref, window = reference()
        dtypes = {k: v.dtype for k, v in trajectory(0).items()}
        for f, e in zip(FRACS, ref):
            got = torch.load(straight / f"tail-ema-{f:g}.pt", weights_only=False)
            assert got["ema_from"] == got["window_from"] == window, (
                got["ema_from"],
                window,
            )
            for k, v in e.items():
                w = got["ema"][k]
                assert w.dtype == dtypes[k] and w.shape == v.shape, (
                    k,
                    w.dtype,
                    w.shape,
                )
                fp32 = w.dtype == torch.float32  # a BF16 cast may round one ulp apart
                tol = dict(rtol=0, atol=2e-6) if fp32 else dict(rtol=2**-7, atol=1e-6)
                assert torch.allclose(w.float(), v.to(dtypes[k]).float(), **tol), k
            back = torch.load(resumed / f"tail-ema-{f:g}.pt", weights_only=False)
            assert back["ema_from"] == window
            assert all(
                torch.equal(back["ema"][k], got["ema"][k]) for k in got["ema"]
            ), "resume not exact"
            again = torch.load(restarted / f"tail-ema-{f:g}.pt", weights_only=False)
            assert again["ema_from"] == 19 != again["window_from"], again["ema_from"]
        print(f"world {world}: PASS", flush=True)
    dist.destroy_process_group()


def main():
    for world in (1, 2, 4):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as out:
            mp.spawn(worker, (world, port, out), nprocs=world, join=True)
    print("PASS tail ema")


if __name__ == "__main__":
    main()
