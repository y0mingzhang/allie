"""Tail EMAs of the weights (trainer --tail-ema): a readout, the run itself unchanged.

Each horizon h (a fraction of the steps) keeps an FP32 EMA of every parameter, decay 1 - 1/h per step, over the last
3 x the longest horizon. Updates come every `every` steps (and at the last), each with the decay of the steps since
the previous one, from the published BF16 weights copied to host memory. The EMAs live on the host, split over the
ranks: each rank keeps its own expert shards (arch moe_shard) and a share of the replicated parameters (whole tensors,
balanced by size), so a rank holds about 4 bytes per parameter it keeps per horizon. Their state goes into the rank
files, so a resume continues them exactly; one that cannot (another world, other settings) restarts them, and the
written file records the step they restarted from (experiments.tail_ema skips it). At the last step rank 0 assembles
whole tensors (every rank's shards in rank order) and writes tail-ema-H.pt in each parameter's dtype (atomic).
"""

import time

import torch
import torch.distributed as dist

from allie.train.checkpoints import durable_save


class TailEMA:
    def __init__(self, model, fracs, total, every, group):
        self.fracs, self.total, self.every, self.group = fracs, total, every, group
        self.horizons = [max(1, round(f * total)) for f in fracs]
        self.start = total - 3 * max(self.horizons)  # the window is start + 1 .. total
        rank, world = dist.get_rank(), dist.get_world_size()
        named = list(model.named_parameters())
        load, self.owner = [0] * world, {}
        for n, p in sorted(named, key=lambda x: (-x[1].numel(), x[0])):
            if getattr(p, "local", False):
                self.owner[n] = None  # every rank keeps its own shard
            else:
                self.owner[n] = min(range(world), key=lambda r: (load[r], r))
                load[self.owner[n]] += p.numel()
        self.named = named
        self.mine = [(n, p) for n, p in named if self.owner[n] in (None, rank)]
        self.ema = self.ema_from = self.last = None
        self.seconds = 0.0  # host time in update(), this leg

    def due(self, step):
        return step > self.start and (
            (step - self.start - 1) % self.every == 0 or step == self.total
        )

    @torch.no_grad()
    def update(self, step):
        """After step's optimizer step, once the published weights are current (sync_params)."""
        if not self.due(step):
            return
        t0 = time.monotonic()
        if self.ema is None:  # also a resume that could not restore them
            self.ema = [
                [p.detach().to("cpu", torch.float32, copy=True) for _, p in self.mine] for _ in self.horizons
            ]
            self.ema_from = self.last = step
            self.seconds += time.monotonic() - t0
            return
        gap, self.last = step - self.last, step
        rates = [1 - (1 - 1 / h) ** gap for h in self.horizons]
        for i, (_, p) in enumerate(self.mine):
            x = p.detach().cpu().float()
            for e, r in zip(self.ema, rates):
                e[i].lerp_(x, r)
        self.seconds += time.monotonic() - t0

    def state_dict(self):
        if self.ema is None:
            return None
        return self.identity() | dict(ema_from=self.ema_from, last=self.last, ema=self.ema)

    def identity(self):
        """What a saved state must match to continue: the schedule, the layout and this rank's kept tensors."""
        world = dist.get_world_size()
        layout = dict(world=world, local=world, rank=dist.get_rank())  # local: ranks a node, all of them
        keys = dict(fracs=self.fracs, horizons=self.horizons, total=self.total, every=self.every, start=self.start)
        return keys | layout | dict(kept=[(n, tuple(p.shape)) for n, p in self.mine])

    def load_state_dict(self, s):
        """Collective (gloo group), on every rank of a resume: the saved EMAs continue only if every rank's state
        matches; otherwise all restart (the written file records where)."""
        mine = self.identity()
        ok = s is not None and {k: s.get(k) for k in mine} == mine
        shapes = [tuple(p.shape) for _, p in self.mine]
        ok = ok and len(s["ema"]) == len(self.horizons)
        ok = ok and all([tuple(t.shape) for t in e] == shapes for e in s["ema"])
        flag = torch.tensor(int(ok))
        if dist.get_world_size() > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        if not flag:
            return
        self.ema = [[t.detach().to("cpu", torch.float32, copy=True) for t in e] for e in s["ema"]]
        self.ema_from, self.last = s["ema_from"], s["last"]

    @torch.no_grad()
    def write(self, out, step):
        """Collective (gloo group): rank 0 writes tail-ema-H.pt per horizon. A rank without EMAs (none were due)
        writes nothing; every rank has the same window, so all agree."""
        if self.ema is None:
            return
        rank, world = dist.get_rank(), dist.get_world_size()
        index = {n: i for i, (n, _) in enumerate(self.mine)}
        span = torch.tensor([self.ema_from, -self.ema_from])
        if world > 1:  # the ranks' EMAs started together (load_state_dict is unanimous); recorded only if so
            dist.all_reduce(span, op=dist.ReduceOp.MAX, group=self.group)
        ema_from = int(span[0]) if span[0] == -span[1] else None
        for f, ema in zip(self.fracs, self.ema):
            whole = {}
            for n, p in self.named:
                o = self.owner[n]
                if o is None:  # every rank's shard in rank order, sent to rank 0 one at a time
                    t = ema[index[n]]
                    if rank == 0:
                        parts = [t] + [torch.empty_like(t) for _ in range(1, world)]
                        for r in range(1, world):
                            dist.recv(parts[r], r, group=self.group)
                        whole[n] = torch.cat(parts).to(p.dtype)
                    else:
                        dist.send(t, 0, group=self.group)
                elif o == 0:
                    if rank == 0:
                        whole[n] = ema[index[n]].to(p.dtype)
                elif rank == o:
                    dist.send(ema[index[n]], 0, group=self.group)
                elif rank == 0:
                    t = torch.empty(p.shape, dtype=torch.float32)
                    dist.recv(t, o, group=self.group)
                    whole[n] = t.to(p.dtype)
            if rank == 0:
                meta = dict(step=step, horizon=f, every=self.every, ema_from=ema_from, window_from=self.start + 1)
                meta["update_seconds"] = self.seconds
                durable_save(meta | dict(ema=whole), out / f"tail-ema-{f:g}.pt")
            del whole
