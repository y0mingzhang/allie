"""DeepSeek-style fine-grained mixture of experts (model track, tier 2).

One shared expert plus E routed experts, top-k by sigmoid affinity; selection adds a per-expert bias
that is nudged toward balanced load after every optimizer step (auxiliary-loss-free balancing,
DeepSeek-V3); gate weights are the selected affinities renormalised to sum to sqrt(k) (V3 uses 2.5 at
k = 8, Kimi K2 2.83). Experts are ReLU^2 MLPs like the dense one. Dispatch is capacity-padded with a
unique row per assignment, so both directions are index_copy / gathers without atomics and shapes are
static for torch.compile. Training: capacity 1.25, overflowing assignments dropped (their rows stay
zero; the dropped fraction is logged). Evaluation: dropless (capacity = tokens), so a token's output
does not depend on the rest of the batch. Nominal active FLOPs match the dense MLP (shared_hidden + k *
expert_hidden = 4d, k experts executed per token); dropped routes do no MLP work.
"""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class MoE(nn.Module):
    def __init__(self, dim, experts, topk, expert_hidden, shared_hidden, capacity=1.25):
        super().__init__()
        self.experts, self.topk, self.capacity = experts, topk, capacity
        bound = 3**0.5 * 0.5 * dim**-0.5  # the dense MLP's c_fc init
        self.router = nn.Parameter(torch.randn(experts, dim) * 0.02)
        self.up = nn.ParameterList(
            nn.Parameter(torch.empty(expert_hidden, dim).uniform_(-bound, bound))
            for _ in range(experts)
        )
        self.down = nn.ParameterList(
            nn.Parameter(torch.zeros(expert_hidden, dim)) for _ in range(experts)
        )
        self.shared_up = nn.Parameter(
            torch.empty(shared_hidden, dim).uniform_(-bound, bound)
        )
        self.shared_down = nn.Parameter(torch.zeros(shared_hidden, dim))
        self.router.label, self.router.lr_mul, self.router.wd_mul = "router", 0.1, 0.0
        for p in (*self.up, *self.down):
            p.label = "moe"
        for p in (self.shared_up, self.shared_down):
            p.label = "mlp_shared"
        for p in (*self.down, self.shared_down):
            p.lr_mul = 2.0  # the dense MLP's c_proj multiplier
        self.register_buffer(
            "bias", torch.zeros(experts)
        )  # selection only; checkpointed
        # assignments per expert, then dropped assignments, accumulated over a step's micro-batches
        self.register_buffer("load", torch.zeros(experts + 1), persistent=False)
        # last step's max / mean and min / mean tokens per expert, experts under 10% of the mean,
        # dropped fraction (collapse logging)
        self.register_buffer(
            "stats", torch.tensor([1.0, 1.0, 0.0, 0.0]), persistent=False
        )

    def forward(self, x):
        shape, d = x.shape, x.shape[-1]
        h = x.reshape(-1, d)
        t, e, k = h.shape[0], self.experts, self.topk
        n = t * k
        g = torch.promote_types(h.dtype, torch.float32)  # FP32 gate, as DeepSeek-V3
        s = torch.sigmoid(F.linear(h.to(g), self.router.to(g)))
        idx = torch.topk(s + self.bias, k, dim=-1).indices
        w = s.gather(1, idx)
        w = w * (k**0.5 / w.sum(-1, keepdim=True))
        flat = idx.flatten()
        onehot = F.one_hot(flat, e)
        pos = (onehot.cumsum(0) * onehot).sum(-1) - 1  # rank within its expert
        cap = math.ceil(self.capacity * n / e) if self.training else t
        if self.training:
            self.load[:e] += onehot.sum(0).float()
            self.load[e] += (pos >= cap).sum().float()
        rows = e * cap + n  # expert slots, then one spill row per assignment
        arange = torch.arange(n, device=h.device)
        slot = torch.where(pos < cap, flat * cap + pos, e * cap + arange)
        buf = h.new_zeros(rows, d).index_copy(
            0, slot, h[:, None].expand(t, k, d).reshape(n, d)
        )
        buf = buf[: e * cap].view(e, cap, d)
        y = F.relu(
            torch.bmm(buf, torch.stack(list(self.up)).type_as(h).transpose(1, 2))
        )
        y = torch.bmm(y.square(), torch.stack(list(self.down)).type_as(h))
        # back to assignments by the inverse map (free rows to spare rows): a gather both ways
        back = torch.arange(n, n + rows, device=h.device).index_copy(0, slot, arange)
        y = torch.cat((y.reshape(e * cap, d), y.new_zeros(n, d)))
        y = y.new_zeros(n + rows, d).index_copy(0, back, y)[:n].view(t, k, d)
        routed = (y * w.type_as(y)[..., None]).sum(1)
        shared = F.relu(F.linear(h, self.shared_up.type_as(h))).square()
        shared = F.linear(shared, self.shared_down.T.type_as(h))
        return (routed + shared).view(shape)

    @torch.no_grad()
    def rebalance(self, rate=1e-3):
        """After an optimizer step: raise the bias of under-loaded experts, lower over-loaded ones."""
        if dist.is_initialized():
            dist.all_reduce(self.load)
        load, dropped = self.load[:-1], self.load[-1]
        self.bias += rate * torch.sign(load.mean() - load)
        total = load.sum().clamp(min=1)
        mean = total / load.numel()
        starved = (load < 0.1 * mean).sum()
        self.stats.copy_(
            torch.stack(
                (load.max() / mean, load.min() / mean, starved, dropped / total)
            )
        )
        self.load.zero_()
