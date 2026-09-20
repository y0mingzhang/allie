"""DeepSeek-style fine-grained mixture of experts (model track, tier 2).

One shared expert plus E routed experts, top-k by sigmoid affinity; selection adds a per-expert bias
that is nudged toward balanced load after every optimizer step (auxiliary-loss-free balancing,
DeepSeek-V3); gate weights are the selected affinities renormalised to sum to sqrt(k) (V3 uses 2.5 at
k = 8, Kimi K2 2.83). Experts are ReLU^2 MLPs like the dense one. Dispatch is capacity-padded with a
unique row per assignment, so both directions are index_copy / gathers without atomics and shapes are
static for torch.compile. Training: capacity 1.25, overflowing assignments dropped (their rows stay
zero; the dropped fraction is logged). Evaluation: dropless (capacity = tokens), so a token's output
does not depend on the rest of the batch. Nominal active FLOPs match the dense MLP (shared_hidden + k *
expert_hidden = 4d, k experts executed per token); dropped routes do no MLP work. kernel="scatter"
replaces the padded dispatch with ScatterMoE's fused gather-GEMM-scatter kernels (modded_smoe):
dropless in training too, no padding, capacity unused.
"""

import math

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch import nn
from torch.distributed import _functional_collectives as funcol
from torch.nn import functional as F

from modded_smoe import parallel_linear
from modded_smoe_tuned import parallel_linear as tuned_linear
from modded_smoe_aligned_linear import parallel_linear as gather_linear

# MoE.stats, per layer over the last step (modded_train logs them as moe_<name>)
STATS = (
    "imbalance",
    "min_load",
    "starved",
    "dropped",
    "bias_routed",
    "bias_range",
    "margin",
)


class AddLoss(torch.autograd.Function):
    """Identity on x whose backward also sends gradient 1 into loss (DeepSeek's auxiliary-loss
    hook): a loss term computed inside a module without changing its return value. Contract: the
    loss's gradient is 1 regardless of how the objective is scaled downstream, so the caller bakes
    the objective's scaling into loss (here the trainer's world / 8); a GradScaler or an external
    loss weight would not reach it and needs the same treatment."""

    @staticmethod
    def forward(ctx, x, loss):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, torch.ones((), dtype=torch.float32, device=grad.device)


BLOCK_RECOMPUTE = False  # set by create_model under --ckpt eager (whole blocks recomputed eagerly)


@torch.compiler.disable
def _recompute(module, *args):
    # Keep the optimized callable wholly outside the outer Dynamo trace, which
    # otherwise unwraps a compiled bound method passed across this boundary.
    if not hasattr(module, "_compiled_experts_out"):
        module._compiled_experts_out = torch.compile(module.experts_out, dynamic=False, fullgraph=True)
    return torch.utils.checkpoint.checkpoint(module._compiled_experts_out, *args, use_reentrant=False)


def full_state(model, state):
    """state_dict with every sharded expert param (whole-expert rows per rank) all-gathered to the
    full tensor; collective, all ranks must call it."""
    for name, p in model.named_parameters():
        if getattr(p, "label", "").endswith("_sh"):
            parts = [torch.empty_like(p) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, p.detach().contiguous())
            # one layer on GPU at a time; only rank 0 keeps the full tensor (it writes model.pt)
            state[name] = torch.cat([t.cpu() for t in parts]) if dist.get_rank() == 0 else p
    return state


def local_state(model, state):
    """Inverse of full_state for this rank: full sharded expert tensors sliced to its rows."""
    state = dict(state)
    for name, p in model.named_parameters():
        if getattr(p, "label", "").endswith("_sh") and state[name].shape[0] != p.shape[0]:
            n = p.shape[0]
            state[name] = state[name][dist.get_rank() * n : (dist.get_rank() + 1) * n]
    return state


class Gather(torch.autograd.Function):
    """Sharded experts (ZeRO-3 style): each rank's whole-expert shard, cast to the compute dtype, gathered
    along the expert dim. The backward reduce-scatters the full gradient back to the shards in FP32,
    averaged like the replicated grads, and returns it in the parameter's dtype."""

    @staticmethod
    def forward(ctx, shard, dtype):
        ctx.dtype = shard.dtype
        x = shard.to(dtype)
        return funcol.wait_tensor(funcol.all_gather_tensor(x, 0, dist.group.WORLD))

    @staticmethod
    def backward(ctx, g):
        g = funcol.reduce_scatter_tensor(
            g.float().contiguous(), "avg", 0, dist.group.WORLD
        )
        return funcol.wait_tensor(g).to(ctx.dtype), None


class MoE(nn.Module):
    def __init__(
        self, dim, experts, topk, expert_hidden, shared_hidden, init=0.02, router_lr_mul=0.1,
        gamma=1e-3, seq=0.0, update="sign", capacity=1.25, kind="relu2", score="sigmoid",
        kernel="pad", shard=False,
    ):  # fmt: skip
        super().__init__()
        assert update in ("sign", "prop") and score in ("sigmoid", "sqrtsoftplus")
        assert kernel in ("pad", "scatter", "scatter-tuned", "scatter-accum", "scatter-gather", "scatter-dualgather")
        self.score, self.kernel = score, kernel
        self.experts, self.topk, self.capacity, self.kind = experts, topk, capacity, kind
        up = 2 if kind == "swiglu" else 1  # SwiGLU experts: gate and value rows
        # bias update speed and rule (sign: gamma * sign(mean - load), DeepSeek-V3; prop: gamma *
        # clamp((mean - load) / mean, -1, 1) x the rebalance() scale, which settles instead of
        # cycling at +-gamma once balanced), sequence-wise balance loss weight
        self.gamma, self.update, self.seq = gamma, update, seq
        bound = 3**0.5 * 0.5 * dim**-0.5  # the dense MLP's c_fc init
        self.router = nn.Parameter(torch.randn(experts, dim) * init)
        # one 3D tensor per layer for all experts (per-expert tensors cost a kernel each per step)
        self.up = nn.Parameter(torch.empty(experts, up * expert_hidden, dim))
        self.up.data.uniform_(-bound, bound)
        self.down = nn.Parameter(torch.zeros(experts, expert_hidden, dim))
        # shard: each rank keeps experts // world whole experts (same init as unsharded, then sliced)
        self.sharded = bool(shard) and dist.is_initialized() and dist.get_world_size() > 1
        if self.sharded:
            w, r = dist.get_world_size(), dist.get_rank()
            assert experts % w == 0, f"{experts} experts over {w} ranks"
            n = experts // w
            dev = "cuda" if dist.get_backend() == "nccl" else "cpu"
            full = [self.up.data.to(dev), self.down.data.to(dev)]
            for t in full:  # rank 0's init for every shard, whatever each rank's seed
                dist.broadcast(t, 0)
            self.up = nn.Parameter(full[0][r * n : (r + 1) * n].to(self.up.device).clone())
            self.down = nn.Parameter(full[1][r * n : (r + 1) * n].to(self.down.device).clone())
        self.shared = shared_hidden > 0
        if self.shared:
            self.shared_up = nn.Parameter(
                torch.empty(up * shared_hidden, dim).uniform_(-bound, bound)
            )
            self.shared_down = nn.Parameter(torch.zeros(shared_hidden, dim))
        shared = (self.shared_up, self.shared_down) if self.shared else ()
        self.router.label, self.router.lr_mul, self.router.wd_mul = (
            "router",
            router_lr_mul,
            0.0,
        )
        # NorMuon stacks a label's params and orthogonalizes each expert matrix; differently shaped
        # SwiGLU ups get their own label
        self.up.label, self.down.label = "moe_up" if up == 2 else "moe", "moe"
        if self.sharded:  # rank-local params: their own (local) NorMuon, no replica reduction
            self.up.label, self.down.label = self.up.label + "_sh", "moe_sh"
        for p in shared:
            p.label = "mlp_shared"
        if up == 2 and self.shared:
            self.shared_up.label = "mlp_shared_up"
        for p in (self.down, *shared[1:]):
            p.lr_mul = 2.0  # the dense MLP's c_proj multiplier
        self.register_buffer(
            "bias", torch.zeros(experts)
        )  # selection only; checkpointed
        # accumulated over a step's micro-batches: assignments per expert, dropped assignments,
        # tokens whose bias changed their top-k, tokens, summed k-th vs (k+1)-th affinity margins
        self.register_buffer("load", torch.zeros(experts + 4), persistent=False)
        self.register_buffer(
            "stats", torch.tensor([1.0, 1, 0, 0, 0, 0, 0]), persistent=False
        )

    def forward(self, x):
        shape, d = x.shape, x.shape[-1]
        h = x.reshape(-1, d)
        t, e, k = h.shape[0], self.experts, self.topk
        n = t * k
        g = torch.promote_types(h.dtype, torch.float32)  # FP32 gate, as DeepSeek-V3
        s = F.linear(h.to(g), self.router.to(g))
        s = torch.sigmoid(s) if self.score == "sigmoid" else F.softplus(s).clamp_min(1e-12).sqrt()
        idx = torch.topk(s + self.bias, k, dim=-1).indices
        w = s.gather(1, idx)
        w = w * (k**0.5 / w.sum(-1, keepdim=True))
        flat = idx.flatten()
        count = flat.new_zeros(e).scatter_add_(0, flat, torch.ones_like(flat))
        order = flat.argsort(stable=True)
        if self.sharded and self.training and torch.is_grad_enabled() and not BLOCK_RECOMPUTE:
            # eager checkpoint: the compiled partitioner keeps opaque autograd Functions' saved tensors
            # (gathered experts, expert activations) for every layer; eager recompute re-gathers them
            routed, dropped = _recompute(self, h, w, flat, count, order)
        else:
            routed, dropped = self.experts_out(h, w, flat, count, order)
        if self.training:
            with torch.no_grad():
                top = torch.topk(s, k + 1, dim=-1)
                own = torch.zeros_like(s).scatter_(1, top.indices[:, :k], 1.0)
                moved = (own.gather(1, idx) < 1).any(-1).sum()
                margin = (top.values[:, k - 1] - top.values[:, k]).sum()
                extra = [dropped, moved, torch.full_like(moved, t), margin]
            self.load[:e] += count.float()
            self.load[e:] += torch.stack([v.float() for v in extra])
        if self.shared:
            shared = self.act(F.linear(h, self.shared_up.type_as(h)))
            routed = routed + F.linear(shared, self.shared_down.T.type_as(h))
        out = routed.view(shape)
        if self.training and self.seq:
            # DeepSeek-V3's sequence-wise balance loss, one 1024-token row = one sequence: per row
            # sum_i f_i P_i with f_i = e / k * share of the row's routes to expert i and P_i = its
            # mean normalised affinity. The trainer's objective is a SUM over tokens (x world / 8),
            # so each row's penalty is weighted by its 1024 input tokens (seq = weight per input
            # token, scored or not; masking only changes which tokens carry the CE): the total is
            # additive over rows, hence independent of how rows are split into micro-batches
            assert t % 1024 == 0
            sel = torch.zeros_like(s).scatter_(1, idx, 1.0).view(-1, 1024, e).mean(1)
            prob = (s / s.sum(-1, keepdim=True)).view(-1, 1024, e).mean(1)
            scale = dist.get_world_size() / 8 if dist.is_initialized() else 1 / 8
            loss = self.seq * scale * 1024 * (sel * e / k * prob).sum()
            out = AddLoss.apply(out, loss)
        return out

    def experts_out(self, h, w, flat, count, order):
        """Routed experts' weighted output for assignments flat (sorted by order, count per expert)."""
        k = self.topk
        if self.sharded:
            up, down = Gather.apply(self.up, h.dtype), Gather.apply(self.down, h.dtype)
        else:
            up, down = self.up.type_as(h), self.down.type_as(h)
        if self.kernel in ("scatter", "scatter-tuned", "scatter-accum", "scatter-gather", "scatter-dualgather"):  # dropless
            offs = count.cumsum(0)
            se = flat[order]
            linear = (gather_linear if self.kernel in ("scatter-gather", "scatter-dualgather") else
                      tuned_linear if self.kernel in ("scatter-tuned", "scatter-accum") else parallel_linear)
            direct = self.kernel == "scatter-accum" and self.training and torch.is_grad_enabled()
            up_args, down_args = {}, {}
            if self.kernel in ("scatter-gather", "scatter-dualgather"):
                up_args = dict(gather=True)
            if self.kernel == "scatter-dualgather":
                down_args = dict(gated=True)
            if direct:
                assert self.up.dtype == self.down.dtype == torch.bfloat16
                up_flat,up_row=self.up.acc
                down_flat,down_row=self.down.acc
                up_args = dict(main_grad=up_flat,main_grad_row=up_row,fresh=self.up.fresh,main_grad_transposed=True)
                down_args = dict(main_grad=down_flat,main_grad_row=down_row,fresh=self.down.fresh)
            y = self.act(linear(h, up.transpose(1, 2), k, se, order, offs, grouped_out=True, **up_args))
            routed = linear(
                y, down, 1, se, order, offs, grouped_in=True, gates=w.type_as(h), **down_args
            )
            dropped = torch.zeros((), dtype=torch.long, device=h.device)
        else:
            routed, dropped = self.padded(h, w, flat, count, order, up, down)
        return routed, dropped

    def padded(self, h, w, flat, count, order, up, down):
        """Capacity-padded dispatch: a unique slot per kept assignment, so both directions are
        index_copy / gathers without atomics and shapes are static. Dropped assignments all write
        the one dummy row (a benign duplicate-index race: that row is discarded)."""
        (t, k), (e, d), n = w.shape, (self.experts, h.shape[1]), flat.numel()
        arange = torch.arange(n, device=h.device)
        start = count.cumsum(0) - count
        pos = torch.empty_like(flat).index_copy_(0, order, arange - start[flat[order]])
        cap = math.ceil(self.capacity * n / e) if self.training else t
        rows = e * cap  # expert slots, then one dummy row for every dropped assignment
        slot = torch.where(pos < cap, flat * cap + pos, rows)
        buf = h.new_zeros(rows + 1, d).index_copy(
            0, slot, h[:, None].expand(t, k, d).reshape(n, d)
        )
        y = self.act(torch.bmm(buf[:rows].view(e, cap, d), up.transpose(1, 2)))
        y = torch.bmm(y, down)
        # back to assignments by the inverse map (free slots to spare rows; the zero dummy row to
        # a dropped assignment, whose output stays zero): a gather both ways
        back = torch.arange(n, n + rows + 1, device=h.device).index_copy(0, slot, arange)
        y = torch.cat((y.reshape(rows, d), y.new_zeros(1, d)))
        y = y.new_zeros(n + rows + 1, d).index_copy(0, back, y)[:n].view(t, k, d)
        return (y * w.type_as(y)[..., None]).sum(1), (pos >= cap).sum()

    def act(self, x):
        if self.kind == "swiglu":
            a, b = x.chunk(2, dim=-1)
            return F.silu(a) * b
        return F.relu(x).square()

    @torch.no_grad()
    def rebalance(self, scale=1.0):
        """After an optimizer step: raise the bias of under-loaded experts, lower over-loaded ones.
        scale: the prop rule's decay (1 for the first 80% of training, then linearly to 0)."""
        if dist.is_initialized():
            dist.all_reduce(self.load)
        e = self.experts
        load = self.load[:e]
        dropped, moved, tokens, margin = self.load[e:]
        mean = load.sum().clamp(min=1) / e
        if self.update == "sign":
            self.bias += self.gamma * torch.sign(load.mean() - load)
        else:
            self.bias += self.gamma * scale * ((mean - load) / mean).clamp(-1, 1)
        tokens = tokens.clamp(min=1)
        self.stats.copy_(
            torch.stack(
                (
                    load.max() / mean, load.min() / mean, (load < 0.1 * mean).sum().float(),
                    dropped / (mean * e), moved / tokens, self.bias.max() - self.bias.min(),
                    margin / tokens,
                )
            )
        )  # fmt: skip
        self.load.zero_()
