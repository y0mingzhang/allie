"""DeepSeek-style fine-grained mixture of experts.

One shared expert plus E routed experts, top-k by sigmoid affinity; selection adds a per-expert bias
that is nudged toward balanced load after every optimizer step (auxiliary-loss-free balancing,
DeepSeek-V3), or set outright to the balancing quantile of the step's router margins (Kimi K3's
Quantile Balancing, update="quantile"); gate weights are the selected affinities renormalised to sum
to sqrt(k) (V3 uses 2.5 at k = 8, Kimi K2 2.83). Experts are SwiGLU MLPs like the dense one, run dropless by modded_smoe's fused
kernels. Nominal active FLOPs match the dense MLP (shared_hidden + k * expert_hidden =
swiglu_hidden).
"""

import contextlib
import functools

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

import modded_smoe

# MoE.stats, per layer over the last step (modded_train logs them as moe_<name>); dropped is always
# 0 (dropless), kept so logs and the load buffer keep their layout
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


# floor of the sums of sigmoid scores that normalise a token's gates (its k selected scores) and its sequence-wise
# balance affinities (all e scores); 0: off, bitwise the plain sums. Below a sum of ~1e-19 the compiled division's
# backward overflows (k**0.5 / sum**2), below ~1e-38 its forward: the big run's step 60081 went nonfinite on one token
# whose 16 selected layer-1 scores summed to 5.7e-20. Floored, such a token's gates sum to k**0.5 * sum / floor.
# Set by modded_train --moe-gate-floor
GATE_FLOOR = 0.0


def floored(total):
    return total.clamp_min(GATE_FLOOR) if GATE_FLOOR else total


STATS_REPS = (
    1  # times a training forward adds its loads and stats (replay_context: 2, then 0)
)


@contextlib.contextmanager
def _reps(n, replay=False):
    global STATS_REPS
    STATS_REPS, modded_smoe.REPLAY = n, replay
    try:
        yield
    finally:
        STATS_REPS, modded_smoe.REPLAY = 1, False


def replay_context():
    """checkpoint context_fn: the forward adds its loads and stats twice, the recompute skips them
    and the routed combine (modded_smoe.combine_op). The recompute used to add them again,
    identically (a deterministic replay). Doubling alone keeps the counts and every rebalance ratio
    and sign but not a margin sum over micro-batches (fl(fl(2a + b) + b) != 2 fl(a + b)); two
    in-order adds keep the buffer bitwise."""
    return _reps(2 if torch.is_grad_enabled() else 1), _reps(0, True)


# Opaque ops read STATS_REPS when they run, so forward and recompute share one compiled graph: a
# replay must save the same tensors, which a graph compiled without the stats need not.
@torch.library.custom_op("allie_moe::stats_topk", mutates_args=())
def stats_topk(s: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.topk(s, k); zeros, uncomputed, while STATS_REPS is 0."""
    if STATS_REPS:
        return tuple(torch.topk(s, k, dim=-1))
    return s.new_zeros(len(s), k), s.new_zeros(len(s), k, dtype=torch.long)


@stats_topk.register_fake
def _(s, k):
    return s.new_empty(len(s), k), s.new_empty(len(s), k, dtype=torch.long)


@torch.library.custom_op("allie_moe::route_topk", mutates_args=())
def route_topk(
    s: torch.Tensor, bias: torch.Tensor, k: int, stats: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch.topk(s + bias, k).indices and torch.topk(s, k + 1), or zeros while STATS_REPS is 0."""
    return modded_smoe.route(s, bias, k, stats and STATS_REPS > 0)


@route_topk.register_fake
def _(s, bias, k, stats):
    t = len(s)
    return (
        s.new_empty(t, k, dtype=torch.long), s.new_empty(t, k + 1),
        s.new_empty(t, k + 1, dtype=torch.long),
    )  # fmt: skip


@torch.library.custom_op("allie_moe::book", mutates_args={"load"})
def book(load: torch.Tensor, u: torch.Tensor) -> None:
    for _ in range(STATS_REPS):
        load.add_(u)


@torch.library.custom_op("allie_moe::book_once", mutates_args={"acc"})
def book_once(acc: torch.Tensor, u: torch.Tensor) -> None:
    """acc += u once per real forward: replay_context's doubled forward and skipped recompute alike."""
    if STATS_REPS:
        acc.add_(u)


def counts(keys, e):
    """Occurrences of 0..e-1 in sorted integer keys. Exact and atomic-free: scatter_add_ runs as a
    sorting index_put under torch.use_deterministic_algorithms."""
    ends = torch.searchsorted(keys, torch.arange(e, device=keys.device), right=True)
    return ends.diff(prepend=ends.new_zeros(1))


def qb_margins(s, bias, idx):
    """x = alpha - (s + bias), alpha the token's (k+1)-th best biased score (Kimi K3): with alpha fixed,
    the token routes to expert j under bias + d iff x_j < d_j."""
    b = s + bias
    alpha = b.scatter(1, idx, float("-inf")).amax(1, keepdim=True)
    return alpha - b


def qb_bin(x):
    """modded_smoe's QB bin of each margin."""
    sm, n = modded_smoe, modded_smoe.QB_LEVELS
    a = x.abs().view(torch.int32).clamp(sm.QB_LO, sm.QB_HI)
    level = (a - sm.QB_LO) >> (23 - sm.QB_MANT)
    return torch.where(x < 0, n - 1 - level, n + level).long()


def qb_counts(s, bias, idx):
    """Per expert, the counts of qb_margins over tokens in modded_smoe's QB bins, [E, QB_BINS]."""
    e, nb = s.shape[1], modded_smoe.QB_BINS
    key = qb_bin(qb_margins(s, bias, idx)) + nb * torch.arange(e, device=s.device)
    return torch.bincount(key.flatten(), minlength=nb * e).view(e, nb)


@functools.cache
def qb_edges(device):
    """The QB bins' edges, ascending: bin i holds x in [edges[i], edges[i + 1]) for x >= 0, in
    (edges[i], edges[i + 1]] for x < 0."""
    sm = modded_smoe
    up = torch.arange(1, sm.QB_LEVELS + 1, dtype=torch.int32) << (23 - sm.QB_MANT)
    up = (up + sm.QB_LO).view(torch.float32).double()
    return torch.cat((-up.flip(0), up.new_zeros(1), up)).to(device)


def qb_shift(hist, k):
    """The d_j putting k / e of each expert's counted tokens below it, the CDF linear within a bin
    (exact at bin edges, so the error is within one bin)."""
    e = hist.shape[0]
    # integer cumsum: a float one has no deterministic CUDA kernel
    h, cum = hist.double(), hist.long().cumsum(1).double()
    q = cum[:, -1:] * (k / e)
    j = torch.searchsorted(cum, q).clamp_(max=hist.shape[1] - 1)
    hj = h.gather(1, j)
    frac = ((q - cum.gather(1, j) + hj) / hj.clamp(min=1)).clamp_(0, 1)
    edges = qb_edges(hist.device)
    lo, hi = edges[j], edges[j + 1]
    return (lo + frac * (hi - lo)).squeeze(1)


@torch.library.custom_op("allie_moe::qb_book", mutates_args={"hist"})
def qb_book(
    hist: torch.Tensor, s: torch.Tensor, bias: torch.Tensor, idx: torch.Tensor
) -> None:
    if not STATS_REPS:
        return
    if s.is_cuda:
        modded_smoe.qb_hist(s, bias, idx, hist, STATS_REPS)
    else:
        hist.add_(qb_counts(s, bias, idx), alpha=STATS_REPS)


class MoE(nn.Module):
    def __init__(
        self, dim, experts, topk, expert_hidden, shared_hidden, init=0.02, router_lr_mul=0.1,
        gamma=1e-3, seq=0.0, update="sign", score="sigmoid", seq_raw=False, router_wd=0.0, center=0.0,
    ):  # fmt: skip
        super().__init__()
        assert update in ("sign", "prop", "quantile")
        assert score in ("sigmoid", "sqrtsoftplus")
        assert update != "quantile" or score == "sigmoid"  # the QB bins end at |x| = 16
        assert topk < experts
        self.experts, self.topk, self.score = experts, topk, score
        # bias update speed and rule (sign: gamma * sign(mean - load), DeepSeek-V3; prop: gamma *
        # clamp((mean - load) / mean, -1, 1) x the rebalance() scale, which settles instead of
        # cycling at +-gamma once balanced; quantile: qb_shift, no gamma), sequence-wise balance loss
        # weight, its counts from the raw top-k of the scores instead of the bias-adjusted selection
        self.gamma, self.update, self.seq, self.seq_raw = gamma, update, seq, seq_raw
        bound = 3**0.5 * 0.5 * dim**-0.5  # the dense MLP's c_fc init
        self.router = nn.Parameter(torch.randn(experts, dim) * init)
        # one 3D tensor per layer for all experts (per-expert tensors cost a kernel each per step);
        # up rows: gate then value
        self.up = nn.Parameter(torch.empty(experts, 2 * expert_hidden, dim))
        self.up.data.uniform_(-bound, bound)
        self.down = nn.Parameter(torch.zeros(experts, expert_hidden, dim))
        self.shared = shared_hidden > 0
        if self.shared:
            self.shared_up = nn.Parameter(
                torch.empty(2 * shared_hidden, dim).uniform_(-bound, bound)
            )
            self.shared_down = nn.Parameter(torch.zeros(shared_hidden, dim))
            self.shared_up.label, self.shared_down.label = "mlp_shared_up", "mlp_shared"
            self.shared_down.lr_mul = 2.0  # the dense MLP's c_proj multiplier
        self.router.label, self.router.lr_mul, self.router.wd_mul = (
            "router",
            router_lr_mul,
            0.0,
        )
        self.router.decoupled_wd = router_wd
        # NorMuon stacks a label's params and orthogonalizes each expert matrix
        self.up.label, self.down.label, self.down.lr_mul = "moe_up", "moe", 2.0
        self.register_buffer(
            "bias", torch.zeros(experts)
        )  # selection only; checkpointed
        # accumulated over a step's micro-batches: assignments per expert, dropped assignments,
        # tokens whose bias changed their top-k, tokens, summed k-th vs (k+1)-th affinity margins
        self.register_buffer("load", torch.zeros(experts + 4), persistent=False)
        self.register_buffer(
            "stats", torch.tensor([1.0, 1, 0, 0, 0, 0, 0]), persistent=False
        )
        if update == "quantile":  # qb_counts over a step's micro-batches
            hist = torch.zeros(experts, modded_smoe.QB_BINS, dtype=torch.int32)
            self.register_buffer("hist", hist, persistent=False)
        self.remat = True  # modded_smoe.routed's: False where a checkpoint recomputes the layer
        # center: EMA decay of the mean router input mu, subtracted before the router (0: off). A shared
        # input direction makes W_j . mu a per-expert constant that saturates the sigmoid scores (big run
        # layer 0: 77% of the input energy in mu, routing nearly token-independent). mu averages past
        # steps' global batch means, so a token's routing never sees other tokens
        assert 0 <= center < 1
        self.center = center
        if center:
            self.register_buffer("mu", torch.zeros(dim))
            self.register_buffer("mu_steps", torch.zeros(()))
            # summed router inputs and their count, once per real forward
            self.register_buffer("mu_sum", torch.zeros(dim + 1), persistent=False)

    def forward(self, x):
        shape, d = x.shape, x.shape[-1]
        h = x.reshape(-1, d)
        t, e, k = h.shape[0], self.experts, self.topk
        g = torch.promote_types(h.dtype, torch.float32)  # FP32 gate, as DeepSeek-V3
        s = F.linear(h.to(g) - self.mu if self.center else h.to(g), self.router.to(g))
        s = (
            torch.sigmoid(s)
            if self.score == "sigmoid"
            else F.softplus(s).clamp_min(1e-12).sqrt()
        )
        if e <= 128:  # the kernel holds a row of experts in registers
            idx, *stats = route_topk(s.detach(), self.bias, k, self.training)
        else:  # trunk's path: torch.topk here, the stats top-k below
            idx, stats = torch.topk(s + self.bias, k, dim=-1).indices, None
        w = s.gather(1, idx)
        w = w * (k**0.5 / floored(w.sum(-1, keepdim=True)))
        flat = idx.flatten()
        order = flat.argsort(stable=True)
        count = counts(flat[order], e)
        up, down = self.up.type_as(h), self.down.type_as(h)
        out = modded_smoe.routed(
            h,
            up.transpose(1, 2),
            down,
            k,
            flat[order],
            order,
            count.cumsum(0),
            w.type_as(h),
            self.remat,
        )
        if self.training:
            with torch.no_grad():
                val, top = stats or stats_topk(s, k + 1)
                own = torch.zeros_like(s).scatter_(1, top[:, :k], 1.0)
                moved = (own.gather(1, idx) < 1).any(-1).sum()
                margin = (val[:, k - 1] - val[:, k]).sum()
                extra = [
                    torch.zeros_like(moved),
                    moved,
                    torch.full_like(moved, t),
                    margin,
                ]
                book(
                    self.load,
                    torch.cat((count.float(), torch.stack([v.float() for v in extra]))),
                )
                if self.update == "quantile":
                    qb_book(self.hist, s.detach(), self.bias, idx)
                if self.center:
                    book_once(self.mu_sum, torch.cat((h.float().sum(0), h.new_full((1,), t, dtype=torch.float32))))
        if self.shared:
            shared = self.act(F.linear(h, self.shared_up.type_as(h)))
            out = out + F.linear(shared, self.shared_down.T.type_as(h))
        out = out.view(shape)
        if self.training and self.seq:
            # DeepSeek-V3's sequence-wise balance loss, 1024 tokens (a row) = one sequence: per row
            # sum_i f_i P_i with f_i = e / k * share of the row's routes to expert i and P_i = its
            # mean normalised affinity. The trainer's objective is a SUM over tokens (x world / 8),
            # so each row's penalty is weighted by its 1024 input tokens (seq = weight per input
            # token, scored or not; masking only changes which tokens carry the CE): the total is
            # additive over rows, hence independent of how rows are split into micro-batches
            assert t % 1024 == 0
            top = torch.topk(s.detach(), k, dim=-1).indices if self.seq_raw else idx
            sel = torch.zeros_like(s).scatter_(1, top, 1.0).view(-1, 1024, e).mean(1)
            prob = (s / floored(s.sum(-1, keepdim=True))).view(-1, 1024, e).mean(1)
            loss = (
                self.seq
                * (dist.get_world_size() / 8)
                * 1024
                * (sel * e / k * prob).sum()
            )
            out = AddLoss.apply(out, loss)
        return out

    @staticmethod
    def act(x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

    @torch.no_grad()
    def rebalance(self, scale=1.0):
        """After an optimizer step: raise the bias of under-loaded experts, lower over-loaded ones.
        scale: the prop rule's decay (1 for the first 80% of training, then linearly to 0); quantile
        solves for the balancing bias instead of stepping toward it, and ignores it."""
        dist.all_reduce(self.load)
        e = self.experts
        load = self.load[:e]
        dropped, moved, tokens, margin = self.load[e:]
        mean = load.sum().clamp(min=1) / e
        if self.update == "quantile":
            dist.all_reduce(self.hist)
            b = self.bias + qb_shift(self.hist, self.topk)
            self.bias.copy_(b - b.mean())
            self.hist.zero_()
        elif self.update == "sign":
            self.bias += self.gamma * torch.sign(load.mean() - load)
        else:
            self.bias += self.gamma * scale * ((mean - load) / mean).clamp(-1, 1)
        tokens = tokens.clamp(min=1)
        if self.center:  # the first step takes its mean outright
            dist.all_reduce(self.mu_sum)
            beta = torch.where(self.mu_steps > 0, self.center, 0.0)
            self.mu.lerp_(self.mu_sum[:-1] / self.mu_sum[-1].clamp(min=1), 1 - beta)
            self.mu_steps += 1
            self.mu_sum.zero_()
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
