"""Expert sharding (arch moe_shard), ZeRO-3 style: each rank keeps experts // world whole routed experts of every
MoE layer (their BF16 rows, FP32 masters and NorMuon state: modded_medium_core.NorMuon local).

gather() all-gathers a layer's BF16 experts where its forward needs them and launches the next layer's gather
(prefetch). A layer under an eager block checkpoint (modded_moe.MoE.remat False) saves its gathered experts for
backward: the checkpoint drops them in the forward and its recompute holds them until the block's own backward.
Any other layer saves its shards and gathers again in backward. The backward computes the full expert grads and
reduce-scatters them (BF16, averaged, in flight until the previous layer's backward is done) into the shards' FP32
grad rows. model.pt keeps whole experts: state_dict() gathers them, load_state_dict slices them (same world size).
"""

import torch
import torch.distributed as dist
from torch import nn

import modded_smoe

SHARDED = []  # every sharded MoE layer, each model's in forward order
PREFETCH = True
_span = []  # per layer: its model's first and end positions
_ready = {}  # layer position -> (works, up, down): gathers in flight
_pending = []  # (work, shard, its grad, the whole grads it reads): reduce-scatters in flight


def shard(layers):
    """Keep this rank's experts of each MoE layer: the same init, sliced; loading slices whole experts."""
    w, r, lo = dist.get_world_size(), dist.get_rank(), len(SHARDED)
    SHARDED.extend(layers)
    _span.extend([(lo, len(SHARDED))] * len(layers))
    for i, m in enumerate(layers):
        assert m.experts % w == 0, f"{m.experts} experts over {w} ranks"
        n = m.experts // w
        for name in ("up", "down"):
            old = getattr(m, name)
            p = nn.Parameter(old.data[r * n : (r + 1) * n].clone())
            p.__dict__.update(old.__dict__)
            p.local = True  # a rank-local NorMuon steps it, no replica to reduce over
            setattr(m, name, p)
        m.pos = lo + i
        m.register_load_state_dict_pre_hook(_slice)


def _slice(m, state, prefix, *_):
    r = dist.get_rank()
    for name in ("up", "down"):
        t, n = state[prefix + name], getattr(m, name).shape[0]
        assert t.shape[0] == m.experts, "model.pt keeps whole experts"
        state[prefix + name] = t[r * n : (r + 1) * n]


def _whole(p):
    return p.new_empty((p.shape[0] * dist.get_world_size(), *p.shape[1:]))


def _launch(i):
    out = [_whole(p) for p in (SHARDED[i].up, SHARDED[i].down)]
    works = [
        dist.all_gather_into_tensor(t, p.detach(), async_op=True)
        for t, p in zip(out, (SHARDED[i].up, SHARDED[i].down))
    ]
    return works, *out


@torch.library.custom_op("allie_shard::gather", mutates_args=())
def gather(
    up: torch.Tensor, down: torch.Tensor, dep: torch.Tensor, pos: int, step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer pos's whole experts from its shards up, down (dep orders the call: its input, or in backward its
    output grad), then the gather of the layer that runs next: pos + step, or pos - 1 in a checkpoint's recompute."""
    works, up, down = _ready.pop(pos, None) or _launch(pos)
    for w in works:
        w.wait()
    nxt, (lo, end) = pos + (-1 if modded_smoe.REPLAY else step), _span[pos]
    if PREFETCH and lo <= nxt < end and nxt not in _ready:
        _ready[nxt] = _launch(nxt)
    return up, down


@gather.register_fake
def _(up, down, dep, pos, step):
    return _whole(up), _whole(down)


def finish():
    """Wait for the reduce-scatters in flight and keep each shard's grad: BF16 until a second micro-batch's arrives,
    then FP32 sums (the replicated rows' copy and adds, without their memory at one micro-batch a step)."""
    while _pending:
        work, p, g, _ = _pending.pop(0)
        work.wait()
        p.part = g if p.fresh else p.part.float().add_(g)
        p.fresh = False


def reset():
    """Before the optimizer step: every grad in, no gather of the old experts left."""
    finish()
    _ready.clear()


@torch.library.custom_op("allie_shard::wgrad", mutates_args=())
def wgrad(
    grad: torch.Tensor, y: torch.Tensor, gates: torch.Tensor, order: torch.Tensor,
    offsets: torch.Tensor, dpre: torch.Tensor, x: torch.Tensor, up: torch.Tensor,
    se: torch.Tensor, pos: int, k: int,
) -> torch.Tensor:  # fmt: skip
    """modded_smoe.Routed's expert grads, reduce-scattered into layer pos's shards (in flight on return; the
    previous layer's, which overlapped this layer's backward, finish first), and its input grad."""
    m = SHARDED[pos]
    ddown = modded_smoe.down_wgrad(grad, y, gates, order, offsets)
    dup = modded_smoe.up_wgrad(dpre, x, order, offsets, k).transpose(1, 2)
    if _pending:
        finish()
    else:
        torch.autograd.Variable._execution_engine.queue_callback(finish)
    for full, p in ((dup, m.up), (ddown, m.down)):
        g = full.new_empty(p.shape)
        work = dist.reduce_scatter_tensor(g, full, op=dist.ReduceOp.AVG, async_op=True)
        _pending.append((work, p, g, full))
    return modded_smoe.input_grad(dpre, up, se, order, k)


@wgrad.register_fake
def _(grad, y, gates, order, offsets, dpre, x, up, se, pos, k):
    return torch.empty_like(x)


class Routed(torch.autograd.Function):
    """modded_smoe.Routed on layer pos's expert shards up_s, down_s."""

    @staticmethod
    def forward(ctx, x, up_s, down_s, k, se, order, offsets, gates, remat, pos):
        up, down = gather(up_s, down_s, x, pos, 1)
        pre, y = modded_smoe.up(x, up.transpose(1, 2), order, offsets, k)
        expanded = modded_smoe.scatter(y, down, se, order, y.shape[1])
        kept = [expanded] * (modded_smoe.SAVE_EXPANDED and not remat)
        w = (up_s, down_s) if remat else (up, down, *kept)
        ctx.save_for_backward(x, se, order, offsets, gates, pre, y, *w)
        ctx.k, ctx.pos, ctx.remat = k, pos, remat
        return modded_smoe.combine_op(expanded.view(*gates.shape, -1), gates)

    @staticmethod
    def backward(ctx, grad):
        x, se, order, offsets, gates, pre, y, up, down, *kept = ctx.saved_tensors
        if ctx.remat:
            up, down = gather(up, down, grad, ctx.pos, -1)
        if kept:
            dgates = (kept[0].view(*gates.shape, -1) @ grad.unsqueeze(-1)).squeeze(-1)
        else:
            dgates = modded_smoe.gates_grad(y, down, se, order, grad, ctx.k)
        del kept
        dpre = modded_smoe.dx(grad, down.permute(0, 2, 1), gates, order, offsets, pre)
        dh = wgrad(grad, y, gates, order, offsets, dpre, x, up, se, ctx.pos, ctx.k)
        return dh, None, None, None, None, None, None, dgates, None, None


def routed(x, up_s, down_s, k, se, order, offsets, gates, remat, pos):
    """modded_smoe.routed of layer pos on its expert shards up_s [E / world, 2H, D], down_s [E / world, H, D]."""
    if torch.is_grad_enabled():
        return Routed.apply(x, up_s, down_s, k, se, order, offsets, gates, remat, pos)
    up, down = gather(up_s, down_s, x, pos, 1)
    return modded_smoe.routed(x, up.transpose(1, 2), down, k, se, order, offsets, gates)


@torch.no_grad()
def state_dict(model, snapshot):
    """Rank 0's snapshot(model.state_dict()), with the sharded experts whole, one tensor at a time on the GPU
    (collective; None off rank 0)."""
    rank = dist.get_rank()
    local = {n: p for n, p in model.named_parameters() if getattr(p, "local", False)}
    state = model.state_dict() if rank == 0 else None
    if rank == 0:
        out = snapshot({k: v for k, v in state.items() if k not in local})
    for n, p in local.items():
        t = _whole(p)
        dist.all_gather_into_tensor(t, p.detach())
        if rank == 0:
            out[n] = snapshot(t)
    return {k: out[k] for k in state} if rank == 0 else None
