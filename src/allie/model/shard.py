"""Expert sharding (arch moe_shard), ZeRO-3 style: each rank keeps experts // world whole routed experts of every
MoE layer (their BF16 rows, FP32 masters and NorMuon state: model.nanogpt.NorMuon local).

gather() all-gathers a layer's BF16 experts where its forward needs them and launches the next layer's gather
(prefetch). A layer under an eager block checkpoint (model.moe.MoE.remat False) saves its gathered experts for
backward: the checkpoint drops them in the forward and its recompute holds them until the block's own backward.
Any other layer saves its shards and gathers again in backward. The backward computes the full expert grads and
reduce-scatters them (BF16, averaged, in flight until the next layer's backward) to the shards (finish()). model.pt
keeps whole experts: state_dict() gathers them, load_state_dict slices them (same world size).
Schedule knobs (the same collectives on the same data, bitwise): DEPTH layers gathered ahead in forward, start() and
backward_start() launch a pass's first gather early, HOLD keeps a forward's gather for the backward outside checkpoints.
HOST (--shard-host-gather, bitwise): no all-gathers. Every rank publishes its shards after the optimizer step into one
node-shared pinned host image (D2H, 0.1 GB per layer per rank) and a gather copies the whole experts from the image
(H2D, its own rows D2D) on the copy engines: no NCCL kernel and no SM, faster than the all-gather and unslowed by
concurrent compute.
"""

import mmap
import os
import random
import time
import uuid

import torch
import torch.distributed as dist
from torch import nn

from allie.model import moe_kernels

SHARDED = []  # every sharded MoE layer, each model's in forward order
PREFETCH = True
# the backward's next gather from the end of a layer's expert grads, not its start (one gathered layer less at the
# backward's peak, less overlap; train.trainer --shard-late-prefetch)
LATE = False
DEPTH = 1  # layers the forward gathers ahead (train.trainer --shard-prefetch)
# layers out of a checkpoint save their forward's gathered experts, not their shards: no gather in their backward
# (train.trainer --shard-hold; 0.8 GB per such layer at width 2048, held from its forward to its backward)
HOLD = False
HOST = False  # gathers from the host image (host_setup, publish) instead of all-gathers
_span = []  # per layer: its model's first and end positions
_ready = {}  # layer position -> (works, up, down): gathers in flight (works: c10d Works, or the copy's event)
_pending = []  # (work, shard, its grad, the whole grads it reads): reduce-scatters in flight
_image = None  # per layer: (up, down) views of the pinned host image, whole experts
_events = []  # [rank][layer]: the publish events (interprocess) a gather of the layer waits on
_pub = _copy = None  # streams: this rank's publishes (D2H), the gathers (H2D)
_group = None  # gloo group: the host barrier after every rank's publish
# test scaffolding: a random host delay up to this many ms before each host gather's enqueue (per rank, per gather), to
# skew the ranks' hosts against the publish events' generations (ALLIE_SHARD_STRESS_MS; 0: off)
_STRESS_MS = float(os.environ.get("ALLIE_SHARD_STRESS_MS", 0))


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


def _rows(p):
    """This rank's rows of a layer's whole experts."""
    n, r = p.shape[0], dist.get_rank()
    return r * n, (r + 1) * n


def _launch(i):
    m = SHARDED[i]
    out = [_whole(p) for p in (m.up, m.down)]
    if not HOST:
        works = [
            dist.all_gather_into_tensor(t, p.detach(), async_op=True)
            for t, p in zip(out, (m.up, m.down))
        ]
        return works, *out
    if _STRESS_MS:
        time.sleep(random.random() * _STRESS_MS / 1e3)
    done = torch.cuda.Event()
    # the whole buffers come from the current stream's pool: their blocks may still be read by its queued kernels, so
    # the copies wait for it (as c10d's collectives do), then for every rank's publish of this layer
    _copy.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_copy):
        for events in _events:
            _copy.wait_event(events[i])
        for t, h, p in zip(out, _image[i], (m.up, m.down)):
            lo, hi = _rows(p)
            t[:lo].copy_(h[:lo], non_blocking=True)
            t[lo:hi].copy_(p.detach(), non_blocking=True)
            t[hi:].copy_(h[hi:], non_blocking=True)
        done.record(_copy)
    return [done], *out


def host_setup(group):
    """Map the node-shared image of every sharded layer's whole experts (/dev/shm, unlinked once every rank has mapped
    it), register it with CUDA, exchange the publish events. group: a gloo group of the ranks, all on this host."""
    global _image, _pub, _copy, _group
    w, r = dist.get_world_size(), dist.get_rank()
    hosts = [None] * w
    dist.all_gather_object(hosts, os.uname().nodename, group=group)
    assert len(set(hosts)) == 1, f"--shard-host-gather needs one host, got {sorted(set(hosts))}"
    shapes = [[(m.experts, *p.shape[1:], p.dtype) for p in (m.up, m.down)] for m in SHARDED]
    size = lambda s: int(torch.Size(s[:-1]).numel()) * torch.empty(0, dtype=s[-1]).element_size()
    nbytes = sum(size(s) for pair in shapes for s in pair)
    st = os.statvfs("/dev/shm")
    assert st.f_bavail * st.f_frsize > nbytes + (1 << 30), f"/dev/shm: {nbytes >> 30} GiB needed"
    name = [f"/dev/shm/allie-shard-{uuid.uuid4().hex}" if r == 0 else None]
    dist.broadcast_object_list(name, 0, group=group)
    if r == 0:
        with open(name[0], "wb") as f:
            f.truncate(nbytes)
    dist.barrier(group=group)
    with open(name[0], "r+b") as f:
        mapped = mmap.mmap(f.fileno(), nbytes)
    dist.barrier(group=group)
    if r == 0:
        os.unlink(name[0])
    flat = torch.frombuffer(mapped, dtype=torch.uint8)
    rc = torch.cuda.cudart().cudaHostRegister(flat.data_ptr(), nbytes, 0)
    assert rc == 0 and flat.is_pinned(), f"cudaHostRegister {rc}"
    _image, at = [], 0
    for pair in shapes:
        views = []
        for s in pair:
            n = size(s)
            views.append(flat[at : at + n].view(s[-1]).view(s[:-1]))
            at += n
        _image.append(tuple(views))
    mine = [torch.cuda.Event(interprocess=True) for _ in SHARDED]
    handles = [None] * w
    dist.all_gather_object(handles, [e.ipc_handle() for e in mine], group=group)
    device = torch.cuda.current_device()
    _events[:] = [
        mine if k == r else [torch.cuda.Event.from_ipc_handle(device, h) for h in hs]
        for k, hs in enumerate(handles)
    ]
    _pub, _copy, _group = torch.cuda.Stream(), torch.cuda.Stream(), group


def publish():
    """After the optimizer step (and once at the start): this rank's shards into the image. A host barrier first: every
    rank has enqueued the last step's gathers, whose waits took the events' last records (a wait enqueued after a record
    takes that record: a late rank's step-s gather must not wait on the step-s+1 publish, which needs its backward).
    Then the per-layer publish events in forward order, then a host barrier: no rank's gather waits on an event not yet
    recorded."""
    dist.barrier(group=_group)
    _pub.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_pub):
        for i, m in enumerate(SHARDED):
            for h, p in zip(_image[i], (m.up, m.down)):
                lo, hi = _rows(p)
                h[lo:hi].copy_(p.detach(), non_blocking=True)
            _events[dist.get_rank()][i].record(_pub)
    dist.barrier(group=_group)


@torch.library.custom_op("allie_shard::gather", mutates_args=())
def gather(
    up: torch.Tensor, down: torch.Tensor, dep: torch.Tensor, pos: int, step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer pos's whole experts from its shards up, down (dep orders the call: its input, or in backward its
    output grad), then the gather of the layer that runs next: pos + step, or pos - 1 in a checkpoint's recompute."""
    works, up, down = _ready.pop(pos, None) or _launch(pos)
    for w in works:
        w.wait()
    step = -1 if moe_kernels.REPLAY else step
    if not (LATE and step < 0):
        for d in range(1, (DEPTH if step > 0 else 1) + 1):
            prefetch(pos + d * step, pos)
    return up, down


def prefetch(nxt, pos):
    """Launch layer nxt's gather for the layer pos's pass reaches next; never in backward for a layer HOLD kept."""
    lo, end = _span[pos]
    held = nxt < pos and HOLD and SHARDED[nxt].remat
    if PREFETCH and lo <= nxt < end and nxt not in _ready and not held:
        _ready[nxt] = _launch(nxt)


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
    """Before the optimizer step: every grad in, no gather of the old experts left (none in flight), the last publish
    done reading them."""
    finish()
    for works, *_ in _ready.values():
        for w in works:
            w.wait()
    _ready.clear()
    if _pub is not None:
        torch.cuda.current_stream().wait_stream(_pub)


def start(layers):
    """After the optimizer step: the next forward's first gathers (layers: a model's MoE layers in forward order)."""
    for m in layers[:DEPTH]:
        prefetch(m.pos, m.pos)


def backward_start(layers):
    """Before a backward: the gather of its first layer that gathers (a checkpoint's recompute, or a layer's own
    backward unless HOLD kept its experts)."""
    for m in reversed(layers):
        if not (m.remat and HOLD):
            prefetch(m.pos, m.pos)
            break


@torch.library.custom_op("allie_shard::wgrad", mutates_args=())
def wgrad(
    grad: torch.Tensor, y: torch.Tensor, gates: torch.Tensor, order: torch.Tensor,
    offsets: torch.Tensor, dpre: torch.Tensor, x: torch.Tensor, up: torch.Tensor,
    se: torch.Tensor, pos: int, k: int,
) -> torch.Tensor:  # fmt: skip
    """moe_kernels.Routed's expert grads, reduce-scattered into layer pos's shards (in flight on return; the
    previous layer's, which overlapped this layer's backward, finish first), and its input grad."""
    m = SHARDED[pos]
    ddown = moe_kernels.down_wgrad(grad, y, gates, order, offsets)
    dup = moe_kernels.up_wgrad(dpre, x, order, offsets, k).transpose(1, 2)
    if _pending:
        finish()
    else:
        torch.autograd.Variable._execution_engine.queue_callback(finish)
    for full, p in ((dup, m.up), (ddown, m.down)):
        g = full.new_empty(p.shape)
        work = dist.reduce_scatter_tensor(g, full, op=dist.ReduceOp.AVG, async_op=True)
        _pending.append((work, p, g, full))
    dh = moe_kernels.input_grad(dpre, up, se, order, k)
    if LATE:
        prefetch(pos - 1, pos)
    return dh


@wgrad.register_fake
def _(grad, y, gates, order, offsets, dpre, x, up, se, pos, k):
    return torch.empty_like(x)


class Routed(torch.autograd.Function):
    """moe_kernels.Routed on layer pos's expert shards up_s, down_s."""

    @staticmethod
    def forward(ctx, x, up_s, down_s, k, se, order, offsets, gates, remat, pos):
        up, down = gather(up_s, down_s, x, pos, 1)
        pre, y = moe_kernels.up(x, up.transpose(1, 2), order, offsets, k)
        saved = x, se, order, offsets, gates, pre, y
        ctx.k, ctx.pos, ctx.remat = k, pos, remat and not HOLD
        if remat or not moe_kernels.SAVE_EXPANDED:
            ctx.save_for_backward(*saved, *((up_s, down_s) if ctx.remat else (up, down)))
            return moe_kernels.routed_out(y, down, se, order, gates)
        expanded = moe_kernels.scatter(y, down, se, order, y.shape[1])
        ctx.save_for_backward(*saved, up, down, expanded)
        return moe_kernels.combine_op(expanded.view(*gates.shape, -1), gates)

    @staticmethod
    def backward(ctx, grad):
        x, se, order, offsets, gates, pre, y, up, down, *kept = ctx.saved_tensors
        if ctx.remat:
            up, down = gather(up, down, grad, ctx.pos, -1)
        if kept:
            dgates = (kept[0].view(*gates.shape, -1) @ grad.unsqueeze(-1)).squeeze(-1)
        else:
            dgates = moe_kernels.gates_grad(y, down, se, order, grad, ctx.k)
        del kept
        dpre = moe_kernels.dx(grad, down.permute(0, 2, 1), gates, order, offsets, pre)
        dh = wgrad(grad, y, gates, order, offsets, dpre, x, up, se, ctx.pos, ctx.k)
        return dh, None, None, None, None, None, None, dgates, None, None


def routed(x, up_s, down_s, k, se, order, offsets, gates, remat, pos):
    """moe_kernels.routed of layer pos on its expert shards up_s [E / world, 2H, D], down_s [E / world, H, D]."""
    if torch.is_grad_enabled():
        return Routed.apply(x, up_s, down_s, k, se, order, offsets, gates, remat, pos)
    up, down = gather(up_s, down_s, x, pos, 1)
    return moe_kernels.routed(x, up.transpose(1, 2), down, k, se, order, offsets, gates)


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
