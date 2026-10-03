"""Chess data/attention/recovery adapter around the pinned medium implementation."""

import copy
import math
import os
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    _create_sparse_block_from_block_mask,
    create_block_mask,
    flex_attention,
)

from allie.model import arch as model_arch
from allie.model import attention as attn_kernels
from allie.model import board as board_cnn
from allie.model import shard as expert_shard
from allie.model import nanogpt as core
from allie.train.runtime import prime_source_key
from allie.train.schedule import Schedule

RUNTIME_SOURCE_KEY = prime_source_key()

BOS, MOVE_START, MOVE_END, VOCAB = 2348, 378, 2346, 2350
# padded logits: 63 think-time bins, then W/D/L
TIME_START, WDL_START = VOCAB, VOCAB + 63
COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"
# the matrices kept in BF16 with FP32 masters in NorMuon (MoE experts included; the router stays FP32)
MASTER_LABELS = (
    "attn",
    "mlp",
    "mlp_proj",
    "moe",
    "moe_up",
    "mlp_shared",
    "mlp_shared_up",
)
flex_kernel = torch.compile(flex_attention, dynamic=False)
# make_context's default backend; --attn-kernel: "triton" (model.attention)
ATTENTION = "flex"


@dataclass
class Config:
    width: int = 512
    head_dim: int = 64
    layers: int = 16
    max_tokens: int = 32768
    scheduled_steps: int = 4700
    lr_scale: float = 1.0
    # continuous clock features: 1 = own time left, 3 = + opponent's, previous think
    feats: int = 0
    input_lr_mul: float = 75.0  # Adam lr multiplier of the feature table
    ckpt: str = "none"  # "eager": activation checkpoints of whole blocks (training)
    ckpt_frac: float = 1.0  # checkpoint only the first ceil(frac * layers) blocks
    arch: dict = field(default_factory=dict)  # model_arch.DEFAULTS switches


@dataclass
class Context:
    documents: torch.Tensor
    same_previous: torch.Tensor
    short_window: int
    long_window: int
    short_mask: object
    long_mask: object
    backend: str
    board: torch.Tensor = None  # (T, 68) board states


def game_blocks(docs, size=128):
    """create_block_mask's (partial, full) blocks for causal same-game attention, from each block's
    first and last game (docs is nondecreasing) instead of the dense token grid."""
    n = docs.numel()
    lo = torch.arange(0, n, size, device=docs.device)
    first, last = docs[lo], docs[(lo + size - 1).clamp(max=n - 1)]
    whole = lo + size <= n
    i = torch.arange(len(lo), device=docs.device)
    below = i[:, None] > i[None, :]
    full = below & (first[None, :] == last[:, None]) & whole[:, None] & whole[None, :]
    some = (below & (last[None, :] == first[:, None])) | (i[:, None] == i[None, :])
    return (some & ~full).to(torch.int8)[None, None], full.to(torch.int8)[None, None]


def make_context(
    inputs, short_window, long_window, backend=None, host=None, board=True, span=None
):
    """Inputs are complete original rows (or shorter rows for correctness tests); host is the
    same rows as a numpy array, if at hand, so the board states encode without a device sync.
    board=False skips them (attention-only tests on synthetic rows). span: the longest game a row
    can hold (default: the row)."""
    backend = backend or ATTENTION
    assert inputs.ndim == 2 and backend in ("flex", "dense", "triton")
    flat = inputs.flatten()
    starts = flat == BOS
    starts[:: inputs.size(1)] = True
    docs = starts.to(torch.int32).cumsum(0)
    same = ~starts
    length = flat.numel()
    blocks = []

    def make(window):
        if backend == "triton":
            return attn_kernels.bounds(starts, inputs.size(1), window)
        if backend == "dense":
            q = torch.arange(length, device=flat.device)[:, None]
            k = torch.arange(length, device=flat.device)[None, :]
            return (q >= k) & (q - k <= window) & (docs[:, None] == docs[None, :])

        def allowed(b, h, q, k):
            return (
                (q < length)
                & (k < length)
                & (q >= k)
                & (q - k <= window)
                & (docs[q.clamp(max=length - 1)] == docs[k.clamp(max=length - 1)])
            )

        # no game exceeds span tokens, so no allowed pair exceeds the window: blocks from game bounds
        if window >= min(inputs.size(1), span or inputs.size(1)) - 1:
            if not blocks:
                blocks.extend(game_blocks(docs))
            return _create_sparse_block_from_block_mask(
                tuple(blocks), allowed, (length, length), 128, 128
            )
        return create_block_mask(
            allowed,
            1,
            None,
            length,
            length,
            device=flat.device,
            BLOCK_SIZE=128,
            _compile=True,
        )

    short = make(short_window)
    long = short if short_window == long_window else make(long_window)
    states = None
    if board:
        states = torch.from_numpy(
            board_cnn.encode(inputs.cpu().numpy() if host is None else host)
        )
        if flat.is_cuda:
            states = states.pin_memory()
        states = states.to(flat.device, non_blocking=True).flatten(0, 1)
    return Context(docs, same, short_window, long_window, short, long, backend, states)


def attention(q, k, v, context, window, scale, gate=None):
    mask = context.short_mask if window == context.short_window else context.long_mask
    if context.backend == "triton":
        return attn_kernels.attention(q, k, v, *mask, scale, gate)
    assert gate is None
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if context.backend == "dense":
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    else:
        y = flex_kernel(q, k, v, block_mask=mask, scale=scale)
    return y.transpose(1, 2)


def attention_qkv(qkv, context, window, scale, cos, sin, gate, ve, vgate):
    """attention() of the raw projection qkv, with the model's q/k norm and rotary, value
    embeddings and output gate inside the Triton kernels (attn_kernels.attention_qkv)."""
    assert context.backend == "triton"
    mask = context.short_mask if window == context.short_window else context.long_mask
    return attn_kernels.attention_qkv(qkv, cos, sin, *mask, scale, gate, ve, vgate)


def configure(cfg, device):
    assert dist.is_initialized(), (
        "Upstream optimizer needs a process group even for one GPU"
    )
    assert cfg.layers >= 8 and cfg.width % cfg.head_dim == 0 and cfg.head_dim % 4 == 0
    assert cfg.width >= 16 and cfg.scheduled_steps > 0
    assert dist.get_world_size() in (1, 2, 4, 8)
    assert cfg.ckpt in ("none", "eager")
    core.device = torch.device(device)
    core.args = SimpleNamespace(
        num_layers=cfg.layers, num_iterations=cfg.scheduled_steps
    )
    core.medium_attention = attention
    core.medium_attention_qkv = attention_qkv
    core.CKPT = cfg.ckpt
    core.CKPT_LAYERS = math.ceil(cfg.ckpt_frac * cfg.layers)
    core.DEFER = cfg.ckpt == "eager"


def build(cfg, shard):
    """The seeded FP32 init on the host, every rank the same; shard: this rank keeps its experts' slice."""
    model = core.GPT(
        VOCAB,
        cfg.layers,
        cfg.width // cfg.head_dim,
        cfg.head_dim,
        cfg.width,
        cfg.max_tokens,
        moe=moe_layers(cfg),
    )
    if shard:
        expert_shard.shard([m for m in model.modules() if isinstance(m, core.MoE)])
    return model


def moe_layers(cfg):
    """Per block: its MoE constructor arguments (model_arch.moe_dims), None for a dense MLP."""
    dense = model_arch.resolve(cfg.arch)["moe_dense_first"]
    if not model_arch.moe_dims(cfg.width, cfg.arch):
        return [None] * cfg.layers
    return [None] * dense + [
        model_arch.moe_dims(cfg.width, cfg.arch, j) for j in range(cfg.layers - dense)
    ]


def create_model(cfg, device="cuda"):
    configure(cfg, device)
    if torch.device(device).type == "cuda":
        # Upstream initializes autograd on the device before model/collectives.
        # Keep that warmup out of module import so CPU inspection still works.
        torch.empty(1, device=device, requires_grad=True).backward()
    world = dist.get_world_size()
    shard = model_arch.resolve(cfg.arch)["moe_shard"] and world > 1
    # sharded, two ranks at a time: the node holds two whole FP32 inits (~40-53 GB each at width 2048), not eight
    turns = world // 2 if shard else 1
    for turn in range(turns):
        if dist.get_rank() * turns // world == turn:
            model = build(cfg, shard).to(device)
        if turns > 1:
            dist.barrier()
    for i, block in enumerate(model.blocks):
        for p in block.parameters():
            p.block = i
    model.use_feats = cfg.feats
    model.use_x0 = model_arch.resolve(cfg.arch)["x0"]
    model.feat_embed.weight.lr_mul = cfg.input_lr_mul
    model.board = board_cnn.build(cfg.width).to(device)
    arch = model_arch.resolve(cfg.arch)
    model.header_feats = arch["header_feats"]
    model.tc_header = arch["tc_header"]
    if model.header_feats:  # zero init, no RNG draw: every other init is unchanged
        w = torch.zeros(128, cfg.width, device=device)
        model.header_embed = torch.nn.Embedding(128, cfg.width, _weight=w)
        w = model.header_embed.weight
        lr = arch["header_lr_mul"]
        w.label, w.wd_mul = "embed2", 5.0
        w.lr_mul = cfg.input_lr_mul if lr is None else lr
    # Follow upstream: BF16 embeddings/gates/head; the board stays FP32
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    for p in model.parameters():
        if not getattr(p, "local", False):
            dist.broadcast(p.detach(), 0)
    small = model_arch.resolve(cfg.arch)["fp32_small_masters"]
    masters = MASTER_LABELS + ("attn_gate", "value_embed_gate") * small
    for p in model.parameters():
        if getattr(p, "label", None) in masters:
            # the FP32 init seeds NorMuon's master shards (it drops it) from host memory, so the
            # GPU never holds the FP32 model next to the BF16 one
            p.fp32, p.master, p.main_grad = p.data.float().cpu(), True, None
            p.data = p.data.bfloat16()
        elif small and p.dtype == torch.bfloat16:  # DistAdam: FP32 moments, master
            p.fp32_state = p.adam_master = True
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return model


def ratings(rows):
    """Rating of whoever plays each target token (rows[:, 1:]), parsed from the game's header."""
    n, length = rows.shape
    pos = np.arange(length)[None, :]
    starts = np.maximum.accumulate(np.where(rows == BOS, pos, 0), axis=1)
    ar = np.arange(n)[:, None]
    digits = lambda first: sum(
        rows[ar, np.minimum(starts + k, length - 1)] * p
        for k, p in zip(range(first, first + 4), (1000, 100, 10, 1))
    )
    return np.where((pos - starts - 11) % 2 == 0, digits(3), digits(7))[:, 1:]


def move_losses(logits, inputs, targets, context, weights=None, mask=None):
    """Sum primary and auxiliary move NLL, with no cross-game auxiliary targets.

    Returns (objective sum, primary sum, primary count). Preserve upstream's
    summed gradient convention; the driver scales by world_size/8 before the
    optimizer's distributed average, independently of physical accumulation.
    Vocabulary padding/metadata never enter the move softmax denominator.
    """
    values = logits.reshape(-1, logits.size(-1))[:, MOVE_START:MOVE_END].float()
    y = targets.flatten()
    valid = (y >= MOVE_START) & (y < MOVE_END)
    if mask is not None:
        valid = valid & mask.flatten()
    logz = values.logsumexp(-1)
    selected = values.gather(
        1, (y - MOVE_START).clamp(0, MOVE_END - MOVE_START - 1)[:, None]
    )[:, 0]
    nll = logz - selected
    primary = (nll * valid).sum()
    objective = primary if weights is None else primary * weights[0]
    target_docs = context.documents + (y == BOS)
    if weights is not None:
        for offset in range(1, weights.numel()):
            aux_y = y[offset:]
            aux_valid = (
                valid[:-offset]
                & valid[offset:]
                & (context.documents[:-offset] == target_docs[offset:])
            )
            aux_selected = values[:-offset].gather(
                1, (aux_y - MOVE_START).clamp(0, MOVE_END - MOVE_START - 1)[:, None]
            )[:, 0]
            objective = (
                objective
                + weights[offset] * ((logz[:-offset] - aux_selected) * aux_valid).sum()
            )
    return objective, primary, valid.sum()


def aux_losses(logits, time, wdl, mask):
    """Summed think-time and outcome NLL on their padded logits: [time, count, wdl, count]."""
    z, out = logits.reshape(-1, logits.size(-1)), []
    for y, lo, k in ((time, TIME_START, 63), (wdl, WDL_START, 3)):
        ok = (y >= 0) & mask
        v = z[:, lo : lo + k].float()
        nll = v.logsumexp(-1) - v.gather(1, y.clamp(min=0)[:, None])[:, 0]
        out += [(nll * ok).sum(), ok.sum()]
    return out


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


class TrainingManager(core.TrainingManager):
    def __init__(self, model, cfg, schedule=None, decay_start=-1, end_step=None):
        super().__init__(model, schedule or Schedule(), decay_start, end_step)
        self.cfg = cfg
        self.schedule_step = 0
        arch = model_arch.resolve(cfg.arch)
        self.adam_every = arch["adam_every"]
        for opt in self.optimizers:
            half = self.adam_every and not isinstance(opt, core.NorMuon)
            for group in opt.param_groups:
                group["initial_lr"] *= cfg.lr_scale
                group["lr"] *= cfg.lr_scale
                if half:  # the same per-token lr, lr^2 decay and moment horizons
                    group["initial_lr"] *= 0.5
                    group["lr"] *= 0.5
                    group["weight_decay"] *= 2
                    group["betas"] = tuple(b**0.5 for b in group["betas"])
                if arch["wd_scale"] != 1:
                    group["weight_decay"] *= arch["wd_scale"]

    def advance_schedule(self, step):
        super().advance_schedule(step)
        self.schedule_step = step

    def rank_state_dict(self, snapshot=cpu_copy):
        # Checkpoint only at completed optimizer boundaries. Even boundaries
        # may legitimately retain gradients for the next odd Adam update.
        for opt in (self.adam_opt, self.scalar_opt):
            assert not opt._reduce_scatter_futures, (
                "Checkpoint before optimizer collectives completed"
            )
        return snapshot(
            dict(
                config=asdict(self.cfg),
                rank=dist.get_rank(),
                world=dist.get_world_size(),
                optimizers=[o.state_dict() for o in self.optimizers],
                gradients={n: p.grad for n, p in self.model.named_parameters()},
                split_embed=self.model.split_embed,
                schedule_step=self.schedule_step,
            )
        )

    def load_rank_state_dict(self, saved):
        assert saved["config"] == asdict(self.cfg)
        assert (
            saved["rank"] == dist.get_rank() and saved["world"] == dist.get_world_size()
        )
        device = next(self.model.parameters()).device

        def to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=device)  # preserve the actual saved dtype
            if isinstance(value, dict):
                return {k: to_device(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_device(v) for v in value]
            if isinstance(value, tuple):
                return tuple(to_device(v) for v in value)
            return copy.deepcopy(value)

        for opt, state in zip(self.optimizers, saved["optimizers"]):
            # PyTorch's standard loader casts state tensors to parameter dtype;
            # upstream Adam deliberately stores BF16 moments for FP32 scalars.
            # Restore tensor values/dtypes explicitly after restoring groups.
            opt.load_state_dict(copy.deepcopy(state))
            for group, original in zip(opt.param_groups, state["param_groups"]):
                for key, value in original.items():
                    if key == "params":
                        continue
                    group[key] = (
                        cpu_copy(value) if key.endswith("_cpu") else to_device(value)
                    )
                for param, index in zip(group["params"], original["params"]):
                    if index in state["state"]:
                        opt.state[param] = to_device(state["state"][index])
        for name, p in self.model.named_parameters():
            g = saved["gradients"][name]
            p.grad = None if g is None else g.to(device=device, dtype=p.dtype)
        self.schedule_step = saved["schedule_step"]
        self.mtp_weights = self.mtp_weights_schedule[self.schedule_step]
        self.model.split_embed = saved["split_embed"]


def config_dict(cfg):
    return asdict(cfg) | dict(
        upstream_commit=COMMIT,
        output_support=[MOVE_START, MOVE_END],
        attention_backend=ATTENTION,
    )
