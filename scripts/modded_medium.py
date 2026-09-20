"""Chess data/attention/recovery adapter around the pinned medium implementation."""

from dataclasses import MISSING, asdict, dataclass, field, fields
from types import SimpleNamespace
import copy
import os

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
import modded_arch
import modded_board
import modded_diffattn
import modded_moe
import modded_medium_core as core
from modded_runtime import prime_source_key
from modded_smoe_tuned import finish_direct_accum

RUNTIME_SOURCE_KEY = prime_source_key()

BOS, MOVE_START, MOVE_END, VOCAB = 2348, 378, 2346, 2350
TIME_START, WDL_START = (
    VOCAB,
    VOCAB + 63,
)  # padded logits: 63 think-time bins, then W/D/L
COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"
MASTER_LABELS = (
    "attn",
    "mlp",
    "mlp_proj",
    "moe",
    "moe_up",
    "mlp_shared",
    "mlp_shared_up",
    "moe_sh",
    "moe_up_sh",
)  # FP32 matrices that --bf16-weights stores in BF16 (MoE experts included; the router stays FP32)
flex_kernel = torch.compile(flex_attention, dynamic=False)
BOARD = False  # set by create_model when the model has a board branch; make_context then encodes


@dataclass
class Config:
    width: int = 512
    head_dim: int = 64
    layers: int = 16
    max_tokens: int = 32768
    scheduled_steps: int = 4700
    extension_steps: int = 40
    # Global rows per optimizer step. Same 1:2:3:4 ratios as upstream.
    initial_batch_rows: int = 128
    lr_scale: float = 1.0
    clock: bool = False  # feed the mover's remaining clock (chessmix clock channel)
    elo: bool = (
        False  # feed the mover's rating bucket at every move-predicting position
    )
    feats: int = 0  # continuous clock features: 1 = own time left, 3 = + opponent's, previous think
    input_lr_mul: float = (
        75.0  # Adam lr multiplier of the clock, Elo and feature tables
    )
    value_embeds: bool = True  # model-track ablations; False zeroes the component
    skips: bool = True  # U-net skip connections and backout
    smear: bool = True  # previous-token smear gate
    doc_rope: bool = False  # rotary positions relative to each game's start
    rope_fp32: bool = False  # FP32 rotary cos/sin tables
    bf16_weights: bool = (
        False  # BF16 attention/MLP matrices, FP32 grads and master shards
    )
    ckpt: str = ""  # activation checkpointing: "mlp" | "block" (training only)
    arch: dict = field(
        default_factory=dict
    )  # model-track switches (modded_arch.DEFAULTS)


@dataclass
class Context:
    documents: torch.Tensor
    same_previous: torch.Tensor
    short_window: int
    long_window: int
    short_mask: object
    long_mask: object
    backend: str
    positions: torch.Tensor = None  # index of each token within its game
    board: torch.Tensor = (
        None  # (T, 68) board states, only for models with a board branch
    )


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


def make_context(inputs, short_window, long_window, backend="flex", host=None):
    """Inputs are complete original rows (or shorter rows for correctness tests); host is the
    same rows as a numpy array, if at hand, so board models encode without a device sync."""
    assert inputs.ndim == 2 and backend in ("flex", "dense")
    flat = inputs.flatten()
    starts = flat == BOS
    starts[:: inputs.size(1)] = True
    docs = starts.to(torch.int32).cumsum(0)
    same = ~starts
    length = flat.numel()
    blocks = []

    def make(window):
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

        # games never cross rows, so no allowed pair exceeds the window: blocks from game bounds
        if window >= inputs.size(1) - 1:
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
    idx = torch.arange(length, device=flat.device)
    positions = idx - torch.where(starts, idx, 0).cummax(0).values
    board = (
        torch.from_numpy(
            modded_board.encode(inputs.cpu().numpy() if host is None else host)
        )
        .to(flat.device)
        .flatten(0, 1)
        if BOARD
        else None
    )
    return Context(
        docs, same, short_window, long_window, short, long, backend, positions, board
    )


def attention(q, k, v, context, window, scale):
    mask = context.short_mask if window == context.short_window else context.long_mask
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if context.backend == "dense":
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    else:
        y = flex_kernel(q, k, v, block_mask=mask, scale=scale)
    return y.transpose(1, 2)


def configure(cfg, device):
    assert dist.is_initialized(), (
        "Upstream optimizer needs a process group even for one GPU"
    )
    assert cfg.layers >= 8 and cfg.width % cfg.head_dim == 0 and cfg.head_dim % 4 == 0
    assert cfg.width >= 16 and cfg.scheduled_steps > 0 and cfg.initial_batch_rows > 0
    assert dist.get_world_size() in (1, 2, 4, 8)
    core.device = torch.device(device)
    core.world_size = dist.get_world_size()
    core.grad_accum_steps = 1  # overwritten by the training harness as needed
    core.args = SimpleNamespace(
        num_layers=cfg.layers,
        num_scheduled_iterations=cfg.scheduled_steps,
        num_extension_iterations=cfg.extension_steps,
        num_iterations=cfg.scheduled_steps + cfg.extension_steps,
        train_bs_schedule=tuple(
            cfg.initial_batch_rows * 1024 * x for x in (1, 2, 3, *([4] * 9))
        ),
        train_bs_extension=cfg.initial_batch_rows * 1024 * 4,
        train_max_seq_len=1024,
        val_batch_size=cfg.max_tokens * core.world_size,
        cooldown_frac=0.70,
        split_embed_frac=2 / 3 / 4,
        block_size=128,
        ws_schedule=(3, 7, 11, 13, 15, 17, 19, 21, 23, 23, 23, 23),
        ws_final=23,
        ws_validate_post_yarn_ext=27,
    )
    core.medium_attention = attention
    core.print0 = lambda s, console=False: (
        print(s, flush=True) if dist.get_rank() == 0 else None
    )


def create_model(cfg, device="cuda"):
    configure(cfg, device)
    if torch.device(device).type == "cuda":
        # Upstream initializes autograd on the device before model/collectives.
        # Keep that warmup out of module import so CPU inspection still works.
        torch.empty(1, device=device, requires_grad=True).backward()
    arch = modded_arch.resolve(cfg.arch)
    assert cfg.ckpt in ("", "mlp", "block", "eager")
    core.CKPT = cfg.ckpt
    modded_moe.BLOCK_RECOMPUTE = cfg.ckpt == "eager"
    assert arch["moe_kernel"] != "scatter-accum" or cfg.bf16_weights, "direct expert accumulation requires BF16 masters"
    assert not (arch["moe_kernel"] == "scatter-accum" and arch["moe_shard"]), "sharded experts reduce-scatter their grads"
    global BOARD
    BOARD = bool(
        arch["board"]
    )  # per construction: no leak across models built in one process
    model = core.GPT(
        VOCAB,
        cfg.layers,
        cfg.width // cfg.head_dim,
        cfg.head_dim,
        cfg.width,
        cfg.max_tokens,
        mlp=arch["mlp"],
        untie_ve=arch["untie_ve"],
        moe=modded_arch.moe_dims(cfg.width, arch),
    ).to(device)
    model.use_clock, model.use_elo = cfg.clock, cfg.elo
    model.use_feats = cfg.feats
    model.doc_rope = cfg.doc_rope
    if cfg.rope_fp32 or arch["full_rope"] or arch["plain_init"]:
        model.yarn.fp32, model.yarn.full = cfg.rope_fp32, arch["full_rope"]
        if arch["plain_init"]:
            model.yarn.base_scale = cfg.head_dim**-0.5
        model.yarn.reset()
    model.use_x0, model.use_embed2 = arch["x0"], arch["embed2"]
    model.aux_detach = (
        VOCAB if arch["aux_detach"] else None
    )  # aux head rows start at VOCAB
    model.softcap, model.use_key_offset = arch["softcap"], arch["key_offset"]
    for block in model.blocks:
        block.attn.qk_norm, block.attn.gates = arch["qk_norm"], arch["gates"]
    core.CAUTIOUS_WD, core.NORMUON = arch["cautious_wd"], arch["normuon"]
    if arch["plain_init"]:
        with torch.no_grad():
            model.scalars[: cfg.layers] = 1.0  # residual lambdas (default 1.05)
    if arch["matrix_adam"]:
        model.matrix_adam, model.matrix_wd = arch["matrix_adam"], arch["matrix_wd"]
        for block in model.blocks:
            block.mlp.c_proj.lr_mul = 1.0
            for p in (block.attn.qkvo_w, block.mlp.c_fc, block.mlp.c_proj):
                p.fp32_state = True
    if arch["uniform_mults"]:
        model.embed2.weight.lr_mul = model.embed2.weight.wd_mul = 1.0
        model.embed.weight.wd_mul = model.lm_head.weight.wd_mul = 1.0
    model.fp32_embed = arch["fp32_embed"]
    fp32 = (
        {model.embed, model.embed2, model.lm_head, *model.value_embeds}
        if arch["fp32_embed"]
        else set()
    )
    for m in fp32:
        m.weight.fp32_state = True
    if arch["board"]:
        model.board = modded_board.build(arch["board"], cfg.width).to(device)
    if arch["diff_attn"]:
        for i, block in enumerate(model.blocks):
            block.attn.diff = modded_diffattn.DiffLambda(cfg.head_dim, i).to(device)
    model.use_value_embeds, model.use_skips, model.use_smear = (
        cfg.value_embeds,
        cfg.skips,
        cfg.smear,
    )
    for table in (model.clock_embed, model.elo_embed, model.feat_embed):
        table.weight.lr_mul = cfg.input_lr_mul
    # Follow upstream: BF16 embeddings/gates/head, FP32 attention/MLP matrices.
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)) and m not in fp32:
            m.weight.data = m.weight.data.bfloat16()
    for p in model.parameters():
        if cfg.bf16_weights and getattr(p, "label", None) in MASTER_LABELS:
            # the FP32 init seeds the optimizers' master shards (they drop it); grads summed in FP32
            p.fp32, p.data, p.master, p.main_grad = (
                p.data,
                p.data.bfloat16(),
                True,
                None,
            )
            direct = arch["moe_kernel"] == "scatter-accum" and getattr(p, "label", None) in ("moe", "moe_up")
            p.register_post_accumulate_grad_hook(finish_direct_accum if direct else core.accumulate_fp32)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
        if hasattr(p, "fp32"):
            dist.broadcast(p.fp32, 0)
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


def elo_buckets(rows):
    """Input channel aligned with rows[:, :-1]: 1 + mover's Elo // 50 (capped at 63) where the
    next token is a move, 0 elsewhere."""
    y = rows[:, 1:]
    move = (y >= MOVE_START) & (y < MOVE_END)
    return np.where(move, 1 + np.clip(ratings(rows) // 50, 0, 62), 0).astype(np.int64)


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
    def __init__(self, model, cfg):
        super().__init__(model)
        self.cfg = cfg
        self.schedule_step = 0
        # Upstream main gets this state from reset(initial_optimizer_state)
        # after its warmup. We compile while training instead of replaying a
        # warmup, so inactive Adam steps must start with communication disabled.
        self.adam_opt.should_sync = False
        self.scalar_opt.should_sync = False
        if modded_arch.resolve(cfg.arch)["adam_every"]:
            self.adam_opt.odd_step_only = self.scalar_opt.odd_step_only = False
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["initial_lr"] *= cfg.lr_scale
                group["lr"] *= cfg.lr_scale

    def advance_schedule(self, step):
        super().advance_schedule(step)
        self.schedule_step = step
        self.batch_size = core.get_bs(step)

    def rank_state_dict(self):
        # Checkpoint only at completed optimizer boundaries. Even boundaries
        # may legitimately retain gradients for the next odd Adam update.
        for opt in (self.adam_opt, self.scalar_opt):
            assert not opt._reduce_scatter_futures, (
                "Checkpoint before optimizer collectives completed"
            )
        return cpu_copy(
            dict(
                config=asdict(self.cfg),
                rank=dist.get_rank(),
                world=dist.get_world_size(),
                optimizers=[o.state_dict() for o in self.optimizers],
                optimizer_flags=[
                    dict(
                        freeze_timer=o.freeze_timer,
                        odd_step_only=o.odd_step_only,
                        should_sync=o.should_sync,
                    )
                    for o in self.optimizers
                ],
                gradients={n: p.grad for n, p in self.model.named_parameters()},
                split_embed=self.model.split_embed,
                schedule_step=self.schedule_step,
                ws_short=self.ws_short,
                ws_long=self.ws_long,
                batch_size=self.batch_size,
                yarn=dict(
                    angular_freq=self.model.yarn.angular_freq,
                    cos=self.model.yarn.cos,
                    sin=self.model.yarn.sin,
                    attn_scale=self.model.yarn.attn_scale,
                ),
            )
        )

    def load_rank_state_dict(self, saved):
        # fields added after a checkpoint was written take their defaults (e.g. bf16_weights=False)
        defaults = {
            f.name: f.default for f in fields(Config) if f.default is not MISSING
        }
        assert defaults | saved["config"] == asdict(self.cfg)
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

        for opt, state, flags in zip(
            self.optimizers, saved["optimizers"], saved["optimizer_flags"]
        ):
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
            for name, value in flags.items():
                setattr(opt, name, value)
        for name, p in self.model.named_parameters():
            g = saved["gradients"][name]
            p.grad = None if g is None else g.to(device=device, dtype=p.dtype)
        for name in ("ws_short", "ws_long", "batch_size", "schedule_step"):
            setattr(self, name, saved[name])
        self.train_loader_send_args = None
        self.mtp_weights = self.mtp_weights_schedule[self.schedule_step]
        self.model.split_embed = saved["split_embed"]
        for name in ("angular_freq", "cos", "sin"):
            getattr(self.model.yarn, name).copy_(saved["yarn"][name].to(device))
        self.model.yarn.attn_scale = saved["yarn"]["attn_scale"]


def config_dict(cfg):
    return asdict(cfg) | dict(
        upstream_commit=COMMIT,
        output_support=[MOVE_START, MOVE_END],
        attention_backend="flex",
        fp8=False,
    )
