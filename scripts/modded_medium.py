"""Chess data/attention/recovery adapter around the pinned medium implementation."""
from dataclasses import dataclass, asdict
from types import SimpleNamespace
import copy
import os

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import modded_medium_core as core
from modded_runtime import prime_source_key

RUNTIME_SOURCE_KEY = prime_source_key()

BOS, MOVE_START, MOVE_END, VOCAB = 2348, 378, 2346, 2350
COMMIT = 'ecbb586296d3dac36fd206211f25d63bad4a6b35'
flex_kernel = torch.compile(flex_attention, dynamic=False)


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


@dataclass
class Context:
    documents: torch.Tensor
    same_previous: torch.Tensor
    short_window: int
    long_window: int
    short_mask: object
    long_mask: object
    backend: str


def make_context(inputs, short_window, long_window, backend='flex'):
    """Inputs are complete original rows (or shorter rows for correctness tests)."""
    assert inputs.ndim == 2 and backend in ('flex', 'dense')
    flat = inputs.flatten()
    starts = flat == BOS
    starts[::inputs.size(1)] = True
    docs = starts.to(torch.int32).cumsum(0)
    same = ~starts
    length = flat.numel()

    def make(window):
        if backend == 'dense':
            q = torch.arange(length, device=flat.device)[:, None]
            k = torch.arange(length, device=flat.device)[None, :]
            return (q >= k) & (q - k <= window) & (docs[:, None] == docs[None, :])

        def allowed(b, h, q, k):
            return (q < length) & (k < length) & (q >= k) & (q-k <= window) & (
                docs[q.clamp(max=length-1)] == docs[k.clamp(max=length-1)])
        return create_block_mask(allowed, 1, None, length, length,
                                 device=flat.device, BLOCK_SIZE=128, _compile=True)

    short = make(short_window)
    long = short if short_window == long_window else make(long_window)
    return Context(docs, same, short_window, long_window, short, long, backend)


def attention(q, k, v, context, window, scale):
    mask = context.short_mask if window == context.short_window else context.long_mask
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if context.backend == 'dense':
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    else:
        y = flex_kernel(q, k, v, block_mask=mask, scale=scale)
    return y.transpose(1, 2)


def configure(cfg, device):
    assert dist.is_initialized(), 'Upstream optimizer needs a process group even for one GPU'
    assert cfg.layers in (8, 12, 16) and cfg.width % cfg.head_dim == 0 and cfg.head_dim % 4 == 0
    assert cfg.width >= 16 and cfg.scheduled_steps > 0 and cfg.initial_batch_rows > 0
    assert dist.get_world_size() in (1, 2, 4, 8)
    core.device = torch.device(device)
    core.world_size = dist.get_world_size()
    core.grad_accum_steps = 1  # overwritten by the training harness as needed
    core.args = SimpleNamespace(
        num_layers=cfg.layers,
        num_scheduled_iterations=cfg.scheduled_steps,
        num_extension_iterations=cfg.extension_steps,
        num_iterations=cfg.scheduled_steps+cfg.extension_steps,
        train_bs_schedule=tuple(cfg.initial_batch_rows*1024*x for x in (1, 2, 3, *([4]*9))),
        train_bs_extension=cfg.initial_batch_rows*1024*4,
        train_max_seq_len=1024,
        val_batch_size=cfg.max_tokens*core.world_size,
        cooldown_frac=.70, split_embed_frac=2/3/4,
        block_size=128, ws_schedule=(3,7,11,13,15,17,19,21,23,23,23,23),
        ws_final=23, ws_validate_post_yarn_ext=27)
    core.medium_attention = attention
    core.print0 = lambda s, console=False: print(s, flush=True) if dist.get_rank() == 0 else None


def create_model(cfg, device='cuda'):
    configure(cfg, device)
    if torch.device(device).type == 'cuda':
        # Upstream initializes autograd on the device before model/collectives.
        # Keep that warmup out of module import so CPU inspection still works.
        torch.empty(1, device=device, requires_grad=True).backward()
    model = core.GPT(VOCAB, cfg.layers, cfg.width//cfg.head_dim, cfg.head_dim,
                     cfg.width, cfg.max_tokens).to(device)
    # Follow upstream: BF16 embeddings/gates/head, FP32 attention/MLP matrices.
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()
    return model


def move_losses(logits, inputs, targets, context, weights=None):
    """Sum primary and auxiliary move NLL, with no cross-game auxiliary targets.

    Returns (objective sum, primary sum, primary count). Preserve upstream's
    summed gradient convention; the driver scales by world_size/8 before the
    optimizer's distributed average, independently of physical accumulation.
    Vocabulary padding/metadata never enter the move softmax denominator.
    """
    values = logits.reshape(-1, logits.size(-1))[:, MOVE_START:MOVE_END].float()
    y = targets.flatten()
    valid = (y >= MOVE_START) & (y < MOVE_END)
    logz = values.logsumexp(-1)
    selected = values.gather(1, (y-MOVE_START).clamp(0, MOVE_END-MOVE_START-1)[:, None])[:, 0]
    nll = logz-selected
    primary = (nll * valid).sum()
    objective = primary if weights is None else primary * weights[0]
    target_docs = context.documents + (y == BOS)
    if weights is not None:
        for offset in range(1, weights.numel()):
            aux_y = y[offset:]
            aux_valid = valid[:-offset] & valid[offset:] & (
                context.documents[:-offset] == target_docs[offset:])
            aux_selected = values[:-offset].gather(1,
                (aux_y-MOVE_START).clamp(0, MOVE_END-MOVE_START-1)[:, None])[:, 0]
            objective = objective + weights[offset] * ((logz[:-offset]-aux_selected)*aux_valid).sum()
    return objective, primary, valid.sum()


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
        for opt in self.optimizers:
            for group in opt.param_groups:
                group['initial_lr'] *= cfg.lr_scale
                group['lr'] *= cfg.lr_scale

    def advance_schedule(self, step):
        super().advance_schedule(step)
        self.schedule_step = step
        self.batch_size = core.get_bs(step)

    def rank_state_dict(self):
        # Checkpoint only at completed optimizer boundaries. Even boundaries
        # may legitimately retain gradients for the next odd Adam update.
        for opt in (self.adam_opt, self.scalar_opt):
            assert not opt._reduce_scatter_futures, 'Checkpoint before optimizer collectives completed'
        return cpu_copy(dict(
            config=asdict(self.cfg), rank=dist.get_rank(), world=dist.get_world_size(),
            optimizers=[o.state_dict() for o in self.optimizers],
            optimizer_flags=[dict(freeze_timer=o.freeze_timer, odd_step_only=o.odd_step_only,
                                  should_sync=o.should_sync) for o in self.optimizers],
            gradients={n: p.grad for n, p in self.model.named_parameters()},
            split_embed=self.model.split_embed, schedule_step=self.schedule_step,
            ws_short=self.ws_short, ws_long=self.ws_long, batch_size=self.batch_size,
            yarn=dict(angular_freq=self.model.yarn.angular_freq, cos=self.model.yarn.cos,
                      sin=self.model.yarn.sin, attn_scale=self.model.yarn.attn_scale)))

    def load_rank_state_dict(self, saved):
        assert saved['config'] == asdict(self.cfg)
        assert saved['rank'] == dist.get_rank() and saved['world'] == dist.get_world_size()
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

        for opt, state, flags in zip(self.optimizers, saved['optimizers'], saved['optimizer_flags']):
            # PyTorch's standard loader casts state tensors to parameter dtype;
            # upstream Adam deliberately stores BF16 moments for FP32 scalars.
            # Restore tensor values/dtypes explicitly after restoring groups.
            opt.load_state_dict(copy.deepcopy(state))
            for group, original in zip(opt.param_groups, state['param_groups']):
                for key, value in original.items():
                    if key == 'params':
                        continue
                    group[key] = cpu_copy(value) if key.endswith('_cpu') else to_device(value)
                for param, index in zip(group['params'], original['params']):
                    if index in state['state']:
                        opt.state[param] = to_device(state['state'][index])
            for name, value in flags.items():
                setattr(opt, name, value)
        for name, p in self.model.named_parameters():
            g = saved['gradients'][name]
            p.grad = None if g is None else g.to(device=device, dtype=p.dtype)
        for name in ('ws_short', 'ws_long', 'batch_size', 'schedule_step'):
            setattr(self, name, saved[name])
        self.train_loader_send_args = None
        self.mtp_weights = self.mtp_weights_schedule[self.schedule_step]
        self.model.split_embed = saved['split_embed']
        for name in ('angular_freq', 'cos', 'sin'):
            getattr(self.model.yarn, name).copy_(saved['yarn'][name].to(device))
        self.model.yarn.attn_scale = saved['yarn']['attn_scale']


def config_dict(cfg):
    return asdict(cfg) | dict(upstream_commit=COMMIT, output_support=[MOVE_START, MOVE_END],
                             attention_backend='flex', fp8=False)
