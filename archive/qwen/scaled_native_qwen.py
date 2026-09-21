"""Scaled scratch controls using the recorded native Qwen implementation.

This changes shape and uses unit Q/K norms. It is not a historical-run replay.
GPU training/timing/recovery must be checked before quality allocations.
"""
from dataclasses import asdict, dataclass
import functools
import os

import torch

from historical_qwen_runtime import import_original, verify_sources
from native_fp32_vector_adam import FP32VectorAdam


@dataclass(frozen=True)
class Shape:
    layers: int = 16
    width: int = 768
    ff: int = 2624
    heads: int = 6
    kv_heads: int = 3
    head_dim: int = 128
    vocab: int = 2350

    def validate(self):
        assert min(asdict(self).values()) > 0
        assert self.width == self.heads*self.head_dim
        assert self.heads % self.kv_heads == 0
        assert self.head_dim % 2 == 0 and self.vocab == 2350

    def parameter_count(self):
        self.validate()
        matrices = self.layers*(2*self.width**2 +
            2*self.width*self.kv_heads*self.head_dim + 3*self.width*self.ff)
        vectors = self.layers*(2*self.width+2*self.head_dim)+self.width
        return matrices+2*self.width*self.vocab+vectors

    def hf_config(self):
        from transformers import Qwen3Config
        self.validate()
        return Qwen3Config(vocab_size=self.vocab, hidden_size=self.width,
            intermediate_size=self.ff, num_hidden_layers=self.layers,
            num_attention_heads=self.heads, num_key_value_heads=self.kv_heads,
            head_dim=self.head_dim, rms_norm_eps=1e-6, rope_theta=1000000.,
            max_position_embeddings=1024, attention_bias=False, attention_dropout=0.,
            tie_word_embeddings=False, use_cache=False, bos_token_id=2348,
            eos_token_id=2348, pad_token_id=None)


def make_model(shape, *, device='cpu', seed=42, flash=True):
    """Construct from native classes; no downloaded weights or implicit norm state."""
    shape.validate()
    model_module, *_ = import_original()
    os.environ['FLASH_ATTEN'] = '1' if flash else '0'
    os.environ['DEVICE'] = 'cpu' if device in ('cpu','meta') else 'cuda'
    with torch.device('meta' if device == 'meta' else 'cpu'):
        model = model_module.Qwen3Model(shape.hf_config()).to(dtype=torch.bfloat16)
    if device != 'meta':
        # Rewind after construction, then use the native BF16 reset traversal.
        # Native reset omits Q/K norms, so initialize them explicitly to one.
        torch.manual_seed(seed)
        model.reset_parameters()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if '.q_norm.' in name or '.k_norm.' in name:
                    p.fill_(1.)
        model.to(device)
    assert sum(p.numel() for p in model.parameters()) == shape.parameter_count()
    return model


def build(shape, *, seed=42, lr=.0025, deterministic=True, compile_model=True):
    """Native FlashAttention, FP32 accumulation and selected FP32 vector Adam."""
    source_sha = verify_sources()
    model_module, pgm, _, _, create_optimizer, OptimizerConfig, DP, _ = import_original()
    assert torch.distributed.is_initialized()
    pgm.setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1,
        dp_size=torch.distributed.get_world_size())
    if deterministic:
        from flash_attn import flash_attn_func
        model_module.flash_attn_func = functools.partial(flash_attn_func, deterministic=True)
        import triton
        from flash_attn.ops.triton import layer_norm as norm_module
        for name in ('_layer_norm_fwd_1pass_kernel','_layer_norm_bwd_kernel'):
            tuner = getattr(norm_module, name)
            tuner.configs = [triton.Config({}, num_warps=4)]
            tuner.cache.clear()
    model = make_model(shape, device='cuda', seed=seed)
    native = create_optimizer(model, OptimizerConfig(name='muon', learning_rate=lr,
        weight_decay=.05, momentum=.9, muon_eps=1e-7, ns_steps=5,
        ns_coefficients=(3.4445,-4.7750,2.0315), adjust_lr_fn='match_rms_adamw'),
        adam_extra_kwargs=None)
    optimizer = FP32VectorAdam(native, vector_decay=0.)
    net = DP(torch.compile(model) if compile_model else model)
    metadata = dict(shape=asdict(shape), parameters=shape.parameter_count(),
        native_source_manifest_sha256=source_sha, seed=seed,
        precision='BF16 model/Muon; FP32 accumulation and normalization masters/Adam',
        initialization='Native BF16 scratch reset with explicit unit Q/K norms; no pretrained weights',
        historical_reproduction=False, deterministic=deterministic,
        compile_model=compile_model, torch=torch.__version__)
    return model, net, optimizer, metadata
