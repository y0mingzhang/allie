"""Original Picotron model/optimizer with explicit recovery-runtime adaptations.

The only loaded weights are the pinned upstream Q/K normalization subset for
reference reproduction. Never loads the fine-tuned chess reference's weights.
"""
import functools
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import AutoConfig

ROOT = Path(os.environ.get('ALLIE_PROJECT_ROOT', '/home/yimingz3/src/allie'))
BASE = ROOT/'results/recipe10x/qwen-reproduction'
SOURCE = BASE/'source-v1'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_sources():
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        assert digest(SOURCE/name) == sha, name
    return digest(SOURCE/'manifest.json')


def import_original():
    os.environ.update(FLASH_ATTEN='1', DTYPE='bfloat16', DEVICE='cuda', CONTEXT_PARALLEL='0')
    sys.path.insert(0, str(SOURCE/'picotron'))
    import picotron.model as model_module
    import picotron.process_group_manager as pgm
    from picotron.checkpoint import InitializationManager, init_model_with_dematerialized_weights
    from picotron.optim import create_optimizer, OptimizerConfig
    from picotron.data_parallel.data_parallel import DataParallelBucket
    # Extract the unmodified scheduler class without importing train.py's W&B
    # logging dependency. The entire original source is verified above.
    tree = ast.parse((SOURCE/'picotron/train.py').read_text())
    scheduler_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'WarmupStableDecayScheduler']
    assert len(scheduler_nodes) == 1
    namespace = {'math':math}
    exec(compile(ast.Module(body=scheduler_nodes, type_ignores=[]), str(SOURCE/'picotron/train.py'), 'exec'), namespace)
    from picotron.utils import set_all_seed
    train_module = SimpleNamespace(set_all_seed=set_all_seed,
        WarmupStableDecayScheduler=namespace['WarmupStableDecayScheduler'])
    return model_module, pgm, InitializationManager, init_model_with_dematerialized_weights, create_optimizer, OptimizerConfig, DataParallelBucket, train_module


def build(seed=42, compile_model=True, deterministic_backward=True):
    source_sha = verify_sources()
    model_module, pgm, Mapper, empty_init, create_optimizer, OptimizerConfig, DP, train_module = import_original()
    pgm.setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=torch.distributed.get_world_size())
    if deterministic_backward:
        # Only backward determinism changes; original inference remains unchanged.
        from flash_attn import flash_attn_func
        model_module.flash_attn_func = functools.partial(flash_attn_func, deterministic=True)
        # The historical norm autotuners do not key on row count. An initial
        #56-row validation forward versus a14-row training forward can therefore
        #choose different launch configurations in fresh processes. Pin the
        #original kernels' warp count; no kernel equations are rewritten.
        import triton
        from flash_attn.ops.triton import layer_norm as norm_module
        for name in ('_layer_norm_fwd_1pass_kernel', '_layer_norm_bwd_kernel'):
            tuner = getattr(norm_module, name)
            tuner.configs = [triton.Config({}, num_warps=4)]
            tuner.cache.clear()
    config_path = ROOT/'results/final-training/reference-model-identity.json'
    reference = json.loads(config_path.read_text())['reference_models']['qwen']
    reference_path = Path(reference['path'])
    assert digest(reference_path/'config.json') == reference['files_sha256']['config.json']
    cfg = AutoConfig.from_pretrained(reference_path, local_files_only=True)
    norms_path = ROOT/'results/recipe10x/qwen-norm-provenance/upstream-qk-norms.safetensors'
    norm_sha = digest(norms_path)
    assert norm_sha == 'a152add83f4dd76abead606356f5485ef599986cb5ba617a1ceeaecc3d4102cd'
    train_module.set_all_seed(seed)
    with empty_init():
        model = model_module.Qwen3Model(cfg)
    mapper = Mapper(model, cfg)
    # Historical loader materializes upstream BF16 tensors on CPU, then resets
    # everything except Q/K norms. Allocate the overwritten tensors directly:
    # unlike loading the full upstream checkpoint this needs no discarded weights.
    state = {k:torch.empty(p.shape, dtype=torch.bfloat16, device='cpu') for k,p in model.named_parameters()}
    norm_state = {mapper.convert_safetensors_to_hf_name(k):v for k,v in load_file(norms_path).items()}
    assert len(norm_state) == 56 and sum(v.numel() for v in norm_state.values()) == 7168
    assert set(norm_state) == {k for k in state if '.q_norm.' in k or '.k_norm.' in k}
    state.update(norm_state)
    expected_norms = {k:v.clone() for k,v in norm_state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    # These are exactly the recorded reset methods and module traversal order.
    model.reset_parameters()
    for k, v in expected_norms.items():
        assert torch.equal(model.state_dict()[k], v)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 1419035648
    del state, norm_state, expected_norms, mapper
    model.to('cuda').train()
    training = json.loads((ROOT/'results/recipe10x/qwen-wandb-audit/prefix-config.json').read_text())['training']['value']
    oc = training['optimizer']
    optimizer = create_optimizer(model, OptimizerConfig(name='muon', learning_rate=.005,
        weight_decay=oc['weight_decay'], momentum=oc['momentum'], muon_eps=oc['muon_eps'],
        ns_steps=oc['ns_steps'], ns_coefficients=tuple(oc['ns_coefficients']),
        adjust_lr_fn=oc['adjust_lr_fn']), adam_extra_kwargs=None)
    # In recorded train.py, device is torch.device('cuda', rank), so its
    # `device == 'cuda'` fused-Adam test is false. Preserve the resulting default.
    scheduler = train_module.WarmupStableDecayScheduler(optimizer, .005, training['lr_schedule'])
    net = torch.compile(model) if compile_model else model
    # Also wrap world1 so FP32 microbatch accumulation matches the multi-GPU
    # bucket path, rather than accumulating directly into BF16 parameter grads.
    net = DP(net)
    metadata = dict(source_manifest_sha256=source_sha, upstream_norm_sha256=norm_sha,
        reference_config_sha256=reference['files_sha256']['config.json'], seed=seed,
        parameters=1419035648, precision='BF16 parameters/momentum; FP32 gradient buckets',
        initialization='Original CPU BF16 reset; pinned upstream Q/K norms retained; no chess weights',
        compile_model=compile_model, deterministic_backward=deterministic_backward,
        norm_kernel_warps=4 if deterministic_backward else 'autotuned',
        norm_adam_fused=False, prefix_schedule=training['lr_schedule'],
        initialization_limit='Original upstream revision is not cryptographically bound to historical run.',
        runtime_deviations=['L40S hardware and smaller accumulated microbatches',
                            'Python3.12/Transformers4.57.1',
                            'Deterministic FlashAttention backward when enabled',
                            'Full optimizer/RNG/data checkpoint recovery added'])
    return model, net, optimizer, scheduler, metadata
