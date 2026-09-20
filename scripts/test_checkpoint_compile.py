"""Real compiled GPT checkpoint vs retain-activation gradient/provenance gate.

Run using torchrun (one or two GPUs). No optimizer or mocked attention/expert
kernels. The reference keeps the SAME per-block compiled graphs without
checkpointing; this isolates recomputation from eager/compiled rounding changes.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from modded_medium import Config, TrainingManager, create_model, core, make_context
from modded_wsd import Schedule, install


@torch.compiler.disable
def retain(module, *args):
    if not hasattr(module, '_compiled'):
        module._compiled = torch.compile(module._forward, dynamic=False, fullgraph=True)
    return module._compiled(*args)


def run(checkpointed, root):
    original = core._eager_checkpoint
    if not checkpointed:
        core._eager_checkpoint = retain
    torch.manual_seed(701)
    cfg = Config(width=128, head_dim=64, layers=8, max_tokens=1024,
                 scheduled_steps=8, extension_steps=0, initial_batch_rows=8,
                 bf16_weights=True, ckpt='eager',
                 arch=dict(mlp='swiglu', moe=[16, 2], moe_seq=.001, moe_kernel='scatter'))
    m = create_model(cfg, device=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith(('down', 'shared_down', 'c_proj')):
                p.normal_(0, .01)
    mgr = TrainingManager(m, cfg)
    mgr.advance_schedule(0)
    net = torch.compile(m, dynamic=False, fullgraph=False)
    def step():
        for p in m.parameters():
            p.grad = None
            if hasattr(p, 'main_grad'):
                p.fresh = True
        g = torch.Generator(device='cuda').manual_seed(410 + dist.get_rank())
        rows = torch.randint(378, 2346, (1, 1025), generator=g, device='cuda')
        x, y = rows[:, :-1], rows[:, 1:]
        ctx = make_context(x, mgr.ws_short * 128, mgr.ws_long * 128)
        z = net(x.flatten(), y.flatten(), ctx, mgr.get_forward_args())
        z.float().square().mean().backward()
        return z
    for _ in range(2):
        step()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = step()
        torch.cuda.synchronize()
    proof = {'output': out.detach().cpu()}
    for name, p in m.named_parameters():
        grad = p.main_grad if hasattr(p, 'main_grad') else p.grad
        if grad is not None:
            proof[name] = grad.detach().cpu().clone()
    region_count = sum(e.count for e in prof.key_averages() if e.key == 'CompiledFunction')
    assert region_count > 0, 'no compiled autograd child observed'
    prof.export_chrome_trace(str(root / f'checkpoint-{checkpointed}-rank{dist.get_rank()}.json'))
    print(json.dumps(dict(checkpointed=checkpointed, rank=dist.get_rank(),
                          compiled_autograd_calls=region_count)), flush=True)
    core._eager_checkpoint = original
    del net, m, mgr, out
    gc.collect()
    torch.cuda.empty_cache()
    return proof


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    install(core, Schedule(warmup_steps=2, mtp_steps=0, split_step=5, batch_rows=8), -1, 8)
    ref = run(False, root)
    got = run(True, root)
    assert ref.keys() == got.keys()
    exact = {k: torch.equal(ref[k], got[k]) for k in ref}
    (root / f'proof-rank{dist.get_rank()}.json').write_text(json.dumps(exact, indent=2) + '\n')
    assert all(exact.values()), {k: v for k, v in exact.items() if not v}
    dist.destroy_process_group()
    print('PASS compiled GPT checkpoint vs retained activations: outputs/all gradients exact', flush=True)


if __name__ == '__main__':
    main()
