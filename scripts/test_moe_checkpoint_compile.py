"""NCCL gather/reduce-scatter proof for the expert-only checkpoint boundary.

Uses actual sharded MoE and compiled expert kernels. Compare checkpointed
compiled child with the identical compiled child retaining its activations.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import modded_moe


@torch.compiler.disable
def retain(module, *args):
    if not hasattr(module, '_compiled_experts_out'):
        module._compiled_experts_out = torch.compile(module.experts_out, dynamic=False, fullgraph=True)
    return module._compiled_experts_out(*args)


def run(checkpointed, root):
    original = modded_moe._recompute
    if not checkpointed:
        modded_moe._recompute = retain
    torch.manual_seed(701)
    m = modded_moe.MoE(128, 16, 2, 80, 128, kind='swiglu', kernel='scatter',
                       shard=True, seq=.001).cuda().train()
    with torch.no_grad():
        g = torch.Generator(device='cuda').manual_seed(810 + dist.get_rank())
        m.down.normal_(0, .01, generator=g)
        m.shared_down.normal_(0, .01, generator=g)
    for name, p in m.named_parameters():
        if name != 'router':
            p.data = p.data.bfloat16()
    net = torch.compile(m, dynamic=False, fullgraph=False)
    def step():
        m.zero_grad(set_to_none=True)
        g = torch.Generator(device='cuda').manual_seed(340 + dist.get_rank())
        x = torch.randn(1024, 128, device='cuda', dtype=torch.bfloat16,
                        generator=g, requires_grad=True)
        y = net(x)
        y.float().square().mean().backward()
        return x, y
    for _ in range(2):
        step()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        x, y = step()
        torch.cuda.synchronize()
    proof = {'output': y.detach().cpu(), 'dx': x.grad.cpu()}
    proof.update({n: p.grad.cpu().clone() for n, p in m.named_parameters()})
    count = sum(e.count for e in prof.key_averages() if e.key == 'CompiledFunction')
    assert count > 0, 'no compiled autograd child observed'
    prof.export_chrome_trace(str(root / f'shard-checkpoint-{checkpointed}-rank{dist.get_rank()}.json'))
    print(json.dumps(dict(checkpointed=checkpointed, rank=dist.get_rank(),
                          compiled_autograd_calls=count)), flush=True)
    modded_moe._recompute = original
    del m, net, x, y
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
    assert dist.get_world_size() == 2
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    modded_moe.BLOCK_RECOMPUTE = False
    ref = run(False, root)
    got = run(True, root)
    exact = {k: torch.equal(ref[k], got[k]) for k in ref}
    (root / f'shard-proof-rank{dist.get_rank()}.json').write_text(json.dumps(exact, indent=2) + '\n')
    assert all(exact.values()), {k: v for k, v in exact.items() if not v}
    dist.destroy_process_group()
    print('PASS sharded MoE: compiled checkpoint vs retained gradients exact', flush=True)


if __name__ == '__main__':
    main()
