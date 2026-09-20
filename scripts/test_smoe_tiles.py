"""Exact candidate-size layer proof for stock-output tile choices, with fullgraph.

No direct-gradient path: outputs, input/router/shared/expert gradients must all
equal stock ScatterMoE, including cumulative FP32 micro-batch accumulation.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
import modded_moe
from modded_moe import MoE
from modded_medium_core import accumulate_fp32


def run(kernel, experts, tokens, skew):
    torch.manual_seed(701)
    m = MoE(2048, experts, 6, 455, 2728, kind='swiglu', kernel=kernel,
            init=.006, seq=.001).cuda().train()
    with torch.no_grad():
        m.down.normal_(0, .02)
        m.shared_down.normal_(0, .02)
        if skew:
            # Bias routes into a subset, leaving many experts entirely unused.
            m.bias[:12].fill_(10)
    for name, p in m.named_parameters():
        if name != 'router':
            p.data = p.data.bfloat16()
            p.main_grad = torch.full_like(p, float('nan'), dtype=torch.float32)
            p.fresh = True
            p.register_post_accumulate_grad_hook(accumulate_fp32)
    f = torch.compile(m, fullgraph=True, dynamic=False)
    out = {}
    for micro in range(2):
        torch.manual_seed(340 + micro)
        x = torch.randn(tokens, 2048, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        y = f(x)
        y.float().square().mean().backward()
        out[f'y{micro}'] = y.detach().cpu()
        out[f'dx{micro}'] = x.grad.cpu()
    for name, p in m.named_parameters():
        g = p.main_grad if hasattr(p, 'main_grad') else p.grad
        assert bool(g.isfinite().all()), name
        out[name] = g.detach().cpu().clone()
    del f, m, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experts', type=int, required=True)
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--skew', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--reference', default='scatter')
    p.add_argument('--candidate', default='scatter-tuned')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    modded_moe.BLOCK_RECOMPUTE = True
    ref = run(a.reference, a.experts, a.tokens, a.skew)
    got = run(a.candidate, a.experts, a.tokens, a.skew)
    equal = {k: torch.equal(ref[k], got[k]) for k in ref}
    report = dict(experts=a.experts, tokens=a.tokens, skew=a.skew,
                  reference=a.reference, candidate=a.candidate,
                  compiled_fullgraph=True, microbatches=2, equal=equal)
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    assert all(equal.values()), equal
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
