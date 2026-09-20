"""Regression for compiled MoE layers sharing optimizer gradient allocations."""
import argparse
import json
import torch
from torch import nn
from modded_moe import MoE
from modded_medium_core import accumulate_fp32
from modded_smoe_tuned import finish_direct_accum


class Stack(nn.Module):
    def __init__(self, kernel, shared):
        super().__init__()
        torch.manual_seed(191)
        self.layers = nn.ModuleList([
            MoE(128, 16, 2, 32, 128, kind='swiglu', kernel=kernel, init=.006)
            for _ in range(3)
        ])
        self.flat = {}
        for name in ('up', 'down', 'shared_up', 'shared_down'):
            ps = [getattr(m, name) for m in self.layers]
            buf = torch.full((len(ps)+2, *ps[0].shape), float('nan'), device='cuda')
            buf[0].fill_(-917); buf[-1].fill_(-917)
            self.flat[name] = buf
            for i,p in enumerate(ps):
                p.data = p.data.cuda().bfloat16()
                p.main_grad = buf[i+1] if shared else torch.full_like(p, float('nan'), dtype=torch.float32)
                p.acc = (buf,i+1) if shared else (p.main_grad,-1)
                p.fresh = True
                p.register_post_accumulate_grad_hook(
                    finish_direct_accum if kernel == 'scatter-accum' and name in ('up','down') else accumulate_fp32)
        self.cuda()

    def forward(self, x):
        for m in self.layers:
            x = x + m(torch.nn.functional.rms_norm(x, (128,)))
        return x


def run(kernel, shared, compiled):
    m = Stack(kernel, shared)
    f = torch.compile(m, fullgraph=True, dynamic=False) if compiled else m
    result = {}
    for step in range(3):
        for p in m.parameters():
            p.grad = None
            if hasattr(p, 'main_grad'): p.fresh = True
        for micro in range(3):
            torch.manual_seed(70+step*10+micro)
            x = torch.randn(1024,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            y = f(x)
            y.float().square().sum().backward()
            grads = {n:(p.main_grad if hasattr(p,'main_grad') else p.grad) for n,p in m.named_parameters()}
            summary = {n:dict(finite=bool(g.isfinite().all()),norm=float(g.float().norm()),fresh=getattr(dict(m.named_parameters())[n],'fresh',None)) for n,g in grads.items()}
            print(json.dumps(dict(kernel=kernel,shared=shared,compiled=compiled,step=step,micro=micro,bad={k:v for k,v in summary.items() if not v['finite']})),flush=True)
            assert all(v['finite'] for v in summary.values()), summary
            for name,buf in m.flat.items():
                assert bool((buf[0]==-917).all() & (buf[-1]==-917).all()), ('guard overwritten',name)
        result[step] = {n:g.detach().cpu().clone() for n,g in grads.items()}
        with torch.no_grad():
            for n,p in m.named_parameters(): p.add_(grads[n].to(p.dtype),alpha=-1e-5)
    return result


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True
    p=argparse.ArgumentParser();p.add_argument('--compiled',action='store_true');p.add_argument('--separate',action='store_true');a=p.parse_args()
    ref=run('scatter',not a.separate,a.compiled)
    got=run('scatter-accum',not a.separate,a.compiled)
    for step in ref:
        for n in ref[step]: torch.testing.assert_close(got[step][n],ref[step][n],atol=0,rtol=0)
    print('PASS stack forward/backward/updates',flush=True)
