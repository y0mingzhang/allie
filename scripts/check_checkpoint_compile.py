"""Isolate nested compilation/checkpoint behavior in a fresh CPU process.

Run all combinations of --preinit and --nonrecursive separately. Compilation
inside an outer Dynamo trace may create an ordinary callable instead of a
compiled block; inspect both the saved callable and actual profiler events.
"""
import argparse
import json
import torch
from torch.utils.checkpoint import checkpoint
from torch._dynamo.utils import counters

p = argparse.ArgumentParser()
p.add_argument('--preinit', action='store_true')
p.add_argument('--nonrecursive', action='store_true')
p.add_argument('--inside', action='store_true')
a = p.parse_args()


@torch.compiler.disable(recursive=not a.nonrecursive)
def wrapper(fn, x):
    return checkpoint(fn, x, use_reentrant=False)


@torch.compiler.disable(recursive=not a.nonrecursive)
def wrapper_module(module, x):
    if not hasattr(module, '_compiled'):
        module._compiled = torch.compile(module._forward, fullgraph=True)
    return checkpoint(module._compiled, x, use_reentrant=False)


class Block(torch.nn.Module):
    def _forward(self, x):
        return x.sin().cos() * x

    def forward(self, x):
        if a.inside:
            return wrapper_module(self, x)
        if not hasattr(self, '_compiled'):
            self._compiled = torch.compile(self._forward, fullgraph=True)
        return wrapper(self._compiled, x)


torch.set_num_threads(2)
m = Block()
if a.preinit:
    m._compiled = torch.compile(m._forward, fullgraph=True)
net = torch.compile(m)
for _ in range(2):
    x = torch.randn(256, requires_grad=True)
    net(x).sum().backward()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    x = torch.randn(256, requires_grad=True)
    net(x).sum().backward()
events = {e.key: e.count for e in prof.key_averages()
          if 'Compiled' in e.key or 'sin' in e.key or 'cos' in e.key}
print(json.dumps(dict(preinit=a.preinit, nonrecursive=a.nonrecursive, inside=a.inside,
                     wrapped=hasattr(m._compiled, '_torchdynamo_orig_callable'),
                     events=events, graphs=dict(counters['stats']))), flush=True)
