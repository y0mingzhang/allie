"""Deliberately fail one oversized allocation on an assigned, otherwise idle GPU."""
import argparse
import json
import pickle
from pathlib import Path

import torch
import modded_memsnap as trace

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
torch.cuda.set_device(0)
out = Path(a.out)
assert not out.with_name(out.stem + '-oom-rank0.pickle').exists()
trace.start(out, 0)
kept = torch.empty(1024*1024, device='cuda', dtype=torch.uint8)
total = torch.cuda.get_device_properties(0).total_memory
try:
    torch.empty(total*2, device='cuda', dtype=torch.uint8)
    raise AssertionError('oversized allocation unexpectedly succeeded')
except torch.OutOfMemoryError:
    trace.dump_on_error()
for kind in ('oom', 'exception'):
    path = trace._active.filename(kind)
    with path.open('rb') as f: snap = pickle.load(f)
    assert any(e['action'] == 'oom' for events in snap['device_traces'] for e in events)
    assert any(b['state'] == 'active_allocated' and b['size'] >= kept.numel()
               for seg in snap['segments'] for b in seg['blocks'])
print(json.dumps({'oom_observer': 'pass', 'exception_fallback': 'pass',
                  'gpu': torch.cuda.get_device_name()}), flush=True)
