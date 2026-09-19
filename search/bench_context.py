"""GPU differential test/benchmark of mask construction, separate from live oracle."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import fast_context

SOURCE=Path('/data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours')
ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def check(a,b):
    assert torch.equal(a.documents,b.documents) and torch.equal(a.same_previous,b.same_previous)
    for field in ('short_mask','long_mask'):
        left,right=getattr(a,field),getattr(b,field)
        for prefix in ('','full_'):
            ln=getattr(left,prefix+'kv_num_blocks');rn=getattr(right,prefix+'kv_num_blocks')
            assert torch.equal(ln,rn),(field,prefix,ln,rn)
            li=getattr(left,prefix+'kv_indices');ri=getattr(right,prefix+'kv_indices')
            active=torch.arange(li.shape[-1],device=li.device)<ln[...,None]
            assert torch.equal(li[active],ri[active]),(field,prefix,'indices')


def main():
    torch.cuda.set_device(0);dist.init_process_group('nccl')
    sys.path.insert(0,str(SOURCE))
    from modded_medium import make_context,Context
    rng=np.random.default_rng(730)
    rows=json.loads((ROOT/'dev.json').read_text())['positions']
    timings=[]
    for size in (4096,8192):
        for iteration in range(8):
            x=np.full(size,2348,np.int64);pos=0
            while True:
                p=rows[int(rng.integers(len(rows)))]['prefix'];n=((len(p)+127)//128)*128
                if pos+n>size:break
                x[pos:pos+len(p)]=p;pos+=n
            xt=torch.as_tensor(x,device='cuda').reshape(1,-1)
            # Include tight windows that generate partially masked edge blocks.
            short,long=(128,384) if iteration%2 else (384,896)
            old=make_context(xt,short,long);new=fast_context.make_context(x,xt,short,long,Context)
            check(old,new)
            for label,fn in [('original',lambda:make_context(xt,short,long)),
                             ('block_metadata',lambda:fast_context.make_context(x,xt,short,long,Context))]:
                torch.cuda.synchronize();t=time.monotonic()
                for _ in range(10):fn()
                torch.cuda.synchronize();timings.append(dict(size=size,iteration=iteration,method=label,
                                                           seconds=(time.monotonic()-t)/10))
    report=dict(status='Exact active partial/full block metadata match; full-model parity still required',timings=timings)
    (ROOT/'context-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    for size in (4096,8192):
        print(size,{name:float(np.median([r['seconds'] for r in timings if r['size']==size and r['method']==name]))
                    for name in ('original','block_metadata')},flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
