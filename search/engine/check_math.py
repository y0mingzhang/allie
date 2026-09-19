"""Check portable math against frozen source, separately from kernel rounding."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .model import DenseBackend, load_checkpoint

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
BASE=Path('/data/group_data/dei-group/yimingz3/allie')


def main():
    torch.set_num_threads(4)
    dist.init_process_group('gloo',init_method='file://'+tempfile.mktemp(),rank=0,world_size=1)
    sys.path.insert(0,str(BASE/'results/recipe10x/data-v1-round2/source-ours'))
    from modded_medium import Config,core,create_model,make_context
    portable,path=load_checkpoint(BASE/'results/pretrain/r2-3e16-control-t20-w20-pf052h-s42/last.pt','cuda')
    state=torch.load(path,map_location='cpu',weights_only=False)
    original=create_model(Config(**state['config']));original.load_state_dict(state['model']);original.eval()
    inf=state['inference'];original.split_embed=inf['split_embed']
    for k in ('angular_freq','cos','sin'):getattr(original.yarn,k).copy_(inf['yarn'][k])
    original.yarn.attn_scale=inf['yarn']['attn_scale']
    schedule=core.ForwardScheduleConfig(None,inf['ws_short'],inf['ws_long'])
    rows=json.loads((ROOT/'dev.json').read_text())['positions']
    ref=np.load(ROOT/'cache-00000.npz')['root'][:32].astype(float)
    if '--trace' in sys.argv:
        traces={};hooks=[]
        for i,block in enumerate(original.blocks):
            hooks.append(block.register_forward_pre_hook(lambda m,a,i=i:traces.__setitem__(f'input{i}',a[0][0].clone())))
            hooks.append(block.register_forward_hook(lambda m,a,o,i=i:traces.__setitem__(f'output{i}',o[0].clone())))
        class Trace(DenseBackend):
            def trace(self,name,x):
                print(name,float((x-traces[name]).abs().max()),float((x[:len(rows[0]['prefix'])]-traces[name][:len(rows[0]['prefix'])]).abs().max()),x[0,:6].tolist(),traces[name][0,:6].tolist(),flush=True)
        ids=torch.tensor(rows[0]['prefix']+[2347]*(512-len(rows[0]['prefix'])),device='cuda')
        ctx=make_context(ids[None],schedule.ws_short*128,schedule.ws_long*128,backend='dense')
        with torch.inference_mode():
            x=portable.embed(ids)
            print('normdiff',float((portable.norm(x)-torch.nn.functional.rms_norm(x,(512,))).abs().max()),flush=True)
            for k,v in original.state_dict().items():
                if not torch.equal(v,portable.state_dict()[k]):print('weight diff',k,flush=True)
            original(ids,ids,ctx,schedule)
            portable(ids,torch.arange(512,device='cuda'),Trace())
        dist.destroy_process_group();return
    result={};outputs={}
    for compiled in (False,True):
        m=torch.compile(portable,dynamic=False) if compiled else portable
        orig=torch.compile(original,dynamic=False) if compiled else original
        predictions=[];diffs=[]
        for r in rows[:32]:
            n=len(r['prefix']);seq=r['prefix']+[2347]*(512-n)
            assert len(seq)==512
            ids=torch.tensor(seq,device='cuda');positions=torch.arange(512,device='cuda')
            ctx=make_context(ids[None],schedule.ws_short*128,schedule.ws_long*128,backend='dense')
            with torch.inference_mode():
                a=m(ids,positions,DenseBackend())
                b=orig(ids,ids,ctx,schedule)[0]
            diffs.append(float((a[:n]-b[:n]).abs().max()))
            predictions.append(a[n-1].float().cpu().numpy())
        z=np.array(predictions,dtype=float)
        def lp(x):
            x=x[:,378:2346];x=x-x.max(1,keepdims=True)
            return x-np.log(np.exp(x).sum(1,keepdims=True))
        a,b=lp(ref),lp(z);targets=np.array([r['target']-378 for r in rows[:32]])
        result['compiled' if compiled else 'eager']=dict(max_vs_same_kernel_source=max(diffs),
            mean_kl_vs_frozen_flex=float((np.exp(a)*(a-b)).sum(1).mean()),
            ce_delta_vs_frozen_flex=float((a-b)[np.arange(len(a)),targets].mean()))
        print(result,flush=True)
    (ROOT/'portable-math.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
