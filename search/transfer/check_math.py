"""Independent frozen-source mathematical parity before GPU kernel comparisons."""
import argparse,json,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .model import load_checkpoint,DenseBackend
from .canonical import ROOT,BASE,MODELS,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('size',choices=MODELS);a=p.parse_args()
    torch.set_num_threads(4);run,study=MODELS[a.size];source=BASE/'results/recipe10x'/study/'source-ours'
    sys.path.insert(0,str(source))
    dist.init_process_group('gloo',init_method='file://'+tempfile.mktemp(),rank=0,world_size=1)
    from modded_medium import Config,create_model,make_context,core
    portable,path=load_checkpoint(BASE/'results/pretrain'/run/'last.pt');state=torch.load(path,map_location='cpu',weights_only=False)
    for name,want in state['source_sha256'].items():assert sha(source/name)==want
    cfg=Config(**state['config']);ref=create_model(cfg,device='cpu');ref.scalars.data=state['model']['scalars'].clone()
    ref.load_state_dict(state['model']);inf=state['inference'];ref.split_embed=inf['split_embed']
    for k in ('angular_freq','cos','sin'):getattr(ref.yarn,k).copy_(inf['yarn'][k])
    ref.yarn.attn_scale=inf['yarn']['attn_scale'];ref.eval();ref.float();portable.float()
    schedule=core.ForwardScheduleConfig(None,inf['ws_short'],inf['ws_long'])
    sample=json.loads((ROOT/'aug-tune-expanded-v1/sample.json').read_text())['positions']
    with np.load(ROOT/'aug-tune-v1/feats.npz') as f:side=f['feats']
    checks=[];start=time.monotonic()
    for row in [sample[i] for i in (0,17,47,100)]:
        ids=torch.tensor(row['prefix']);n=len(ids);lo=row['column']-n
        feats=torch.tensor(side[row['row'],lo:row['column']]);pad=(-n)%16
        ids=torch.cat((ids,ids.new_full((pad,),2348)))
        feats=torch.cat((feats,feats.new_full((pad,3),-1)))
        positions=torch.arange(len(ids))
        ctx=make_context(ids[None],schedule.ws_short*128,schedule.ws_long*128,backend='dense')
        with torch.inference_mode():
            a=ref(ids,ids,ctx,schedule,feat_seq=feats)[0]
            b=portable(ids,positions,DenseBackend(),feats,ctx.board)
        a,b=a[:n],b[:n]
        diff=float((a-b).abs().max());torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-5)
        checks.append(dict(length=n,max_absolute_diff=diff))
    out=ROOT/'transfer-v1'/('math-'+p.parse_args().size+'.json')
    out.write_text(json.dumps(dict(model=run,mode='FP32 eager same SDPA math, frozen-source vs port',checks=checks,
        seconds=time.monotonic()-start,checkpoint_sha256=sha(path)),indent=2)+'\n')
    print('MATH PASS',run,checks,flush=True);dist.destroy_process_group()


if __name__=='__main__':main()
