"""Full frozen-checkpoint parity before enabling fast context in the resident server."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import fast_context
from bench_context import SOURCE,ROOT,check


def main():
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=True;dist.init_process_group('nccl')
    sys.path.insert(0,str(SOURCE))
    from modded_medium import Config,Context,core,create_model,make_context
    ckpt=Path('/data/group_data/dei-group/yimingz3/allie/results/pretrain/r2-3e16-control-t20-w20-pf052h-s42/last.pt')
    pointer=torch.load(ckpt,map_location='cpu',weights_only=False)
    state=torch.load(ckpt.parent/pointer['directory']/'model.pt',map_location='cpu',weights_only=False)
    import hashlib
    for f,h in state['source_sha256'].items():assert hashlib.sha256((SOURCE/f).read_bytes()).hexdigest()==h
    model=create_model(Config(**state['config']));model.scalars.data=state['model']['scalars'].to('cuda',dtype=model.scalars.dtype)
    model.load_state_dict(state['model']);inf=state['inference'];model.split_embed=inf['split_embed']
    for k in ('angular_freq','cos','sin'):getattr(model.yarn,k).copy_(inf['yarn'][k].to('cuda'))
    model.yarn.attn_scale=inf['yarn']['attn_scale'];model.eval()
    cos=model.yarn.cos.clone();sin=model.yarn.sin.clone()
    schedule=core.ForwardScheduleConfig(None,inf['ws_short'],inf['ws_long'])
    net=torch.compile(model,dynamic=False,fullgraph=True)
    rows=json.loads((ROOT/'dev.json').read_text())['positions']
    tests=[]
    for lo in (0,128,1024,1728):
        pos=0;x=np.full(4096,2348,np.int64);rp=np.zeros(4096,np.int64);idx=[]
        used=[]
        for row in rows[lo:lo+32]:
            p=row['prefix'];n=((len(p)+127)//128)*128
            if pos+n>4096:break
            x[pos:pos+len(p)]=p;rp[pos:pos+len(p)]=np.arange(len(p));idx.append(pos+len(p)-1)
            pos+=n;used.append(p)
        xt=torch.as_tensor(x,device='cuda').reshape(1,-1);rot=torch.as_tensor(rp,device='cuda')
        model.yarn.cos[:4096].copy_(cos[rot]);model.yarn.sin[:4096].copy_(sin[rot])
        old=make_context(xt,schedule.ws_short*128,schedule.ws_long*128)
        new=fast_context.make_context(x,xt,schedule.ws_short*128,schedule.ws_long*128,Context)
        check(old,new);predictions=[];times=[]
        for label,ctx in [('original',old),('block_metadata',new)]:
            with torch.inference_mode():
                result=net(xt.flatten(),xt.flatten(),ctx,schedule).reshape(-1,2432)
                predictions.append(result[torch.as_tensor(idx,device='cuda')].float().cpu().numpy())
                torch.cuda.synchronize();start=time.monotonic()
                for _ in range(10):net(xt.flatten(),xt.flatten(),ctx,schedule)
                torch.cuda.synchronize();times.append((time.monotonic()-start)/10)
        assert np.array_equal(*predictions),float(np.max(np.abs(predictions[0]-predictions[1])))
        # Also match the running service, guarding a mistaken restore/rotary adapter.
        from client import Oracle
        assert np.array_equal(predictions[1],Oracle()(used))
        tests.append(dict(start=lo,prefixes=len(used),max_logit_difference=0,original_forward_seconds=times[0],fast_forward_seconds=times[1]))
    (ROOT/'context-model-parity.json').write_text(json.dumps(dict(status='bit-identical to original context and live service',tests=tests),indent=2)+'\n')
    print(json.dumps(tests,indent=2),flush=True);dist.destroy_process_group()


if __name__=='__main__':main()
