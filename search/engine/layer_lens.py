"""Root logit-lens features via eager batched SDPA; resident checkpoint unchanged."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest

LAYERS=[1,3,5]

class BatchBackend:
    def __init__(self,batch,length,last):
        self.batch,self.length,self.last=batch,length,last;self.states={}
    def previous_embedding(self,x,positions):
        b,l=self.batch,self.length;x=x.reshape(b,l,-1)
        return torch.cat((torch.zeros_like(x[:,:1]),x[:,:-1]),dim=1).reshape(-1,x.shape[-1])
    def shift_keys(self,k,positions,layer):
        b,l=self.batch,self.length;h,d=k.shape[-2:]
        v=k.reshape(b,l,h,d);out=v.clone()
        out[:,1:,:,d//4:d//2]=v[:,:-1,:,d//4:d//2]
        out[:,1:,:,3*d//4:]=v[:,:-1,:,3*d//4:]
        return out.reshape_as(k)
    def attention(self,q,k,v,layer,window,scale):
        b,l=self.batch,self.length
        q,k,v=(x.reshape(b,l,*x.shape[-2:]).transpose(1,2) for x in (q,k,v))
        if window>=l-1:
            out=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
        else:
            ix=torch.arange(l,device=q.device);mask=(ix[:,None]>=ix[None,:])&(ix[:,None]-ix[None,:]<=window)
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=scale)
        return out.transpose(1,2).reshape(b*l,*out.shape[1:2],out.shape[-1])
    def trace(self,name,x):
        if name in [f'output{i}' for i in LAYERS]:self.states[name]=x[self.last].clone()

@torch.no_grad()
def query(oracle,prefixes):
    model=oracle.runner.model.model.math;device=model.embed.weight.device
    n=len(prefixes);length=max(map(len,prefixes));token=np.zeros((n,length),np.int64)
    for i,p in enumerate(prefixes):token[i,:len(p)]=p
    ids=torch.as_tensor(token,device=device).flatten()
    pos=torch.arange(length,device=device).repeat(n);last=torch.tensor([i*length+len(p)-1 for i,p in enumerate(prefixes)],device=device)
    backend=BatchBackend(n,length,last)
    hidden=model.forward_hidden(ids,pos,backend)
    states=[model.norm(backend.states[f'output{i}']) for i in LAYERS]+[hidden[last]]
    logits=torch.stack([model.logits(h) for h in states]).float().cpu().numpy()
    return logits,n*length

def groups(rows,start,end,limit=4096):
    group=[];longest=0
    for i in range(start,end):
        length=len(rows[i]['prefix'])
        if group and max(longest,length)*(len(group)+1)>limit:
            yield group;group=[];longest=0
        group.append(i);longest=max(longest,length)
    if group:yield group

def run(oracle,spec):
    smoke=bool(spec.get('smoke'));out=ROOT/('layer-lens-smoke-v1' if smoke else 'aug-layer-lens-v1');out.mkdir(exist_ok=True)
    sample=ROOT/'aug-tune-expanded-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    if smoke:rows=rows[:32]
    plan=dict(layers=LAYERS,positions=len(rows),sample_sha256=digest(sample),max_padded_tokens=4096,
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('model.py')]},
        research='Layer contrast inspired by DoLa, https://arxiv.org/abs/2309.03883. This is a chess calibration experiment, not a claim that language factuality results transfer. Intermediate residuals read through final RMS normalization and frozen head; true final hidden includes learned backout. Compare global calibration control and intermediate/final features, all legal moves positive.',
        semantics='Separate eager batched SDPA root pass on the same resident weights. No mutations to serving model or its KV state. Parent search policy stays frozen; feature ratios use paired eager logits. No external model/data or weight training.',
        cost='One additional full-prefix forward plus four readouts per query. Padding and actual prefix tokens, wall time and final-vs-SGLang parity reported.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    start=time.monotonic();stats=[]
    for lo in range(0,len(rows),512):
        path=out/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+512];n=len(part);k=max(len(r['legal']) for r in part)
        result=np.zeros((4,n,k));begin=time.monotonic();padded=0
        for ix in groups(rows,lo,lo+n):
            z,cost=query(oracle,[rows[i]['prefix'] for i in ix]);padded+=cost
            for j,i in enumerate(ix):result[:,i-lo,:len(rows[i]['legal'])]=z[:,j,rows[i]['legal']]
        stat=dict(seconds=time.monotonic()-begin,padded_tokens=padded,prefix_tokens=sum(len(r['prefix']) for r in part),full_prefix_queries=n)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,logits=result,game=[r['game'] for r in part],ply=[r['ply'] for r in part],stats=json.dumps(stat))
        tmp.replace(path);stats.append(stat);print('layer lens',lo+n,'/',len(rows),stat['seconds'],flush=True)
    report=dict(positions=len(rows),seconds=time.monotonic()-start,blocks=stats,plan_sha256=digest(pp))
    if smoke:
        from .model import DenseBackend
        prefixes=[r['prefix'] for r in rows[:8]];batch,_=query(oracle,prefixes)
        model=oracle.runner.model.model.math;device=model.embed.weight.device
        # Independent single-document backend checks shape/causal/state indexing.
        singles=[]
        with torch.no_grad():
            for p in prefixes:
                t=torch.tensor(p,device=device);pos=torch.arange(len(p),device=device)
                singles.append(model.forward(t,pos,DenseBackend())[-1].float().cpu().numpy())
        oracle.reset();sg=oracle(prefixes)
        dense=np.stack(singles)
        report['single_vs_batch_max_logit']=float(np.abs(dense-batch[-1]).max())
        report['sglang_vs_eager_max_logit']=float(np.abs(sg-batch[-1]).max())
        # Same shape, altered future suffix: compare intermediate/final features
        # before that suffix through each backend.
        assert report['single_vs_batch_max_logit']<.3,report
        assert report['sglang_vs_eager_max_logit']<.3,report
        # A prefix's result must be independent of other sequences in the batch,
        # up to the observed BF16 matmul tolerance.
        altered=[prefixes[0],*[[0]*len(p) for p in prefixes[1:]]]
        changed,_=query(oracle,altered)
        report['other_document_max_logit']=float(np.abs(changed[:,0]-batch[:,0]).max())
        assert report['other_document_max_logit']==0.,report
        print('SMOKE',report,flush=True)
    atomic(out/'worker.json',report);return report

if __name__=='__main__':raise SystemExit('Resident-engine task')
