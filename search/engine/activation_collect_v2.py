"""Read-only prefix-gradient activation correction; no neural parameter updates."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .features import full
from .header_adaptation import history

STEPS=[0.,1.,4.,16.,64.]
HISTORIES=[8,32]


def rms(x):return (x.square().mean(-1,keepdim=True)+1e-12).sqrt()


def head_logits(h,w):
    h=h/rms(h);u=h@w.T
    return 23*torch.sigmoid((u+5)/7.5)


def correction(past,target,root,w,valid):
    """Exact gradient of normalized frozen-head past CE, transferred via inner product."""
    norm=rms(past);h=past/norm;u=h@w.T;s=torch.sigmoid((u+5)/7.5);z=23*s
    p=z.softmax(-1);res=p.clone()
    res.scatter_add_(-1,target[...,None],-torch.ones_like(target[...,None],dtype=p.dtype))
    gu=(23/7.5)*s*(1-s)*res;gh=gu@w
    g=(gh-h*(gh*h).mean(-1,keepdim=True))/norm
    g=g*valid[...,None]
    kernel=(past*root[:,None,:]).mean(-1)
    count=valid.sum(1).clamp_min(1)
    delta=-(kernel[...,None]*g).sum(1)/count[:,None]
    return delta,p,kernel


def test():
    # Independent autograd identity-adapter derivative; the current label never exists.
    torch.manual_seed(782)
    dtype=torch.float64;b,t,d,v=3,4,5,7
    past=torch.randn(b,t,d,dtype=dtype);root=torch.randn(b,d,dtype=dtype);w=torch.randn(v,d,dtype=dtype)*.3
    target=torch.randint(v,(b,t));valid=torch.tensor([[1,1,1,1],[1,1,0,0],[0,0,0,0]],dtype=torch.bool)
    delta,_,_=correction(past,target,root,w,valid)
    for i in range(b):
        a=torch.zeros(d,d,dtype=dtype,requires_grad=True)
        z=head_logits(past[i]+past[i]@a.T,w)
        loss=(torch.nn.functional.cross_entropy(z,target[i],reduction='none')*valid[i]).sum()/valid[i].sum().clamp_min(1)
        ga=torch.autograd.grad(loss,a)[0]
        expected=-(ga@root[i])/d
        torch.testing.assert_close(delta[i],expected,atol=1e-11,rtol=1e-9)
    torch.testing.assert_close(delta[2],torch.zeros(d,dtype=dtype),atol=0,rtol=0)
    # Zero step is exact for this readout; ratios therefore leave frozen parent unchanged.
    torch.testing.assert_close(head_logits(root+0*delta,w),head_logits(root,w),atol=0,rtol=0)
    print('PASS identity-adapter gradient reference, missing history, exact zero-step ratio',flush=True)


@torch.no_grad()
def run(oracle,spec):
    start=time.monotonic();out=ROOT/spec['output'];out.mkdir(exist_ok=True)
    sample=ROOT/spec['sample'];rows=json.loads(sample.read_text())['positions']
    plan=dict(sample_sha256=digest(sample),steps=STEPS,histories=HISTORIES,max_tokens=4096,
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('features.py'),Path(__file__).with_name('header_adaptation.py')]},
        math='For past own move t, g_t=grad_h CE(frozen_head(RMSNorm(h_t)),past_target_t). Current delta_h=-mean_t(g_t * dot(h_t,h_root)/d). This equals one identity-adapter gradient step divided by d; evaluated without mutating any parameter. Bound perturbation RMS to0.3 of root RMS. Re-read frozen normalized head at scalar steps0/1/4/16/64. Prefix only.',
        validation='Current target is never passed into correction. Past target indices are strictly inside supplied prefix; latent gradients use past logits only. Reference test against autograd. Scalar output mixing selected only inside August fit-game folds. Explicit readout arithmetic FP32; score feature is ratio against SAME zero-step readout, so baseline remains exact.',
        cost='One extra full-prefix query per root plus past and corrected-root head projections. No external data, persistent adaptation or neural-weight updates.',
        hypothesis='Learned output embeddings transfer historical errors across related actions, unlike direct same-token residual. Report centered logit-change correlation to both matching-kernel residual and old cosine-kernel residual; if >0.9 expect redundancy.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    w=oracle.runner.model.model.math.lm_head.weight[378:2346].float()
    device=w.device;stats=[]
    for lo in range(0,len(rows),512):
        path=out/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+512];n=len(part);k=max(len(r['legal']) for r in part);result=np.zeros((2,5,n,k));old=np.zeros((2,n,k));same=np.zeros_like(old)
        perturb=np.zeros((2,n));count=np.zeros(n,int);begin=time.monotonic();prefill=0;forward=0.;group=[];size=0;seen=set()
        def flush():
            nonlocal group,size,seen,prefill,forward
            if not group:return
            prefixes=[p for i,p in group];oracle.reset();_,hidden=full(oracle,prefixes);m=len(group)
            ph=np.zeros((m,32,512),np.float32);rh=np.zeros((m,512),np.float32);target=np.zeros((m,32),np.int64);valid=np.zeros((m,32),bool);at=0
            for j,(i,p) in enumerate(group):
                ix=history(p);count[i]=len(ix);valid[j,:len(ix)]=True
                ph[j,:len(ix)]=hidden[at+ix-1];rh[j]=hidden[at+len(p)-1]
                target[j,:len(ix)]=np.array(p)[ix]-378;at+=len(p)
            assert at==len(hidden)
            ht=torch.from_numpy(ph).to(device);hr=torch.from_numpy(rh).to(device)
            y=torch.from_numpy(target).to(device);vm=torch.from_numpy(valid).to(device)
            for hi,limit in enumerate(HISTORIES):
                vv=vm.clone();vv[:,limit:]=False
                delta,p,kernel=correction(ht,y,hr,w,vv)
                ratio=rms(delta)/rms(hr);perturb[hi,[i for i,prefix in group]]=ratio[:,0].cpu().numpy()
                resid=torch.nn.functional.one_hot(y,1968).to(p.dtype)-p
                # Same signed kernel / count as activation gradient, and actual old cosine kernel.
                count_t=vv.sum(1).clamp_min(1)
                r_same=(resid*(kernel*vv)[...,None]).sum(1)/count_t[:,None]
                cosine=kernel/(rms(ht)[:,:,0]*rms(hr))
                a=torch.softmax(torch.where(vv,cosine/.1,torch.full_like(cosine,-1e9)),dim=1)*vv
                a=a/a.sum(1,keepdim=True).clamp_min(1e-30)
                r_old=(resid*a[...,None]).sum(1)
                for si,step in enumerate(STEPS):
                    gain=step*delta;scale=torch.clamp(.3*rms(hr)/rms(gain),max=1.)
                    z=head_logits(hr+gain*scale,w).cpu().numpy()
                    for j,(i,prefix) in enumerate(group):result[hi,si,i,:len(part[i]['legal'])]=z[j,np.array(part[i]['legal'])-378]
                ro,rs=r_old.cpu().numpy(),r_same.cpu().numpy()
                for j,(i,prefix) in enumerate(group):
                    ids=np.array(part[i]['legal'])-378;old[hi,i,:len(ids)]=ro[j,ids];same[hi,i,:len(ids)]=rs[j,ids]
            prefill+=oracle.new_tokens;forward+=oracle.forward_seconds
            group=[];size=0;seen=set()
        for i,r in enumerate(part):
            p=r['prefix']
            if group and (size+len(p)>4096 or tuple(p) in seen):flush()
            group.append((i,p));size+=len(p);seen.add(tuple(p))
        flush()
        stat=dict(seconds=time.monotonic()-begin,prefill_tokens=prefill,forward_seconds=forward,full_prefix_queries=n)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,logits=result,old=old,same=same,perturb_rms=perturb,count=count,game=[r['game'] for r in part],ply=[r['ply'] for r in part],stats=json.dumps(stat))
        tmp.replace(path);stats.append(stat);print('activation adaptation',lo+n,'/',len(rows),stat['seconds'],flush=True)
    stats=[]
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:stats.append(json.loads(str(f['stats'])))
    result=dict(positions=len(rows),seconds=time.monotonic()-start,blocks=stats,plan_sha256=digest(pp))
    atomic(out/'worker.json',result);return result

if __name__=='__main__':test()
