"""Likelihood of strictly past moves under alternative mover Elo headers."""
import json
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest

OFFSETS=[-400,-200,0,200,400]


def condition(prefix,delta):
    p=list(prefix);assert len(p)>=11 and all(0<=t<=9 for t in p[3:11])
    mover=(len(p)-11)%2;start=3+4*mover
    rating=int(''.join(map(str,p[start:start+4])))
    if delta and rating>=400:
        value=min(3200,max(400,rating+delta));p[start:start+4]=list(map(int,f'{value:04d}'))
    assert p[:3]==prefix[:3] and p[11:]==prefix[11:]
    opponent=7 if start==3 else 3;assert p[opponent:opponent+4]==prefix[opponent:opponent+4]
    return p


def history(prefix,limit=32):
    n=len(prefix)-11
    ix=np.array([11+j for j in range(n-2,-1,-2)][:limit],int)
    assert ((ix>=11)&(ix<len(prefix))).all()
    return ix


def test():
    p=[0,0,0,1,8,0,0,2,0,0,0]+[400,401,402,403,404]
    assert history(p).tolist()==[14,12]
    q=condition(p,200);assert q[3:7]==p[3:7] and q[7:11]==[2,2,0,0]
    q=condition(p[:-1],-200);assert q[3:7]==[1,6,0,0] and q[7:11]==p[7:11]
    assert not len(history(p[:11]));assert not len(history(p[:12]))
    assert condition(p,0)==p
    print('PASS own-move parity, causal history selection, header isolation',flush=True)


@contextmanager
def likelihood_capture(oracle,prefixes):
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    old=oracle.runner.forward_extend;found=[]
    ix=[];targets=[];root_index=[];off=0
    for p in prefixes:
        hist=history(p);ix.extend((off+hist-1).tolist());targets.extend([p[j]-378 for j in hist]);root_index.append(off+len(p)-1);off+=len(p)
    @torch.no_grad()
    def wrapped(batch,*args,**kwargs):
        assert batch.extend_num_tokens==off and all(v==0 for v in batch.extend_prefix_lens_cpu)
        batch.capture_hidden_mode=CaptureHiddenMode.FULL
        out,extra=old(batch,*args,**kwargs)
        h=out.hidden_states;assert h is not None and h.shape==(off,512)
        # Preserve the checkpoint's BF16 head transform before converting to FP32.
        wanted=torch.tensor(ix,device=h.device,dtype=torch.long)
        selected=h[wanted];head=oracle.runner.model.model.math.lm_head
        z=torch.nn.functional.linear(selected,head.weight.to(h.dtype));z=(23*torch.sigmoid((z+5)/7.5)).float()
        targets_gpu=torch.tensor(targets,device=h.device,dtype=torch.long)
        logp=torch.log_softmax(z[:,378:2346],dim=1)
        ll=logp[torch.arange(len(ix),device=h.device),targets_gpu].cpu().numpy()
        found.append(ll)
        return out,extra
    oracle.runner.forward_extend=wrapped
    try:yield found
    finally:oracle.runner.forward_extend=old


def run(oracle,spec):
    test();start=time.monotonic();out=ROOT/('header-adaptation-smoke-v1' if spec.get('smoke') else 'aug-header-adaptation-v1');out.mkdir(exist_ok=True)
    sample=ROOT/'aug-tune-expanded-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    if spec.get('smoke'):rows=rows[:16]
    smoke=json.loads((ROOT/'feature-smoke-v1/results.json').read_text());assert smoke['future_mask_identity']
    plan=dict(positions=len(rows),offsets=OFFSETS,max_history_moves=32,sample_sha256=digest(sample),roots_per_block=512,max_prefill_tokens=4096,
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('direct.py'),Path(__file__).parent/'sglang_models/allie.py']},
        semantics='Change current mover Elo digits only, opponent and TC fixed. Past evidence is raw move-vocabulary log likelihood of up to32 strictly earlier own moves, newest first. Root policy uses legal moves. Prefill contains only current prefix; target/future excluded. Unknown rating<400 stays unchanged. Prior width and likelihood power selected on fit-game CV only.',
        interpretation='Offset is a latent header/rating-gap adjustment, not identified playing strength. The original LM already sees these past moves; test whether explicit Bayes adaptation improves its use of evidence.',
        costs='Five full-prefix queries per position; all charged. Static mixtures and adaptive variants share identical queries. No cross-game player memory or neural-weight updates.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    stats=[]
    for lo in range(0,len(rows),512):
        path=out/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        begin=time.monotonic();part=rows[lo:lo+512];n=len(part);k=max(len(r['legal']) for r in part)
        probability=np.zeros((5,n,k));evidence=np.zeros((5,n,32));counts=np.array([len(history(r['prefix'])) for r in part])
        fwd=0.;prefill=0;calls=0
        for vi,delta in enumerate(OFFSETS):
            group=[];size=0;seen=set()
            def flush():
                nonlocal group,size,seen,fwd,prefill,calls
                if not group:return
                sequences=[p for i,p in group];oracle.reset()
                with likelihood_capture(oracle,sequences) as history_ll:z=oracle(sequences)
                assert len(history_ll)==1;ll=history_ll[0];at=0
                for j,(i,p) in enumerate(group):
                    legal=np.array(part[i]['legal']);probability[vi,i,:len(legal)]=softmax(z[j,legal].astype(float))
                    ct=counts[i];evidence[vi,i,:ct]=ll[at:at+ct];at+=ct
                assert at==len(ll)
                fwd+=oracle.forward_seconds;prefill+=oracle.new_tokens;calls+=len(group)
                group=[];size=0;seen=set()
            for i,r in enumerate(part):
                p=condition(r['prefix'],delta)
                if group and (size+len(p)>4096 or tuple(p) in seen):flush()
                group.append((i,p));size+=len(p);seen.add(tuple(p))
            flush()
        assert calls==5*n and np.isfinite(evidence).all()
        stat=dict(seconds=time.monotonic()-begin,forward_seconds=fwd,prefill_tokens=prefill,full_prefix_queries=calls)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,probability=probability,evidence=evidence,counts=counts,game=[r['game'] for r in part],ply=[r['ply'] for r in part],stats=json.dumps(stat))
        tmp.replace(path);stats.append(stat);print('header posterior',lo+n,'/',len(rows),stat['seconds'],flush=True)
    # Re-read all completed blocks for accurate resume accounting.
    stats=[]
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:stats.append(json.loads(str(f['stats'])))
    result=dict(positions=len(rows),invocation_seconds=time.monotonic()-start,blocks=stats,plan_sha256=digest(pp))
    atomic(out/'worker.json',result);return result

if __name__=='__main__':test()
