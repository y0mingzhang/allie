"""Model-only smoothing of decimal rating precision, with larger-shift controls."""
import json,time,os
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .sample_data import read
from .residual_screen import analyze

OFFSETS=(0,-3,3,-25,25,-100,100)


def perturb(prefix,offset):
    p=list(prefix);assert len(p)>=11 and all(0<=x<=9 for x in p[3:11])
    ratings=[int(''.join(map(str,p[a:a+4]))) for a in (3,7)]
    actual=max(-min(ratings),min(offset,9999-max(ratings)))
    for a,r in zip((3,7),ratings):p[a:a+4]=map(int,f'{r+actual:04d}')
    assert p[:3]==prefix[:3] and p[11:]==prefix[11:]
    assert int(''.join(map(str,p[3:7])))-int(''.join(map(str,p[7:11])))==ratings[0]-ratings[1]
    return p,actual


def test():
    prefix=[20,30,40]+list(map(int,'19952007'))+[500,1000]
    assert perturb(prefix,0)[0]==prefix
    for delta in OFFSETS:
        p,d=perturb(prefix,delta);assert d==delta
    edge=[20,30,40]+list(map(int,'00019999'))+[500]
    assert perturb(edge,100)[1]==0 and perturb(edge,-100)[1]==-1
    print('PASS identity, numeric offsets, exact rating-gap preservation and boundary handling',flush=True)


def run(oracle,spec):
    test();start=time.monotonic();out=ROOT/'aug-rating-precision-v1';out.mkdir(exist_ok=True)
    d=read('aug-tune-expanded-v1');rows,games,mask,ids=(d[k] for k in ('rows','games','mask','ids'));n,k=mask.shape
    plan=dict(sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),offsets=list(OFFSETS),roots_per_block=512,
        sources={f:digest(Path(__file__).with_name(f)) for f in ('rating_precision.py','direct.py','residual_screen.py')},
        hypothesis='Decimal Elo tokenization may encode nuisance precision. The earlier +/-200/400 shifts preserved the last two digits, so they did not isolate this issue. Compare common shifts+/-3 and+/-25 with+/-100. Both players shift equally, preserving their rating gap; no moves, time control, or model weights change.',
        correction='For each magnitude average the zero/negative/positive query policies, then tilt the frozen1000 parent by (smoothed/requeried_zero)^kappa. The zero query controls batching/numerical changes. Fit kappa in[0,2], shared or per Elo, inside the original game folds. No golden tuning.',
        accounting='Charge all three full-prefix queries per magnitude/position even if the collection cache physically shares a prefix. These are hypothetical strength queries, not changes to the actual rating used by evaluation or the search. Only current game information; no external memory.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    for lo in range(0,n,plan['roots_per_block']):
        path=out/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+plan['roots_per_block']];size=len(part);queries=[];effective=[]
        for delta in OFFSETS:
            changes=[perturb(r['prefix'],delta) for r in part]
            queries.extend(x[0] for x in changes);effective.append([x[1] for x in changes])
        oracle.reset();tick=time.monotonic();z=oracle(queries).reshape(len(OFFSETS),size,2432)
        logits=z[:,:,378:2346][:,np.arange(size)[:,None],ids[lo:lo+size]]
        probability=softmax(np.where(mask[None,lo:lo+size],logits.astype(float),-np.inf),axis=2)
        stat=dict(seconds=time.monotonic()-tick,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,
            logical_full_prefixes=len(queries),logical_prefill_tokens=sum(map(len,queries)),job=os.environ.get('SLURM_JOB_ID'))
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,probability=probability,effective=effective,game=games[lo:lo+size],ply=[r['ply'] for r in part],stats=json.dumps(stat))
        tmp.replace(path);print('rating-precision',lo+size,n,stat['seconds'],flush=True)
    prob=np.zeros((len(OFFSETS),n,k));stats=[]
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
            prob[:,lo:hi]=f['probability'];stats.append(json.loads(str(f['stats'])))
    features,costs,diagnostics={},{},{}
    for magnitude,pair in [(3,(1,2)),(25,(3,4)),(100,(5,6))]:
        mixture=prob[[0,*pair]].mean(0);name=f'smooth{magnitude}'
        features[name]=np.where(mask,np.log(np.maximum(mixture,1e-300))-np.log(np.maximum(prob[0],1e-300)),0.)
        costs[name]=dict(extra_nodes=np.full(n,3),summary=dict(extra_full_prefixes=3,
            logical_prefill_tokens=3*sum(len(r['prefix']) for r in rows),physical_collection_seconds=sum(s['seconds'] for s in stats),collection_arms=7))
        kl=(prob[0]*(np.log(np.maximum(prob[0],1e-300))-np.log(np.maximum(mixture,1e-300)))).sum(1)
        diagnostics[name]=dict(mean_kl=float(kl.mean()),kl_quantiles=np.quantile(kl,[.1,.5,.9,.99]).tolist(),top1_changed=float(np.mean(prob[0].argmax(1)!=mixture.argmax(1))))
    worker=dict(seconds=time.monotonic()-start,blocks=stats,plan_sha256=digest(pp),diagnostics=diagnostics);atomic(out/'worker.json',worker)
    result=analyze(out.name,features,costs,plan['correction'],bounds=(0.,2.))
    return dict(study=out.name,fit_cv_selected=result['fit_cv_selected'],diagnostics=diagnostics,worker_seconds=worker['seconds'])
