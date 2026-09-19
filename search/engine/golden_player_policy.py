"""Frozen golden test of static and past-choice-adapted search mixtures."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,softmax
from .service import ROOT,atomic
from .balanced_eval import digest
from .analyze_player_search import FACTORS,posterior,mix
from .golden_metrics import summarize,controls

OUT=ROOT/'golden-player-search-v1'
def freeze():
    OUT.mkdir(exist_ok=True);dev=ROOT/'aug-player-search-v1/results.json';report=json.loads(dev.read_text())
    assert report['fit_cv_selected']==dict(macro_ce='adapt_s1.0_e1.0',expert_ce='adapt_s0.5_e0.25')
    inputs=sorted((ROOT/'golden-adaptive-temperature-v1').glob('[0-9]*.npz'))
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in ('golden_player_search.py','analyze_player_search.py','native_board.py','board.cpp','handles.py','direct.py','golden_metrics.py')]]
    plan=dict(sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),dev_sha256=digest(dev),
        laws_sha256=digest(ROOT/'training-cm-laws.json'),max_history=8,roots_per_batch=256,
        parameters=report['parameters'],
        methods=dict(constant=[0.,0.],prior_s05=[.5,0.],prior_s10=[1.,0.],adaptive_s05=[.5,.25],adaptive_s10=[1.,1.]),
        cached_root_input_hashes={p.name:digest(p) for p in inputs},sources={p.name:digest(p) for p in files},
        selection='Both August fit-CV selected adaptive configurations, plus their identical cheaper static mixtures and the constant search control. All arms reported; no selection on golden. August extra history gains small/uncertain, so exploratory comparison, not a prior claim of dominance.',
        semantics='Same fixed checkpoint, actual pre-existing1000 root trees from golden-adaptive-temperature-v1. New historical queries use only prefix moves; per-query posterior uses globally frozen August calibration. No golden fitting. Historical roots and children charged even if physically reused; extra prefill reported. Reused golden sample requires fresh final confirmation.')
    pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return plan

def analyze():
    start=time.monotonic();plan=freeze();rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    n=len(rows);ar=np.arange(n);group=np.array([r['cell']%4 for r in rows])
    index=np.load(OUT/'history-index.npy');valid=index>=0;worker=json.loads((OUT/'worker.json').read_text())
    assert worker['index_sha256']==digest(OUT/'history-index.npy')
    nh=worker['unique_past_positions'];saved=[]
    for path in sorted(OUT.glob('history-*.npz')):
        with np.load(path) as z:saved.append({k:z[k] for k in ('z','q','mask','target','nodes','prefix_lengths')})
    kh=max(x['z'].shape[1] for x in saved);hz=np.zeros((nh,kh));hq=np.zeros_like(hz);hm=np.zeros((nh,kh),bool)
    ht=np.zeros(nh,int);hn=np.zeros(nh,int);hl=np.zeros(nh,int);at=0
    for z in saved:
        end=at+len(z['target']);kk=z['z'].shape[1];hz[at:end,:kk]=z['z'];hq[at:end,:kk]=z['q'];hm[at:end,:kk]=z['mask'];ht[at:end]=z['target'];hn[at:end]=z['nodes'];hl[at:end]=z['prefix_lengths'];at=end
    assert at==nh
    evidence=np.zeros((n,len(FACTORS)))
    for g in range(4):
        fp=plan['parameters']['history'][str(g)]
        z=fp['alpha']*hz[None,:,:]+(FACTORS*fp['beta'])[:,None,None]*hq[None,:,:]
        lp=np.where(hm[None,:,:],z,-np.inf);ll=(lp[:,np.arange(nh),ht]-logsumexp(lp,axis=2)).T
        for i in np.flatnonzero(group==g):evidence[i]=ll[index[i,valid[i]]].sum(0)
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    cq=np.zeros((n,k));root=np.zeros((n,2432));nodes=np.zeros(n);port=[];legal=[]
    for lo in range(0,n,1024):
        path=ROOT/'golden-adaptive-temperature-v1'/f'{lo:06d}.npz'
        assert digest(path)==plan['cached_root_input_hashes'][path.name]
        with np.load(path) as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1]
            np.testing.assert_array_equal(z['game'],[r['game'] for r in rows[lo:hi]])
            np.testing.assert_array_equal(z['ply'],[r['ply'] for r in rows[lo:hi]])
            np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk])
            cq[lo:hi,:kk]=z['q'][0];root[lo:hi]=z['z'];nodes[lo:hi]=z['constant_nodes'];port.append(z['port_raw']);legal.append(z['new_legal'])
    logits=root[:,378:2346][ar[:,None],ids]
    alpha=np.array([plan['parameters']['root'][str(g)]['alpha'] for g in group]);beta=np.array([plan['parameters']['root'][str(g)]['beta'] for g in group])
    base=softmax(np.where(mask,alpha[:,None]*logits+beta[:,None]*cq,-np.inf),axis=1)
    child_cost=np.where(valid,hn[np.maximum(index,0)],0).sum(1);root_cost=valid.sum(1)
    prefills=np.where(valid,hl[np.maximum(index,0)],0).sum(1)
    scores,costs=controls(rows);scores['new_raw']=np.concatenate(port);scores['new_legal']=np.concatenate(legal)
    costs['new_raw']=np.zeros(n);costs['new_legal']=np.zeros(n)
    for name,(sigma,power) in plan['methods'].items():
        p=base if sigma==0 else mix(base,cq,beta,mask,posterior(evidence,sigma,power))
        chosen=np.where(p==p.max(1)[:,None],ids,1968).min(1)
        scores[name]=np.stack([-np.log(p[ar,target]),chosen==ids[ar,target],p.max(1)],axis=1)
        costs[name]=nodes+(child_cost+root_cost if power else 0)
    result=summarize(rows,scores,costs,references=('new_legal','constant','prior_s05','prior_s10'))
    report=dict(methods=result,positions=n,plan_sha256=digest(OUT/'plan.json'),worker=worker,
        mean_extra_past_roots=float(root_cost.mean()),mean_extra_past_children=float(child_cost.mean()),mean_extra_prefill_tokens=float(prefills.mean()),
        analysis_seconds=time.monotonic()-start,
        caveat=plan['semantics']+' Total standalone search runtime is NOT the history-only wall time; current-root scoring was cached. Count-based cost includes both stages. CM conditional on frozen training law; no familywise correction.')
    atomic(OUT/'results.json',report)
    with (OUT/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(plan['methods']),scores=np.stack([scores[name] for name in plan['methods']]),nodes=np.stack([costs[name] for name in plan['methods']]),game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    for name in plan['methods']:print(name,{k:result[name][k] for k in ('macro','expert_macro','mean_nodes','macro_training_eq_cm','expert_macro_training_eq_cm')},flush=True)
    return report
if __name__=='__main__':analyze()
