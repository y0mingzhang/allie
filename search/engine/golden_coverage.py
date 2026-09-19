"""Golden check of the August fit-CV-selected root coverage allocation."""
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT,GLOBAL_STOP,atomic,digest
from .coverage_native import load
from .fit_policy import loss_gradient
from .golden_metrics import summarize,controls

OUT=ROOT/'golden-coverage-v1'


def freeze():
    OUT.mkdir(exist_ok=True);dev=ROOT/'aug-coverage-v1/results.json';d=json.loads(dev.read_text())
    assert set(d['fit_cv_selected'].values())=={'coverage_bernoulli_1000_elo'}
    params={str(b):d['results'][f'coverage_bernoulli_{b}_elo']['parameters'] for b in (256,1000)}
    methods=dict(coverage_fixed=[1000]*4,coverage_routed=[256,256,1000,1000])
    plan=dict(methods=methods,parameters=params,cpuct=2.5,dev_sha256=digest(dev),roots_per_block=512,
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),laws_sha256=digest(ROOT/'training-cm-laws.json'),
        sources={p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(s) for s in ('golden_metrics.py','backups.cpp','mcts_native.hpp','board.cpp','direct.py','fit_policy.py')]]},
        selection='Root sqrt(prior*(1-prior)) quota selected by August fit-gameCV for both metrics. Interior PUCT cp2.5, zero FPU. Incremental August CIs vs cp2.5 overlap zero; exploratory golden family check. Fixed1000 plus the previously frozen Elo routing form; all output parameters from corresponding August budgets. No golden retuning.',
        cost_note='Fixed method executes all1000 simulations; routed cost is cached prefix stopping. Logical per-position NN requests and physical runtime/token counts are separate. Reused golden sample; no fresh confirmation claim.')
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic(p,plan)
    return plan


def run(oracle,spec):
    plan=freeze();module=load();rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];bs=plan['roots_per_block'];start=time.monotonic()
    for lo in range(0,len(rows),bs):
        path=OUT/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);ar=np.arange(n);groups=np.array([r['cell']%4 for r in part]);oracle.reset();begin=time.monotonic()
        z=oracle([r['prefix'] for r in part]);tree=module.Tree([r['prefix'] for r in part],z,[1000]*n,[2.5]*n,0,0.,2.)
        k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
        qs=[];nodes=[];unique=0
        for step in range(1,1001):
            prefixes=tree.select()
            if prefixes:
                before=oracle.next_row;tree.update(oracle(prefixes));unique+=oracle.next_row-before
            if step in (256,1000):
                q=np.zeros((n,k))
                for i,(moves,values) in enumerate(tree.backups([.1])[0]):
                    lookup=dict(zip(moves,values));q[i,mask[i]]=[lookup[a] for a in ids[i,mask[i]]]
                qs.append(q);nodes.append(tree.evals)
        stats=tree.stats();del tree;nodes=np.array(nodes);logits=z[:,378:2346][ar[:,None],ids].astype(float);payload={}
        for name,budgets in plan['methods'].items():
            p=np.zeros_like(logits);nc=np.zeros(n)
            for g in np.unique(groups):
                selected=groups==g;budget=budgets[g];b=[256,1000].index(budget);f=plan['parameters'][str(budget)][str(int(g))]
                p[selected],_=loss_gradient([f['alpha'],f['beta']],logits[selected],qs[b][selected],mask[selected],target[selected],'forward',return_policy=True)
                nc[selected]=nodes[b,selected]
            predicted=np.where(mask&(p==p.max(1)[:,None]),ids,1968).min(1)
            payload[name]=np.stack([-np.log(p[ar,target]),predicted==ids[ar,target],p.max(1)],axis=1);payload[name+'_nodes']=nc
        stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_nonroot_nn_requests=unique)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,**payload,z=z,ids=ids,mask=mask,q=np.array(qs),evaluated_nodes=nodes,stats=json.dumps(stats),game=np.array([r['game'] for r in part]))
        tmp.replace(path);print('Golden root coverage',lo+n,'/',len(rows),stats['seconds'],flush=True)
    scores,costs=controls(rows);stats=[]
    previous=[];previous_nodes=[]
    for lo in range(0,len(rows),bs):
        with np.load(ROOT/f'golden-explore-v1/{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            previous.append(z['cp25_fixed']);previous_nodes.append(z['cp25_fixed_nodes'])
    scores['cp25_fixed']=np.concatenate(previous);costs['cp25_fixed']=np.concatenate(previous_nodes)
    for name in plan['methods']:scores[name]=[];costs[name]=[]
    for lo in range(0,len(rows),bs):
        with np.load(OUT/f'{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            for name in plan['methods']:scores[name].append(z[name]);costs[name].append(z[name+'_nodes'])
            stats.append(json.loads(str(z['stats'])))
    for name in plan['methods']:scores[name]=np.concatenate(scores[name]);costs[name]=np.concatenate(costs[name])
    analysis_start=time.monotonic();results=summarize(rows,scores,costs)
    old=json.loads((ROOT/'golden-augcal-v1/results.json').read_text())['methods']
    for name,ref in [('four_ply','previous_four_ply'),('legal','legal'),('soft1000','soft0.11000_forward_elo')]:
        for key in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm'):np.testing.assert_allclose(results[name][key],old[ref][key],atol=1e-12,rtol=0)
    report=dict(methods=results,positions=len(rows),plan_sha256=digest(OUT/'plan.json'),
        scoring_seconds=sum(s['seconds'] for s in stats),analysis_seconds=time.monotonic()-analysis_start,
        new_tokens=sum(s['new_tokens'] for s in stats),unique_nonroot_nn_requests=sum(s['unique_nonroot_nn_requests'] for s in stats),
        caveat=plan['cost_note']+' CM conditional on transferred training-law shape; bootstrap excludes law-fit and cumulative benchmark reuse uncertainty.')
    atomic(OUT/'results.json',report)
    for name,r in results.items():print(name,*(r[k] for k in ('mean_nodes','macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm')),flush=True)
    return dict(positions=len(rows),elapsed_seconds=time.monotonic()-start)
