"""Frozen deeper root-coverage check using the verified handle/forest stack."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from .balanced_eval import ROOT,GLOBAL_STOP,atomic,digest
from .threadforest_native import load
from .handles import HandleOracle
from .fit_policy import loss_gradient
from .golden_metrics import summarize,controls

OUT=ROOT/'golden-fast-coverage-v1'

def freeze():
    OUT.mkdir(exist_ok=True)
    shallow=ROOT/'aug-coverage-v1/results.json';deep=ROOT/'aug-coverage-deep-v1/results.json'
    a=json.loads(shallow.read_text());b=json.loads(deep.read_text())
    assert set(b['fit_cv_selected'].values())=={'coverage_bernoulli_4000_elo'}
    params={'1000':a['results']['coverage_bernoulli_1000_elo']['parameters'],
            '4000':b['results']['coverage_bernoulli_4000_elo']['parameters']}
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in
        ('threadforest.cpp','threadforest_native.py','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py','fit_policy.py','golden_metrics.py')]]
    plan=dict(budgets=[1000,4000],parameters=params,roots_per_block=320,threads=2,skip_forced=True,
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),laws_sha256=digest(ROOT/'training-cm-laws.json'),
        development_hashes={p.parent.name:digest(p) for p in (shallow,deep)},sources={p.name:digest(p) for p in files},
        selection='4000 selected by both August fit-game CV metrics;1000 is the existing selected control with its original output coefficients. No golden parameter fitting. All forced single-legal-move roots skip search (exact policy shortcut).',
        cost_note='4000 actually executed;1000 is a prefix-tree counterfactual reconstructed from original logical simulation births. Implementation changes CPU scheduling and GPU batch composition; own port raw/legal and old actual1000 controls are reported. No Pareto-dominance claim for higher cost alone. Reused golden sample, no fresh confirmation.')
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:atomic(path,plan)
    return plan

def run(oracle,spec):
    plan=freeze();module=load();rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    bs=plan['roots_per_block'];start=time.monotonic()
    for lo in range(0,len(rows),bs):
        path=OUT/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);ar=np.arange(n);groups=np.array([r['cell']%4 for r in part])
        budgets=[0 if len(r['legal'])==1 else 4000 for r in part]
        assert sum(budgets)+sum(len(r['prefix']) for r in part)<oracle.runner.max_total_num_tokens
        assert sum(budgets)+n<oracle.capacity
        oracle.reset();begin=time.monotonic();bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits
        tree=module.Tree([r['prefix'] for r in part],z,budgets,[2.5]*n,plan['threads'])
        while not tree.done:
            h=tree.select()
            if len(h):tree.update(bridge(h))
        k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
        data=tree.compact();qs=[];nodes=[]
        for budget in plan['budgets']:
            qs.append(module.reduce(data,budget,.1,.1)[ar[:,None],ids])
            nodes.append(tree.prefix_evals(budget))
        nodes=np.array(nodes);np.testing.assert_array_equal(nodes[-1],tree.evals)
        stats=tree.stats();del tree,data
        logits=z[:,378:2346][ar[:,None],ids].astype(float);payload={}
        def metrics(p,moves,targets):
            predicted=np.where(p==p.max(1)[:,None],moves,1968).min(1)
            return np.stack([-np.log(p[ar,targets]),predicted==moves[ar,targets],p.max(1)],axis=1)
        payload['fast_port_raw']=metrics(softmax(z[:,378:2346].astype(float),axis=1),np.broadcast_to(np.arange(1968),(n,1968)),np.array([r['target']-378 for r in part]))
        payload['fast_legal']=metrics(softmax(np.where(mask,logits,-np.inf),axis=1),ids,target)
        for j,budget in enumerate(plan['budgets']):
            p=np.zeros_like(logits)
            for g in np.unique(groups):
                selected=groups==g;f=plan['parameters'][str(budget)][str(int(g))]
                p[selected],_=loss_gradient([f['alpha'],f['beta']],logits[selected],qs[j][selected],mask[selected],target[selected],'forward',return_policy=True)
            payload[f'fast{budget}']=metrics(p,ids,target);payload[f'fast{budget}_nodes']=nodes[j]
        stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_nonroot_nn_requests=bridge.queries)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,**payload,z=z,ids=ids,mask=mask,q=np.array(qs),evaluated_nodes=nodes,stats=json.dumps(stats),
            game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]))
        tmp.replace(path);print('Fast golden coverage',lo+n,'/',len(rows),stats['seconds'],flush=True)
    scores,costs=controls(rows);stats=[];old=[];old_nodes=[]
    for lo in range(0,len(rows),512):
        with np.load(ROOT/f'golden-coverage-v1/{lo:06d}.npz') as f:
            assert list(f['game'])==[r['game'] for r in rows[lo:lo+len(f['game'])]]
            old.append(f['coverage_fixed']);old_nodes.append(f['coverage_fixed_nodes'])
    scores['coverage1000']=np.concatenate(old);costs['coverage1000']=np.concatenate(old_nodes)
    names=['fast_port_raw','fast_legal','fast1000','fast4000']
    for name in names:scores[name]=[];costs[name]=[]
    for lo in range(0,len(rows),bs):
        with np.load(OUT/f'{lo:06d}.npz') as f:
            part=rows[lo:lo+len(f['game'])];assert list(f['game'])==[r['game'] for r in part];np.testing.assert_array_equal(f['ply'],[r['ply'] for r in part])
            for name in names:
                scores[name].append(f[name]);costs[name].append(f[name+'_nodes'] if name.startswith('fast1') or name.startswith('fast4') else np.zeros(len(part)))
            stats.append(json.loads(str(f['stats'])))
    for name in names:scores[name]=np.concatenate(scores[name]);costs[name]=np.concatenate(costs[name])
    analysis=time.monotonic();result=summarize(rows,scores,costs,references=('fast_legal','four_ply','coverage1000','fast1000'))
    report=dict(methods=result,positions=len(rows),plan_sha256=digest(OUT/'plan.json'),
        scoring_seconds=sum(s['seconds'] for s in stats),analysis_seconds=time.monotonic()-analysis,
        new_tokens=sum(s['new_tokens'] for s in stats),unique_nonroot_nn_requests=sum(s['unique_nonroot_nn_requests'] for s in stats),
        caveat=plan['cost_note']+' CM conditional on frozen training law; intervals exclude law uncertainty and repeated benchmark reuse.')
    atomic(OUT/'results.json',report)
    for name in names:print(name,{k:result[name][k] for k in ('macro','expert_macro','mean_nodes','macro_training_eq_cm','expert_macro_training_eq_cm')},flush=True)
    return dict(positions=len(rows),elapsed_seconds=time.monotonic()-start)
