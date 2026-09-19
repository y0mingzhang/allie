"""Freeze and check the August-selected subtree-dependent Bellman temperature."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .balanced_eval import ROOT,GLOBAL_STOP,atomic,digest
from .threadforest_native import load
from .adaptive_temperature_native import load as reducer_load
from .handles import HandleOracle
from .fit_policy import loss_gradient
from .golden_metrics import summarize,controls
OUT=ROOT/'golden-adaptive-temperature-v1'

def freeze():
    OUT.mkdir(exist_ok=True);dev=ROOT/'aug-adaptive-temperature-v1/results.json';d=json.loads(dev.read_text())
    assert set(d['fit_cv_selected'].values())=={'subtree'}
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in ('adaptive_temperature.cpp','adaptive_temperature_native.py',
        'threadforest.cpp','threadforest_native.py','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py','fit_policy.py','golden_metrics.py')]]
    plan=dict(budget=1000,methods=dict(constant=[0,.1,16],subtree=[3,.2,16]),
        parameters={name:d['results'][key]['parameters'] for name,key in [('constant','constant010'),('subtree','subtree')]},
        roots_per_block=1024,threads=4,skip_forced=True,
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),laws_sha256=digest(ROOT/'training-cm-laws.json'),
        dev_sha256=digest(dev),sources={p.name:digest(p) for p in files},
        selection='Subtree temperature selected by both August fit-game CV metrics. Paired August CIs include zero narrowly; exploratory golden check, not already established gain. Same actual1000-simulation trees for both backups, same source code and batches. Output coefficients fit on August only.',
        cost_note='Both methods share exactly the same tree and node count. Four expansion threads, compact-handle inference. Reused golden sample; no fresh final confirmation. Subtree size is a heuristic exploration proxy, not independent samples.')
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:atomic(path,plan)
    return plan

def run(oracle,spec):
    plan=freeze();module=load();reducer=reducer_load()
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];bs=plan['roots_per_block'];start=time.monotonic()
    for lo in range(0,len(rows),bs):
        path=OUT/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);ar=np.arange(n);groups=np.array([r['cell']%4 for r in part])
        budgets=[0 if len(r['legal'])==1 else plan['budget'] for r in part]
        assert sum(budgets)+sum(len(r['prefix']) for r in part)<oracle.runner.max_total_num_tokens
        oracle.reset();begin=time.monotonic();bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits
        tree=module.Tree([r['prefix'] for r in part],z,budgets,[2.5]*n,plan['threads'])
        while not tree.done:
            h=tree.select()
            if len(h):tree.update(bridge(h))
        k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
        data=tree.compact();qs={}
        for name,(mode,tau,scale) in plan['methods'].items():qs[name]=reducer.reduce(data,1000,mode,tau,scale)[ar[:,None],ids]
        ref=module.reduce(data,1000,.1,.1)[ar[:,None],ids];np.testing.assert_array_equal(qs['constant'],ref)
        nodes=np.array(tree.evals);stats=tree.stats();del tree,data
        logits=z[:,378:2346][ar[:,None],ids].astype(float);payload={}
        def metrics(p,moves,targets):
            predicted=np.where(p==p.max(1)[:,None],moves,1968).min(1)
            return np.stack([-np.log(p[ar,targets]),predicted==moves[ar,targets],p.max(1)],axis=1)
        payload['port_raw']=metrics(softmax(z[:,378:2346].astype(float),axis=1),np.broadcast_to(np.arange(1968),(n,1968)),np.array([r['target']-378 for r in part]))
        payload['new_legal']=metrics(softmax(np.where(mask,logits,-np.inf),axis=1),ids,target)
        for name in plan['methods']:
            p=np.zeros_like(logits)
            for g in np.unique(groups):
                selected=groups==g;f=plan['parameters'][name][str(int(g))]
                p[selected],_=loss_gradient([f['alpha'],f['beta']],logits[selected],qs[name][selected],mask[selected],target[selected],'forward',return_policy=True)
            payload[name]=metrics(p,ids,target);payload[name+'_nodes']=nodes
        stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_nonroot_nn_requests=bridge.queries)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,**payload,z=z,ids=ids,mask=mask,q=np.stack(list(qs.values())),stats=json.dumps(stats),
            game=[r['game'] for r in part],ply=[r['ply'] for r in part])
        tmp.replace(path);print('Golden adaptive temperature',lo+n,'/',len(rows),stats['seconds'],flush=True)
    scores,costs=controls(rows);stats=[];old=[];old_nodes=[]
    for lo in range(0,len(rows),512):
        with np.load(ROOT/f'golden-coverage-v1/{lo:06d}.npz') as f:
            assert list(f['game'])==[r['game'] for r in rows[lo:lo+len(f['game'])]]
            old.append(f['coverage_fixed']);old_nodes.append(f['coverage_fixed_nodes'])
    scores['coverage1000']=np.concatenate(old);costs['coverage1000']=np.concatenate(old_nodes)
    names=['port_raw','new_legal',*plan['methods']]
    for name in names:scores[name]=[];costs[name]=[]
    for lo in range(0,len(rows),bs):
        with np.load(OUT/f'{lo:06d}.npz') as f:
            part=rows[lo:lo+len(f['game'])];assert list(f['game'])==[r['game'] for r in part];np.testing.assert_array_equal(f['ply'],[r['ply'] for r in part])
            for name in names:
                scores[name].append(f[name]);costs[name].append(f[name+'_nodes'] if name in plan['methods'] else np.zeros(len(part)))
            stats.append(json.loads(str(f['stats'])))
    for name in names:scores[name]=np.concatenate(scores[name]);costs[name]=np.concatenate(costs[name])
    analysis=time.monotonic();result=summarize(rows,scores,costs,references=('new_legal','four_ply','coverage1000','constant'))
    report=dict(methods=result,positions=len(rows),plan_sha256=digest(OUT/'plan.json'),
        scoring_seconds=sum(s['seconds'] for s in stats),analysis_seconds=time.monotonic()-analysis,
        new_tokens=sum(s['new_tokens'] for s in stats),unique_nonroot_nn_requests=sum(s['unique_nonroot_nn_requests'] for s in stats),
        caveat=plan['cost_note']+' CM conditional on frozen training law; intervals exclude law uncertainty and repeated benchmark reuse.')
    atomic(OUT/'results.json',report)
    for name in names:print(name,{k:result[name][k] for k in ('macro','expert_macro','mean_nodes','macro_training_eq_cm','expert_macro_training_eq_cm')},flush=True)
    return dict(positions=len(rows),elapsed_seconds=time.monotonic()-start)
