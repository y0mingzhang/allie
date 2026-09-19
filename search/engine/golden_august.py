"""Freeze August fit-CV choices, then score the unchanged golden sample.

Retain compact raw tree summaries so later dev-selected output policies can be
tested without repeating neural search. Never choose parameters from golden.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.special import softmax,logsumexp
from .balanced_eval import ROOT,GLOBAL_STOP,atomic,digest
from .fit_policy import loss_gradient
from .distributional_native import load,test

OUT=ROOT/'golden-augcal-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    base=json.loads((ROOT/'aug-search-v1/results.json').read_text())
    dist=json.loads((ROOT/'aug-search-v1/distributional-results.json').read_text())
    methods={}
    def regular(key,budget):
        prefix,direction,group=key.rsplit('_',2)
        value=prefix[:-len(str(budget))] if budget else 'zero'
        return dict(budget=budget,value=value,direction=direction,group=group,parameters=base['results'][key]['parameters'])
    for budget in (16,64,256,1000):
        keys=[k for k in base['results'] if any(k.startswith(p+str(budget)+'_') for p in ('mcts','expectation','soft0.1'))]
        for metric in ('macro_ce','expert_ce'):
            key=min(keys,key=lambda k:base['results'][k]['fit_game_cv'][metric])
            methods[key]=regular(key,budget)
    cheap=min((k for k in base['results'] if k.startswith('temperature_')),
        key=lambda k:base['results'][k]['fit_game_cv']['macro_ce'])
    methods['aug_direct']=dict(budget=0,value='zero',direction='forward',group=cheap.rsplit('_',1)[1],parameters=base['results'][cheap]['parameters'],source=cheap)
    methods['mcts1000_reverse_global']=regular('mcts1000_reverse_global',1000)
    assert len(set(dist['fit_cv_selected'].values()))==1
    combo=next(iter(dist['fit_cv_selected'].values()));assert combo=='wdl1000_q_soft_elo'
    methods[combo]=dict(budget=1000,features=['z','q','soft0.5'],group='elo',parameters=dist['results'][combo]['parameters'])
    # Previous fixed-output method is a numerical regression control.
    old=json.loads((ROOT/'golden-mcts1000-v1/plan.json').read_text())['methods']['reverse']
    methods['old_mcts_reverse']=dict(budget=1000,value='mcts',direction='reverse',group='global',parameters={'0':old})
    paths=[Path(__file__),*[Path(__file__).with_name(n) for n in ('analyze_augcal.py','fit_policy.py','distributional.cpp','distributional_native.py','backups.cpp','board.cpp','mcts_native.hpp','direct.py','model.py')]]
    plan=dict(methods=methods,budgets=[16,64,256,1000],roots_per_block=128,
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),weights_sha256=digest(ROOT/'serving-export/model.safetensors'),
        sources={p.name:digest(p) for p in paths},
        calibration_sources={name:digest(ROOT/'aug-search-v1'/name) for name in ('results.json','distributional-results.json')},
        laws_sha256=digest(ROOT/'training-cm-laws.json'),
        selection='Per-budget fit-game-CV macro/expert choices on August. No confirmation-fold or golden selection. All frozen methods reported; same reused golden sample, not fresh untouched data.')
    dest=OUT/'plan.json'
    if dest.exists():assert json.loads(dest.read_text())==plan
    else:atomic(dest,plan)
    return plan


def apply(method,z,values,mask,y,cells,ids):
    group={'global':np.zeros(len(z),int),'elo':cells%4,'format':cells//4}[method['group']]
    p=np.zeros_like(z)
    for g in np.unique(group):
        selected=group==g;params=method['parameters'][str(int(g))]
        if 'features' in method:
            x=np.stack([z,values['mcts'],values['soft0.5']],axis=2)
            p[selected]=softmax(np.where(mask[selected],np.einsum('nkd,d->nk',x[selected],params['theta']),-np.inf),axis=1)
        else:
            q=np.zeros_like(z) if method['value']=='zero' else values[method['value']]
            p[selected],_=loss_gradient([params['alpha'],params['beta']],z[selected],q[selected],mask[selected],y[selected],method['direction'],return_policy=True)
    # Match the evaluator's vocabulary-order tie-break, not legal-generation order.
    prediction=np.where(mask&(p==p.max(1)[:,None]),ids,1968).min(1)
    return np.stack([-np.log(p[np.arange(len(z)),y]),prediction==ids[np.arange(len(z)),y],p.max(1)],axis=1)


def run(oracle,spec):
    plan=freeze();module=load();test(module)
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];bs=plan['roots_per_block'];start=time.monotonic()
    for lo in range(0,len(rows),bs):
        dest=OUT/f'{lo:06d}.npz'
        if dest.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);ar=np.arange(n);oracle.reset();begin=time.monotonic()
        root=oracle([r['prefix'] for r in part]);ids=np.zeros((n,max(len(r['legal']) for r in part)),int);mask=np.zeros_like(ids,bool);y=np.zeros(n,int)
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;y[i]=r['legal'].index(r['target'])
        z=root[:,378:2346][ar[:,None],ids].astype(float);cells=np.array([r['cell'] for r in part])
        lookup=[{int(a):j for j,a in enumerate(row[:int(m.sum())])} for row,m in zip(ids,mask)]
        tree=module.Tree([r['prefix'] for r in part],root,[1000]*n,[1.25]*n)
        snapshots=[];counts=[];nodecounts=[];times=[];wdls=[]
        for step in range(1,1001):
            prefixes=tree.select()
            if prefixes:tree.update(oracle(prefixes))
            if step not in plan['budgets']:continue
            q=np.zeros((4,*ids.shape),np.float32);visits=np.zeros_like(ids,np.int32);wdl=np.zeros((*ids.shape,3),np.float32)
            for i,(moves,v,scores,prior) in enumerate(tree.snapshot()):
                ix=[lookup[i][m] for m in moves];q[0,i,ix]=scores;visits[i,ix]=v
            for j,snap in enumerate(tree.backups([float('inf'),.5,.1]),1):
                for i,(moves,scores) in enumerate(snap):q[j,i,[lookup[i][m] for m in moves]]=scores
            for i,(moves,p) in enumerate(tree.distribution()):wdl[i,[lookup[i][m] for m in moves]]=p
            snapshots.append(q);counts.append(visits);nodecounts.append(tree.evals);wdls.append(wdl);times.append(time.monotonic()-begin)
        stats=tree.stats();del tree
        logits=root[:,378:2346].astype(float);target=np.array([r['target']-378 for r in part]);raw=softmax(logits,axis=1)
        payload=dict(port_raw=np.stack([logsumexp(logits,axis=1)-logits[ar,target],logits.argmax(1)==target,raw.max(1)],axis=1))
        payload['legal']=apply(dict(group='global',value='zero',direction='forward',parameters={'0':dict(alpha=1.,beta=0.)}),z,{},mask,y,cells,ids)
        for name,method in plan['methods'].items():
            index=plan['budgets'].index(method['budget']) if method['budget'] else 0
            values=dict(zip(('mcts','expectation','soft0.5','soft0.1'),snapshots[index]))
            payload[name]=apply(method,z,values,mask,y,cells,ids)
        # Same tree, positions, output constants: verify against the earlier run.
        with np.load(ROOT/f'golden-mcts1000-v1/{lo:06d}.npz') as old:
            for name,reference in [('port_raw','port_raw'),('legal','legal'),('old_mcts_reverse','mcts_reverse')]:
                np.testing.assert_allclose(payload[name],old[reference],atol=2e-6,rtol=0)
        stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
        temp=dest.with_suffix('.partial')
        with temp.open('wb') as f:np.savez_compressed(f,**payload,root=root,ids=ids,mask=mask,q=np.array(snapshots),
            visits=np.array(counts),wdl=np.array(wdls),evaluated_nodes=np.array(nodecounts),elapsed=np.array(times),
            game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]),stats=json.dumps(stats))
        temp.replace(dest)
        if lo%(bs*4)==0:print('Golden August calibration',lo+n,'/',len(rows),flush=True)
    subprocess.run([sys.executable,'-B','-m','search.engine.analyze_augcal'],check=True,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
    return dict(stage='Frozen August-selected golden frontier comparison complete',positions=len(rows),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':freeze()
