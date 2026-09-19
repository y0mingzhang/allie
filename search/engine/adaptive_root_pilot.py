"""Adaptive root CE-curvature quotas, tested against a same-stack static control."""
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic,inside
from .balanced_eval import digest
from .adaptive_root_native import load,test
from .handles import HandleOracle

VARIANTS=[['quota_static',0.],['quota_update50',.5],['quota_update100',1.]]

def run(oracle,spec):
    module=load();test(module);out=inside(spec.get('output',ROOT/'aug-adaptive-root-v1'));out.mkdir(exist_ok=True)
    source=ROOT/'aug-tune-v1/sample.json';rows=json.loads(source.read_text())['positions'];bs=1024;budgets=[64,256,1000]
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in
        ('adaptive_root.cpp','adaptive_root_native.py','threadforest.cpp','threadforest_native.py','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py')]]
    plan=dict(variants=VARIANTS,budgets=budgets,roots_per_batch=bs,block=32,inverse_temps=[2.,4.,6.,8.],threads=4,skip_forced=True,
        sample_sha256=digest(source),sources={p.name:digest(p) for p in files},
        stage='August potentially model-training-seen development; game-disjoint fit/confirmation. Root quota uses sqrt((1-mix)*p*(1-p)+mix*pi*(1-pi)), pi=softmax(logp+beta*Qsoft). Weights update every32 logical visits. Beta fixed a priori at2/4/6/8 by Elo, not fit on targets. Interior ordinary PUCTcp2.5, same soft output. Static control shares this stack, batching and block schedule. No golden or CM claim.')
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:atomic(path,plan)
    start=time.monotonic()
    for name,mix in VARIANTS:
        folder=out/name;folder.mkdir(exist_ok=True)
        for lo in range(0,len(rows),bs):
            path=folder/f'{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            part=rows[lo:lo+bs];n=len(part);ar=np.arange(n);oracle.reset();begin=time.monotonic()
            prefixes=[r['prefix'] for r in part];budget=[0 if len(r['legal'])==1 else 1000 for r in part]
            assert sum(budget)+sum(map(len,prefixes))<oracle.runner.max_total_num_tokens
            bridge=HandleOracle(oracle,prefixes);z=bridge.root_logits
            beta=[plan['inverse_temps'][r['cell']%4] for r in part]
            tree=module.Tree(prefixes,z,budget,[2.5]*n,plan['threads'],mix,beta,plan['block'])
            while not tree.done:
                h=tree.select()
                if len(h):tree.update(bridge(h))
            k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool)
            for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
            data=tree.compact();q=[];cost=[]
            for b in budgets:
                q.append(module.reduce(data,b,.1,.1)[ar[:,None],ids]);cost.append(tree.prefix_evals(b))
            np.testing.assert_array_equal(cost[-1],tree.evals)
            stats=tree.stats();del data,tree
            stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_nonroot_nn_requests=bridge.queries)
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,z=z,q=np.array(q),ids=ids,mask=mask,evaluated_nodes=np.array(cost),stats=json.dumps(stats),
                game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]))
            tmp.replace(path);print('Adaptive root pilot',name,lo+n,'/',len(rows),stats['seconds'],flush=True)
    result=dict(positions=len(rows),variants=len(VARIANTS),seconds=time.monotonic()-start)
    atomic(out/'worker.json',result);return result
