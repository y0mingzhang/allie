"""Expanded August development trees: constant vs count-dependent backup."""
import json,time
from pathlib import Path
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .threadforest_native import load
from .adaptive_temperature_native import load as reducer_load
from .handles import HandleOracle

def run(oracle,spec):
    out=ROOT/'aug-expanded-search-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-tune-expanded-v1/sample.json';rows=json.loads(source.read_text())['positions']
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in ('adaptive_temperature.cpp','adaptive_temperature_native.py','threadforest.cpp','threadforest_native.py','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py')]]
    plan=dict(budgets=[256,1000],methods=dict(constant=[0,.1,16],subtree=[3,.2,16]),
        roots_per_batch=1024,threads=4,skip_forced=True,sample_sha256=digest(source),
        sources={p.name:digest(p) for p in files},
        semantics='All backups and budgets from same actual1000 root-coverage trees. Prefix budgets are logical-birth counterfactuals. Fit Elo alpha/beta with August fit-game CV; all six arms on game-disjoint confirmation. No golden tuning. Higher cost alone is not dominance.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    module=load();reducer=reducer_load();start=time.monotonic();stats=[]
    for lo in range(0,len(rows),plan['roots_per_batch']):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        path=out/f'{lo:06d}.npz';part=rows[lo:lo+plan['roots_per_batch']]
        if not path.exists():
            begin=time.monotonic();oracle.reset();n=len(part);ar=np.arange(n)
            budgets=[0 if len(r['legal'])==1 else 1000 for r in part]
            assert sum(budgets)+sum(len(r['prefix']) for r in part)<oracle.runner.max_total_num_tokens
            bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits
            tree=module.Tree([r['prefix'] for r in part],z,budgets,[2.5]*n,plan['threads'])
            while not tree.done:
                h=tree.select()
                if len(h):tree.update(bridge(h))
            k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool)
            for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
            data=tree.compact();qs=[];costs=[]
            for budget in plan['budgets']:
                qs.append(np.stack([reducer.reduce(data,budget,*cfg)[ar[:,None],ids] for cfg in plan['methods'].values()]))
                costs.append(tree.prefix_evals(budget))
            np.testing.assert_array_equal(qs[-1][0],module.reduce(data,1000,.1,.1)[ar[:,None],ids])
            np.testing.assert_array_equal(costs[-1],tree.evals)
            stat=tree.stats();del tree
            stat.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_nonroot_nn_requests=bridge.queries)
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,**data,z=z,q=np.array(qs),ids=ids,mask=mask,evaluated_nodes=np.array(costs),stats=json.dumps(stat),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
            tmp.replace(path);del data
            print('Adaptive deep',lo+n,'/',len(rows),stat['seconds'],flush=True)
        with np.load(path) as z:stats.append(json.loads(str(z['stats'])))
    report=dict(positions=len(rows),seconds=time.monotonic()-start,scoring_seconds=sum(s['seconds'] for s in stats),new_tokens=sum(s['new_tokens'] for s in stats))
    atomic(out/'worker.json',report);return report
