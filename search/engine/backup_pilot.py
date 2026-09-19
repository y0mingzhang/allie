"""Equal-tree-cost backup ablation and shared-prefix MCTS node-budget ladder."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .backup_native import load, test
from .service import ROOT,inside,atomic,GLOBAL_STOP


def run(oracle,spec):
    module=load();test(module)
    source=ROOT/'dev.json';rows=json.loads(source.read_text())['positions']
    out=inside(spec['output']);out.mkdir(exist_ok=True)
    budgets=[16,64,256,1000]
    temperatures=[float('inf'),.5,.1,.025,0.]
    labels=['mcts_average','human_expectation','soft_0.5','soft_0.1','soft_0.025','minimax']
    paths=[Path(__file__),Path(__file__).with_name('backup_native.py'),Path(__file__).with_name('backups.cpp'),Path(__file__).with_name('mcts_native.hpp')]
    plan=dict(spec=spec,budgets=budgets,backup_names=labels,dev_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        semantics='All backup variants reuse exactly the same repaired-MCTS tree. Unexpanded actions retain the node critic; soft values are tau*log E_prior exp(Q/tau). Snapshot budgets share the identical first simulations.',
        stage='Development only; no human label or actual future enters search or allocation.')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:atomic(out/'plan.json',plan)
    block=128;start=time.monotonic()
    for lo in range(0,len(rows),block):
        dest=out/f'{lo:06d}.npz'
        if dest.exists():continue
        if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():raise RuntimeError('STOP')
        part=rows[lo:lo+block];oracle.reset();begin=time.monotonic()
        z=oracle([r['prefix'] for r in part])
        tree=module.Tree([r['prefix'] for r in part],z,[budgets[-1]]*len(part),[1.25]*len(part))
        values=[];visits=[];evals=[];times=[]
        for step in range(1,budgets[-1]+1):
            prefixes=tree.select()
            if prefixes:tree.update(oracle(prefixes))
            if step not in budgets:continue
            q=np.zeros((len(labels),len(part),1968),np.float32)
            v=np.zeros((len(part),1968),np.int32)
            for i,(ids,counts,scores,prior) in enumerate(tree.snapshot()):q[0,i,ids]=scores;v[i,ids]=counts
            for k,snapshot in enumerate(tree.backups(temperatures),1):
                for i,(ids,scores) in enumerate(snapshot):q[k,i,ids]=scores
            values.append(q);visits.append(v);evals.append(tree.evals);times.append(time.monotonic()-begin)
        stats=tree.stats();assert sum(evals[-1])==stats['evaluated_leaves'];del tree
        stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,
            timing_note='Includes all five alternative backups at all four snapshots; standalone method wall time must be benchmarked separately.')
        tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,q=np.array(values),visits=np.array(visits),evaluated_nodes=np.array(evals),
            elapsed=np.array(times),root=z,game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]),stats=json.dumps(stats))
        tmp.replace(dest)
        print('Backup pilot',lo+len(part),'/',len(rows),flush=True)
    return dict(stage='Development backup and budget ladder caches complete',positions=len(rows),elapsed_seconds=time.monotonic()-start)
