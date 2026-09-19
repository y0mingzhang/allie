"""Private balanced-development comparison, shared MCTS budget ladder and 4ply."""
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,inside,atomic,GLOBAL_STOP
from .distributional_native import load,test


def run(oracle,spec):
    from . import tree as tree_module
    tree_module=importlib.reload(tree_module)  # Adds per-root counts; same numerical path.
    module=load();test(module)
    source=inside(spec['input']);sample=json.loads(source.read_text());rows=sample['positions']
    out=inside(spec['output']);out.mkdir(exist_ok=True)
    budgets=[16,64,256,1000];taus=[float('inf'),.5,.1,.025,0.]
    names=['mcts_average','human_expectation','soft_0.5','soft_0.1','soft_0.025','minimax']
    paths=[Path(__file__),Path(tree_module.__file__),*[Path(__file__).with_name(n) for n in ('distributional.cpp','distributional_native.py','backups.cpp','mcts_native.hpp','board.cpp')]]
    plan=dict(spec=spec,budgets=budgets,backup_names=names,tree_widths=[4,2,2],
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        export=json.loads((ROOT/'serving-export/provenance.json').read_text()),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        stage='Development only. All methods use fold0 for calibration; fold1 reports every arm. Golden and test untouched. No human target enters inference.')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:atomic(out/'plan.json',plan)
    start=time.monotonic()
    for kind,block in [('tree',32),('mcts',128)]:
        for lo in range(0,len(rows),block):
            dest=out/f'{kind}-{lo:06d}.npz'
            if dest.exists():continue
            if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():raise RuntimeError('STOP')
            part=rows[lo:lo+block];oracle.reset();begin=time.monotonic()
            ids=np.zeros((len(part),max(len(r['legal']) for r in part)),np.int32)
            mask=np.zeros_like(ids,bool)
            for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
            index=np.arange(len(part))[:,None]
            if kind=='tree':
                q,z,stats=tree_module.batch(part,oracle,widths=(4,2,2),batch_size=1024)
                q=q[:,index,ids]
                nodes=np.cumsum(np.array(stats['nodes_by_depth_and_root']),axis=0)
                extra={}
            else:
                z=oracle([r['prefix'] for r in part])
                tree=module.Tree([r['prefix'] for r in part],z,[budgets[-1]]*len(part),[1.25]*len(part))
                values=[];visits=[];nodes=[];elapsed=[];distributions=[]
                for step in range(1,budgets[-1]+1):
                    prefixes=tree.select()
                    if prefixes:tree.update(oracle(prefixes))
                    if step not in budgets:continue
                    lookup=[{int(a):j for j,a in enumerate(row[:int(m.sum())])} for row,m in zip(ids,mask)]
                    v=np.zeros_like(ids,np.int32);a=np.zeros((len(names),*ids.shape),np.float32)
                    for i,(moves,counts,scores,prior) in enumerate(tree.snapshot()):
                        ix=[lookup[i][m] for m in moves];a[0,i,ix]=scores;v[i,ix]=counts
                    for k,snap in enumerate(tree.backups(taus),1):
                        for i,(moves,scores) in enumerate(snap):a[k,i,[lookup[i][m] for m in moves]]=scores
                    wdl=np.zeros((*ids.shape,3),np.float32)
                    for i,(moves,probs) in enumerate(tree.distribution()):wdl[i,[lookup[i][m] for m in moves]]=probs
                    distributions.append(wdl)
                    values.append(a);visits.append(v);nodes.append(tree.evals);elapsed.append(time.monotonic()-begin)
                q=np.array(values);nodes=np.array(nodes);stats=tree.stats()
                assert nodes[-1].sum()==stats['evaluated_leaves'];del tree
                extra=dict(visits=np.array(visits),elapsed=np.array(elapsed),wdl=np.array(distributions))
            stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
            tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,q=q,root=z,ids=ids,mask=mask,evaluated_nodes=nodes,
                game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]),stats=json.dumps(stats),**extra)
            tmp.replace(dest)
            if lo%(block*4)==0:print('Balanced dev',kind,lo+len(part),'/',len(rows),flush=True)
    return dict(stage='Private balanced development caches complete',positions=len(rows),elapsed_seconds=time.monotonic()-start)
