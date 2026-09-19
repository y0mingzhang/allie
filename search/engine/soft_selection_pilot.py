"""Matched-budget first-play-urgency and exploration-strength pilot."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic,inside
from .soft_selection_native import load,test

VARIANTS=[('soft_cp25',1,0.,2.5),('soft_cp5',1,0.,5.)]


def run(oracle,spec):
    module=load();test(module)
    source=ROOT/'aug-tune-v1/sample.json';rows=json.loads(source.read_text())['positions']
    out=inside(spec.get('output',ROOT/'aug-soft-selection-v1'));out.mkdir(exist_ok=True);bs=512;budgets=[64,256,1000]
    variants=spec.get('variants',VARIANTS)
    assert all(len(v)==4 and v[1] in (0,1,2) and 0<=v[2]<=1 and 0<v[3]<=10 for v in variants)
    sources=[Path(__file__),*[Path(__file__).with_name(s) for s in ('soft_selection.cpp','soft_selection_native.py','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','direct.py')]]
    plan=dict(variants=variants,budgets=budgets,roots_per_batch=bs,
        sample_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        stage='August model-training-seen development. Only tree allocation changes. Shared output calibration is independently fit on August fit folds. Selection uses incremental soft Bellman values at tau0.1. No next-move labels enter search.')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==json.loads(json.dumps(plan))
    else:atomic(out/'plan.json',plan)
    start=time.monotonic()
    # Live-GPU zero-mode control must reproduce the existing compact cache.
    if not (out/'identity.json').exists():
        part=rows[:bs];oracle.reset();z=oracle([r['prefix'] for r in part])
        t=module.Tree([r['prefix'] for r in part],z,[1000]*len(part),[1.25]*len(part),0,0.,False,.1)
        for _ in range(1000):
            p=t.select()
            if p:t.update(oracle(p))
        with np.load(ROOT/'aug-compact-v1/000000.npz') as ref:
            np.testing.assert_array_equal(z,ref['z'])
            for key,value in t.compact().items():np.testing.assert_array_equal(value,ref[key])
        del t;atomic(out/'identity.json',dict(equal=True,positions=bs,budget=1000,source=plan['sources']))
    for name,mode,reduction,cpuct in variants:
        folder=out/name;folder.mkdir(exist_ok=True)
        for lo in range(0,len(rows),bs):
            path=folder/f'{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            part=rows[lo:lo+bs];oracle.reset();begin=time.monotonic()
            z=oracle([r['prefix'] for r in part])
            tree=module.Tree([r['prefix'] for r in part],z,[1000]*len(part),[cpuct]*len(part),mode,reduction,True,.1)
            k=max(len(r['legal']) for r in part);ids=np.zeros((len(part),k),int);mask=np.zeros_like(ids,bool)
            for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
            values=[];cost=[]
            for step in range(1,1001):
                p=tree.select()
                if p:tree.update(oracle(p))
                if step in budgets:
                    q=np.zeros_like(ids,float)
                    for i,(moves,score) in enumerate(tree.backups([.1])[0]):
                        lookup=dict(zip(moves,score));q[i,:int(mask[i].sum())]=[lookup[a] for a in ids[i,mask[i]]]
                    values.append(q);cost.append(tree.evals)
            data=tree.compact();stats=tree.stats();del tree
            stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez_compressed(f,**data,z=z,q=np.array(values),ids=ids,mask=mask,evaluated_nodes=np.array(cost),
                    game=np.array([r['game'] for r in part]),stats=json.dumps(stats))
            tmp.replace(path)
            print('Selection pilot',name,lo+len(part),'/',len(rows),stats['seconds'],flush=True)
    result=dict(positions=len(rows),variants=len(variants),seconds=time.monotonic()-start)
    atomic(out/'worker.json',result);return result
