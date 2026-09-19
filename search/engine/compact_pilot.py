"""Cache model-evaluated trees once; subsequent backup ideas require no GPU."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic,inside
from .compact_native import load,test


def run(oracle,spec):
    module=load();test(module)
    sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    out=inside(spec['output']);out.mkdir(exist_ok=True)
    bs=int(spec.get('roots_per_batch',512));budget=int(spec.get('budget',1000))
    sources=[Path(__file__),*[Path(__file__).with_name(s) for s in
        ('compact.cpp','compact_native.py','backups.cpp','mcts_native.hpp','board.cpp','direct.py')]]
    plan=dict(spec=spec,sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        export=json.loads((ROOT/'serving-export/provenance.json').read_text()),
        stage='August training-seen development only. No labels enter inference. Full compact trees retained for CPU-only counterfactual backups.')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:atomic(out/'plan.json',plan)
    start=time.monotonic()
    for lo in range(0,len(rows),bs):
        path=out/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];oracle.reset();begin=time.monotonic()
        z=oracle([r['prefix'] for r in part]);tree=module.Tree([r['prefix'] for r in part],z,[budget]*len(part),[1.25]*len(part))
        snapshots={};cost=[]
        budgets=[b for b in (16,64,256,1000,4000) if b<=budget]
        assert budgets[-1]==budget
        for step in range(1,budget+1):
            prefixes=tree.select()
            if prefixes:tree.update(oracle(prefixes))
            if step in budgets:
                cost.append(tree.evals)
                if lo==0:snapshots[step]=tree.backups([.1])
        data=tree.compact();stats=tree.stats();del tree
        # Validate snapshot reconstruction on real neural predictions too.
        for b,reference in snapshots.items():
            q=module.reduce(data,b,.1,.1)
            for i,(ids,values) in enumerate(reference[0]):
                np.testing.assert_allclose(q[i,ids],values,atol=3e-12,rtol=0)
        stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,
                     forward_seconds=oracle.forward_seconds)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:
            np.savez_compressed(f,**data,z=z,evaluated_nodes=np.array(cost),budgets=budgets,
                game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]),stats=json.dumps(stats))
        tmp.replace(path)
        print('Compact tree cache',lo+len(part),'/',len(rows),stats,flush=True)
    result=dict(positions=len(rows),budget=budget,roots_per_batch=bs,seconds=time.monotonic()-start)
    atomic(out/'worker.json',result);return result
