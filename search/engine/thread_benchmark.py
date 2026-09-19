"""Choose CPU expansion threads from timing, requiring bit-exact model results."""
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic
from .threadforest_native import load
from .handles import HandleOracle
from .balanced_eval import digest

def run(oracle,spec):
    module=load();out=ROOT/'thread-benchmark-v1';out.mkdir(exist_ok=True)
    sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    plan=dict(cases=[[512,1000],[160,4000]],threads=[1,2,4],sample_sha256=digest(sample),
        sources={p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(s) for s in
        ('threadforest.cpp','threadforest_native.py','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py')]]})
    atomic(out/'plan.json',plan);records={};start=time.monotonic()
    for n,budget in plan['cases']:
        part=rows[:n];prefixes=[r['prefix'] for r in part];budgets=[0 if len(r['legal'])==1 else budget for r in part]
        ref=None
        for threads in plan['threads']:
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            oracle.reset();tick=time.monotonic();bridge=HandleOracle(oracle,prefixes);z=bridge.root_logits
            tree=module.Tree(prefixes,z,budgets,[2.5]*n,threads);select=update=nn=0.
            while not tree.done:
                t=time.monotonic();p=tree.select();select+=time.monotonic()-t
                if len(p):
                    t=time.monotonic();v=bridge(p);nn+=time.monotonic()-t
                    t=time.monotonic();tree.update(v);update+=time.monotonic()-t
            q=np.concatenate([values for moves,values in tree.backups([.1])[0]]);nodes=np.array(tree.evals)
            stats=tree.stats();stats.update(seconds=time.monotonic()-tick,select_seconds=select,update_seconds=update,
                oracle_wall=nn,forward_seconds=oracle.forward_seconds,forwards=oracle.calls,new_tokens=oracle.new_tokens)
            del tree
            if ref is None:ref=(z,q,nodes)
            else:
                for a,b in zip(ref,(z,q,nodes)):np.testing.assert_array_equal(a,b)
            records[f'{n}-{budget}-t{threads}']=stats
            print('Thread benchmark',n,budget,threads,stats,flush=True)
    result=dict(measurements=records,identity=True,seconds=time.monotonic()-start)
    atomic(out/'results.json',result);return result
