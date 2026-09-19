"""Benchmark independent root-action scheduling without selecting on golden."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic,inside
from .forest_native import load
from .fit_policy import loss_gradient

def run(oracle,spec):
    module=load()
    out=inside(spec.get('output',ROOT/'forest-benchmark-v1'));out.mkdir(exist_ok=True)
    sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    sources=[Path(__file__),*[Path(__file__).with_name(s) for s in ('forest.cpp','forest_native.py','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','direct.py')]]
    plan=dict(sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        cases=[[512,1000],[160,4000]],methods=['sequential','forest','sequential_skip_forced','forest_skip_forced'],
        stage='Timing and numerical equivalence only; same position order. Skip forced is an exact policy shortcut available to both methods. No golden selection.')
    atomic(out/'plan.json',plan);measurements={};begin=time.monotonic()
    for n,budget in plan['cases']:
        part=rows[:n];prefixes=[r['prefix'] for r in part];k=max(len(r['legal']) for r in part)
        ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool)
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
        for method in plan['methods']:
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            key=f'{n}-{budget}-{method}';path=out/(key+'.npz')
            if path.exists():
                with np.load(path) as saved:measurements[key]=json.loads(str(saved['stats']))
                continue
            budgets=[0 if 'skip_forced' in method and len(r['legal'])==1 else budget for r in part]
            oracle.reset();start=time.monotonic();z=oracle(prefixes)
            prefill=time.monotonic()-start
            start_tree=time.monotonic()
            parallel=method.startswith('forest')
            tree=module.Tree(prefixes,z,budgets,[2.5]*n) if parallel else module.Reference(prefixes,z,budgets,[2.5]*n,0,0.,2.)
            setup=time.monotonic()-start_tree;selection=updating=oracle_wall=0.;rounds=0;unique=0
            while not tree.done if parallel else rounds<max(budgets):
                tick=time.monotonic();p=tree.select();selection+=time.monotonic()-tick
                if p:
                    before=oracle.next_row;tick=time.monotonic();pred=oracle(p);oracle_wall+=time.monotonic()-tick
                    unique+=oracle.next_row-before;tick=time.monotonic();tree.update(pred);updating+=time.monotonic()-tick
                rounds+=1
            tick=time.monotonic();q=np.zeros((n,k))
            for i,(moves,score) in enumerate(tree.backups([.1])[0]):
                lookup=dict(zip(moves,score));q[i,mask[i]]=[lookup[a] for a in ids[i,mask[i]]]
            reduction=time.monotonic()-tick
            stats=tree.stats();stats.update(seconds=time.monotonic()-start,prefill=prefill,setup=setup,
                select_seconds=selection,update_seconds=updating,oracle_wall=oracle_wall,reduction_seconds=reduction,
                forward_seconds=oracle.forward_seconds,new_tokens=oracle.new_tokens,forwards=oracle.calls,
                rounds=rounds,unique_nonroot_nn_requests=unique,mean_nodes=float(np.mean(tree.evals)))
            nodes=np.array(tree.evals);del tree
            temp=path.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,z=z,q=q,ids=ids,mask=mask,nodes=nodes,stats=json.dumps(stats))
            temp.replace(path);measurements[key]=stats;print('Forest benchmark',key,stats,flush=True)
    comparisons={}
    # Fixed coefficients from the existing 1000-sim August fit, for a numerical
    # sensitivity diagnostic only. No labels are used by the scheduler.
    fits=json.loads((ROOT/'aug-coverage-v1/results.json').read_text())['results']['coverage_bernoulli_1000_elo']['parameters']
    for n,budget in plan['cases']:
        policies={};root_z={};q_values={}
        part=rows[:n];ar=np.arange(n);groups=np.array([r['cell']%4 for r in part]);target=np.array([r['legal'].index(r['target']) for r in part])
        for method in plan['methods']:
            key=f'{n}-{budget}-{method}'
            with np.load(out/(key+'.npz')) as saved:
                logits=saved['z'][:,378:2346][ar[:,None],saved['ids']].astype(float)
                p=np.zeros_like(logits);q=saved['q'];mask=saved['mask'];q_values[method]=q;root_z[method]=saved['z']
                for g in np.unique(groups):
                    select=groups==g;fit=fits[str(int(g))]
                    p[select],_=loss_gradient([fit['alpha'],fit['beta']],logits[select],q[select],mask[select],target[select],'forward',return_policy=True)
                policies[method]=p
        for a,b in [('sequential','forest'),('sequential_skip_forced','forest_skip_forced'),('sequential','sequential_skip_forced')]:
            pa,pb=policies[a],policies[b];kl=np.sum(np.where(mask,pa*np.log(np.maximum(pa,1e-300)/np.maximum(pb,1e-300)),0),axis=1)
            comparisons[f'{n}-{budget}-{a}-vs-{b}']=dict(speedup=measurements[f'{n}-{budget}-{a}']['seconds']/measurements[f'{n}-{budget}-{b}']['seconds'],
                max_policy_abs=float(np.max(np.abs(pa-pb))),mean_kl=float(kl.mean()),max_kl=float(kl.max()),
                mean_ce_delta=float(np.mean(np.log(pa[ar,target]/pb[ar,target]))),
                max_root_logit_abs=float(np.max(np.abs(root_z[a]-root_z[b]))),
                max_q_abs=float(np.max(np.abs(q_values[a]-q_values[b]))))
    result=dict(measurements=measurements,comparisons=comparisons,seconds=time.monotonic()-begin,quality_claim=False)
    atomic(out/'results.json',result);print('Forest comparisons',comparisons,flush=True);return result
