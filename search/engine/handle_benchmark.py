"""End-to-end handle bridge benchmark, same search and root order."""
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic,inside
from .handleforest_native import load
from .forest_native import load as load_prefix
from .handles import HandleOracle
from .balanced_eval import digest
from .fit_policy import loss_gradient

def run(oracle,spec):
    module=load();old=load_prefix();out=inside(spec.get('output',ROOT/'handle-benchmark-v1'));out.mkdir(exist_ok=True)
    sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    sources=[Path(__file__),*[Path(__file__).with_name(s) for s in ('handleforest.cpp','handleforest_native.py','handles.py','forest.cpp','forest_native.py','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','direct.py')]]
    plan=dict(sample_sha256=digest(sample),sources={p.name:digest(p) for p in sources},
        cases=[[512,1000],[160,4000]],methods=['prefix','handle'],repeats=2,
        stage='Same root-action forest, node budget and forced-move skip. Root order frozen, performance and numerical checks only.')
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:atomic(path,plan)
    fits=json.loads((ROOT/'aug-coverage-v1/results.json').read_text())['results']['coverage_bernoulli_1000_elo']['parameters']
    result={};comparisons={};begin=time.monotonic()
    for n,budget in plan['cases']:
        part=rows[:n];prefixes=[r['prefix'] for r in part];k=max(len(r['legal']) for r in part)
        ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);ar=np.arange(n);groups=np.array([r['cell']%4 for r in part]);target=np.array([r['legal'].index(r['target']) for r in part])
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
        budgets=[0 if len(r['legal'])==1 else budget for r in part]
        for rep in range(2):
            for method in (plan['methods'] if rep==0 else plan['methods'][::-1]):
                key=f'{n}-{budget}-{method}-{rep}';dest=out/(key+'.npz')
                if dest.exists():
                    with np.load(dest) as f:result[key]=json.loads(str(f['stats']))
                    continue
                if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
                oracle.reset();start=time.monotonic()
                if method=='handle':
                    bridge=HandleOracle(oracle,prefixes);z=bridge.root_logits
                    tree=module.Tree(prefixes,z,budgets,[2.5]*n);query=bridge
                else:
                    z=oracle(prefixes);tree=old.Tree(prefixes,z,budgets,[2.5]*n);query=oracle
                prefill_setup=time.monotonic()-start;selection=updating=oracle_wall=0.;rounds=0;unique=0
                while not tree.done:
                    tick=time.monotonic();p=tree.select();selection+=time.monotonic()-tick
                    if len(p):
                        before=oracle.next_row;tick=time.monotonic();pred=query(p);oracle_wall+=time.monotonic()-tick;unique+=oracle.next_row-before
                        tick=time.monotonic();tree.update(pred);updating+=time.monotonic()-tick
                    rounds+=1
                q=np.zeros((n,k))
                for i,(moves,score) in enumerate(tree.backups([.1])[0]):
                    lookup=dict(zip(moves,score));q[i,mask[i]]=[lookup[a] for a in ids[i,mask[i]]]
                stats=tree.stats();stats.update(seconds=time.monotonic()-start,prefill_setup=prefill_setup,
                    select_seconds=selection,update_seconds=updating,oracle_wall=oracle_wall,
                    forward_seconds=oracle.forward_seconds,new_tokens=oracle.new_tokens,forwards=oracle.calls,
                    rounds=rounds,unique_nonroot_nn_requests=unique,mean_nodes=float(np.mean(tree.evals)))
                nodes=np.array(tree.evals);del tree
                p=np.zeros_like(q);logits=z[:,378:2346][ar[:,None],ids].astype(float)
                for g in np.unique(groups):
                    select=groups==g;fit=fits[str(int(g))]
                    p[select],_=loss_gradient([fit['alpha'],fit['beta']],logits[select],q[select],mask[select],target[select],'forward',return_policy=True)
                temp=dest.with_suffix('.partial')
                with temp.open('wb') as f:np.savez_compressed(f,z=z,q=q,policy=p,ids=ids,mask=mask,nodes=nodes,stats=json.dumps(stats))
                temp.replace(dest);result[key]=stats;print('Handle benchmark',key,stats,flush=True)
        for rep in range(2):
            with np.load(out/f'{n}-{budget}-prefix-{rep}.npz') as a,np.load(out/f'{n}-{budget}-handle-{rep}.npz') as b:
                pa,pb=a['policy'],b['policy'];kl=np.sum(np.where(mask,pa*np.log(np.maximum(pa,1e-300)/np.maximum(pb,1e-300)),0),axis=1)
                comparisons[f'{n}-{budget}-{rep}']=dict(speedup=result[f'{n}-{budget}-prefix-{rep}']['seconds']/result[f'{n}-{budget}-handle-{rep}']['seconds'],
                    max_policy_abs=float(np.max(np.abs(pa-pb))),mean_kl=float(kl.mean()),max_kl=float(kl.max()),
                    mean_ce_delta=float(np.mean(np.log(pa[ar,target]/pb[ar,target]))),max_root_logit_abs=float(np.max(np.abs(a['z']-b['z']))),
                    max_q_abs=float(np.max(np.abs(a['q']-b['q']))),max_node_difference=int(np.max(np.abs(a['nodes']-b['nodes']))))
    report=dict(measurements=result,comparisons=comparisons,seconds=time.monotonic()-begin,quality_claim=False)
    atomic(out/'results.json',report);print('Handle comparisons',comparisons,flush=True);return report
