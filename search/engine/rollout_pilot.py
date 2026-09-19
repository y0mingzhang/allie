"""Human-policy rollout critic: four trajectories per legal root move."""
import json,time
from pathlib import Path
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .rollout_native import load
from .handles import HandleOracle

def run(oracle,spec):
    module=load();out=ROOT/'aug-rollout-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-tune-v1/sample.json';rows=json.loads(source.read_text())['positions']
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in ('rollout.cpp','rollout_native.py','handles.py','board.cpp','mcts_native.hpp','direct.py')]]
    plan=dict(variants=[['human_rollout',4]],budgets=[1,4,8,16],budget_unit='ply horizon, not simulation count',
        roots_per_batch=256,replicas=4,seed=1926781,sample_sha256=digest(source),sources={p.name:digest(p) for p in files},
        method='Each legal root action once, then4 independent human-policy continuations under unchanged Elo/TC headers. Categorical legal-policy sampling on both sides, fixed counter RNG by root/action/replica/depth. Leaf WDL transformed to ROOT mover value. Exact terminal outcomes replace WDL. Forced root move skips all search; context-limit trajectories freeze last critic.',
        selection='Fit global/Elo alpha,beta on August fold0 with game CV; all horizon arms reported on fold1. No golden until frozen. Empirical rollout variance reported separately.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    folder=out/'human_rollout';folder.mkdir(exist_ok=True);start=time.monotonic();stats=[]
    for lo in range(0,len(rows),plan['roots_per_batch']):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        path=folder/f'{lo:06d}.npz';part=rows[lo:lo+plan['roots_per_batch']]
        if not path.exists():
            oracle.reset();begin=time.monotonic();prefixes=[r['prefix'] for r in part]
            bridge=HandleOracle(oracle,prefixes);z=bridge.root_logits;n=len(part);ar=np.arange(n);k=max(len(r['legal']) for r in part)
            ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool)
            for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
            tree=module.Rollouts(prefixes,z,list(range(lo,lo+n)),plan['replicas'],max(plan['budgets']),plan['seed'])
            qs=[];variances=[];costs=[]
            for dep in range(1,max(plan['budgets'])+1):
                h=tree.select()
                if len(h):tree.update(bridge(h))
                if dep in plan['budgets']:
                    snap=tree.snapshot();qs.append(snap['q'][ar[:,None],ids]);variances.append(snap['variance'][ar[:,None],ids]);costs.append(np.array(tree.evals))
            assert bridge.queries==sum(tree.evals)
            stat=dict(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_nonroot_nn_requests=bridge.queries)
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,z=z,q=np.array(qs),variance=np.array(variances),ids=ids,mask=mask,evaluated_nodes=np.array(costs),stats=json.dumps(stat),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
            tmp.replace(path);print('Human rollouts',lo+n,'/',len(rows),stat,flush=True)
        with np.load(path) as z:stats.append(json.loads(str(z['stats'])))
    report=dict(positions=len(rows),invocation_seconds=time.monotonic()-start,scoring_seconds=sum(x['seconds'] for x in stats),new_tokens=sum(x['new_tokens'] for x in stats),nonroot_nodes=sum(x['unique_nonroot_nn_requests'] for x in stats))
    atomic(out/'worker.json',report);return report
