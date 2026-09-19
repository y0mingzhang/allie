"""Bounded tactical extensions as a value residual on the frozen current policy."""
import json,time,os
from pathlib import Path
import numpy as np
from .service import ROOT,atomic,GLOBAL_STOP
from .balanced_eval import digest
from .sample_data import read
from .tactical_native import load,test
from .residual_screen import analyze

def run(oracle,spec):
    m=load();test(m);start=time.monotonic();out=ROOT/'aug-tactical-v1';out.mkdir(exist_ok=True)
    d=read('aug-tune-expanded-v1');rows,games,ids,mask=(d[k] for k in ('rows','games','ids','mask'));n,k=mask.shape
    plan=dict(sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),sources={f:digest(Path(__file__).with_name(f)) for f in ('tactical_pilot.py','tactical.cpp','tactical_native.py','board.cpp','mcts_native.hpp','residual_screen.py','direct.py')},
        budget=256,max_depth=6,noncheck_branches=2,roots_per_block=256,temperatures=['expectation',.2,.05,'max'],
        semantics='All legal root actions first. Thereafter only captures/promotions (top2 by neural prior), or all evasions while in check. Best-first queue by sqrt(rootprior)*path probability; up to4 new leaves/root each round. Maximum256 nonterminal NN nodes/root total,6plies. No material evaluation or externalengine. Missing edges retain own node critic.',
        correction='Quiescent root-action Q minus independently evaluated one-ply Q, a feature tilting the frozen1000 policy. Global and Elo scalars fit insideAugust game folds; originaloneply control. Every extraquery and rootprefill charged. CM not computed forAugust.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    for lo in range(0,n,plan['roots_per_block']):
        path=out/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+plan['roots_per_block']];begin=time.monotonic();oracle.reset();z=oracle([r['prefix'] for r in part]);ptokens=oracle.new_tokens
        t=m.Tree([r['prefix'] for r in part],z,plan['budget'],plan['max_depth'],plan['noncheck_branches'])
        p=t.select()
        if p:t.update(oracle(p))
        idx=ids[lo:lo+len(part)].astype(int);ar=np.arange(len(part))[:,None]
        initial=t.q(np.inf)[ar,idx];initial_nodes=np.array(t.evals)
        while not t.done:
            ps=t.select()
            if ps:t.update(oracle(ps))
        qs=np.stack([t.q(tau)[ar,idx] for tau in [np.inf,.2,.05,0.]])
        nodes=np.array(t.evals);assert (nodes<=256).all()
        stats=dict(job=os.environ.get('SLURM_JOB_ID'),device=str(oracle.runner.device),seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,prefill_tokens=ptokens,forward_seconds=oracle.forward_seconds,nonterminal_queries=int(nodes.sum()))
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,q=qs,initial=initial,initial_nodes=initial_nodes,nodes=nodes,stats=json.dumps(stats),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
        tmp.replace(path);print('tactical',lo+len(part),n,stats['seconds'],float(nodes.mean()),flush=True)
    qs=np.zeros((4,n,k));initial=np.zeros((n,k));nodes=np.zeros(n);initial_nodes=np.zeros(n);stats=[]
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
            qs[:,lo:hi]=f['q'];initial[lo:hi]=f['initial'];nodes[lo:hi]=f['nodes'];initial_nodes[lo:hi]=f['initial_nodes'];stats.append(json.loads(str(f['stats'])))
    features=dict(original_oneply=initial)
    for i,name in enumerate(['expectation','.2','.05','max']):features['tactical_'+name]=qs[i]-initial
    worker=dict(seconds=time.monotonic()-start,blocks=stats,plan_sha256=digest(pp));atomic(out/'worker.json',worker)
    costs={name:dict(extra_nodes=nodes+1,summary=dict(mean_extra_nonroot_nodes=float(nodes.mean()),extra_full_prefixes=1,seconds=sum(s['seconds'] for s in stats),prefill_tokens=sum(s['prefill_tokens'] for s in stats))) for name in features}
    costs['original_oneply']=dict(extra_nodes=initial_nodes+1,summary=dict(mean_extra_nonroot_nodes=float(initial_nodes.mean()),extra_full_prefixes=1))
    result=analyze('aug-tactical-v1',features,costs,plan['correction'])
    return dict(selected=result['fit_cv_selected'],worker=worker)
