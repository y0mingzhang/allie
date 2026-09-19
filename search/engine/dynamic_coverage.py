"""Execute the already frozen coverage router with actual mixed budgets."""
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT, GLOBAL_STOP, atomic, digest
from .coverage_native import load
from .fit_policy import loss_gradient
from .golden_metrics import summarize, controls

OUT = ROOT / 'golden-dynamic-coverage-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    parent = ROOT / 'golden-coverage-v1/plan.json'; p = json.loads(parent.read_text())
    plan = dict(budgets=p['methods']['coverage_routed'], parameters=p['parameters'],
        parent_sha256=digest(parent), sample_sha256=p['sample_sha256'], laws_sha256=p['laws_sha256'], roots_per_block=512,
        sources={f.name:digest(f) for f in [Path(__file__), *[Path(__file__).with_name(s) for s in
            ('coverage.cpp','coverage_native.py','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','direct.py','golden_metrics.py','fit_policy.py')]]},
        stage='Actual execution of already frozen/reported root-coverage Elo stopping; no new parameter selection. Cached/live differences reported, live result authoritative. Reused golden sample.')
    assert plan['sample_sha256'] == digest(ROOT/'golden-balanced-v1/sample.json')
    assert plan['laws_sha256'] == digest(ROOT/'training-cm-laws.json')
    path=OUT/'plan.json'
    if path.exists(): assert json.loads(path.read_text()) == plan
    else: atomic(path,plan)
    return plan


def run(oracle,spec):
    plan=freeze();module=load();rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];bs=plan['roots_per_block'];begin=time.monotonic()
    for lo in range(0,len(rows),bs):
        path=OUT/f'{lo:06d}.npz'
        if path.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);ar=np.arange(n);groups=np.array([r['cell']%4 for r in part])
        budgets=[plan['budgets'][g] for g in groups];oracle.reset();start=time.monotonic();z=oracle([r['prefix'] for r in part])
        tree=module.Tree([r['prefix'] for r in part],z,budgets,[2.5]*n,0,0.,2.);unique=0
        for _ in range(max(budgets)):
            prefixes=tree.select()
            if prefixes:
                before=oracle.next_row;tree.update(oracle(prefixes));unique+=oracle.next_row-before
        k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int);q=np.zeros((n,k))
        for i,((moves,values),r) in enumerate(zip(tree.backups([.1])[0],part)):
            ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
            lookup=dict(zip(moves,values));q[i,mask[i]]=[lookup[a] for a in ids[i,mask[i]]]
        nodes=np.array(tree.evals);stats=tree.stats();del tree;assert nodes.sum()==stats['evaluated_leaves']
        logits=z[:,378:2346][ar[:,None],ids].astype(float);p=np.zeros_like(q)
        for g in np.unique(groups):
            selected=groups==g;f=plan['parameters'][str(plan['budgets'][g])][str(int(g))]
            p[selected],_=loss_gradient([f['alpha'],f['beta']],logits[selected],q[selected],mask[selected],target[selected],'forward',return_policy=True)
        predicted=np.where(mask&(p==p.max(1)[:,None]),ids,1968).min(1)
        score=np.stack([-np.log(p[ar,target]),predicted==ids[ar,target],p.max(1)],axis=1)
        stats.update(seconds=time.monotonic()-start,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,unique_nonroot_nn_requests=unique)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,score=score,policy=p,ids=ids,mask=mask,z=z,nodes=nodes,stats=json.dumps(stats),game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]))
        tmp.replace(path);print('Dynamic coverage',lo+n,'/',len(rows),stats['seconds'],flush=True)
    scores,costs=controls(rows);live=[];cached=[];fixed=[];live_nodes=[];cached_nodes=[];fixed_nodes=[];stats=[];drift=[]
    for lo in range(0,len(rows),bs):
        with np.load(OUT/f'{lo:06d}.npz') as z:
            part=rows[lo:lo+len(z['game'])];assert list(z['game'])==[r['game'] for r in part]
            np.testing.assert_array_equal(z['ply'],[r['ply'] for r in part])
            live.append(z['score']);live_nodes.append(z['nodes']);stats.append(json.loads(str(z['stats'])))
            live_p=z['policy'];ids=z['ids'];mask=z['mask']
        with np.load(ROOT/f'golden-coverage-v1/{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in part];np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(mask,z['mask'])
            cached.append(z['coverage_routed']);cached_nodes.append(z['coverage_routed_nodes'])
            fixed.append(z['coverage_fixed']);fixed_nodes.append(z['coverage_fixed_nodes'])
            ar=np.arange(len(part));groups=np.array([r['cell']%4 for r in part]);target=np.array([r['legal'].index(r['target']) for r in part]);lp=z['z'][:,378:2346][ar[:,None],ids].astype(float)
            cp=np.zeros_like(lp)
            for g in np.unique(groups):
                select=groups==g;b=plan['budgets'][g];f=plan['parameters'][str(b)][str(int(g))]
                cp[select],_=loss_gradient([f['alpha'],f['beta']],lp[select],z['q'][[256,1000].index(b),select],mask[select],target[select],'forward',return_policy=True)
            for a,b,m in zip(cp,live_p,mask):
                a=a[m];b=b[m];drift.append([np.max(np.abs(a-b)),np.sum(np.abs(a-b)),np.sum(a*np.log(a/b))])
    for name,sc,nc in [('dynamic',live,live_nodes),('cached_router',cached,cached_nodes),('coverage_fixed',fixed,fixed_nodes)]:scores[name]=np.concatenate(sc);costs[name]=np.concatenate(nc)
    analysis=time.monotonic();result=summarize(rows,scores,costs,references=('legal','four_ply','soft1000','cached_router'))
    d=np.array(drift)
    report=dict(methods=result,positions=len(rows),plan_sha256=digest(OUT/'plan.json'),
        scoring_seconds=sum(s['seconds'] for s in stats),analysis_seconds=time.monotonic()-analysis,
        new_tokens=sum(s['new_tokens'] for s in stats),unique_nonroot_nn_requests=sum(s['unique_nonroot_nn_requests'] for s in stats),
        cached_live_policy_difference=dict(max_abs=float(d[:,0].max()),max_l1=float(d[:,1].max()),mean_kl=float(d[:,2].mean()),max_kl=float(d[:,2].max())),
        caveat=plan['stage']+' CM conditional on frozen training-law shape. Paired bootstrap excludes law-fit and accumulated benchmark-reuse uncertainty.')
    atomic(OUT/'results.json',report)
    print('Dynamic coverage result',result['dynamic']['macro'],result['dynamic']['expert_macro'],report['cached_live_policy_difference'],flush=True)
    return dict(positions=len(rows),seconds=time.monotonic()-begin)
