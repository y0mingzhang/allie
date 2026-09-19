"""Execute the frozen cap768 policy with actual per-root stopping budgets."""
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT,GLOBAL_STOP,atomic,digest
from .backup_native import load
from .fit_policy import loss_gradient

OUT=ROOT/'golden-dynamic-router-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    parent=ROOT/'golden-router-v1/plan.json';p=json.loads(parent.read_text())
    plan=dict(method=p['methods']['cap768'],parent_plan_sha256=digest(parent),
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),laws_sha256=digest(ROOT/'training-cm-laws.json'),
        sources={f.name:digest(f) for f in (Path(__file__),Path(__file__).with_name('analyze_dynamic_router.py'),Path(__file__).with_name('direct.py'),Path(__file__).with_name('backups.cpp'))},
        roots_per_block=512,stage='Execution validation of already frozen/reported cap768; no retuning and no fresh-holdout claim. Mixed budgets executed directly, not inferred from fully grown trees.')
    assert p['sample_sha256']==plan['sample_sha256'] and p['laws_sha256']==plan['laws_sha256']
    dest=OUT/'plan.json'
    if dest.exists():assert json.loads(dest.read_text())==plan
    else:atomic(dest,plan)
    return plan


def run(oracle,spec):
    plan=freeze();module=load();rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];bs=plan['roots_per_block'];start=time.monotonic()
    for lo in range(0,len(rows),bs):
        dest=OUT/f'{lo:06d}.npz'
        if dest.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);ar=np.arange(n);groups=np.array([r['cell']%4 for r in part])
        budgets=[plan['method']['budgets_by_elo'][g] for g in groups]
        oracle.reset();begin=time.monotonic();root=oracle([r['prefix'] for r in part])
        unique_requests=0
        tree=module.Tree([r['prefix'] for r in part],root,budgets,[1.25]*n)
        for _ in range(max(budgets)):
            prefixes=tree.select()
            if prefixes:
                before=oracle.next_row;tree.update(oracle(prefixes));unique_requests+=oracle.next_row-before
        k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int);q=np.zeros((n,k))
        for i,((moves,values),r) in enumerate(zip(tree.backups([.1])[0],part)):
            ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
            lookup=dict(zip(moves,values));q[i,mask[i]]=[lookup[a] for a in ids[i,mask[i]]]
        nodes=np.array(tree.evals);stats=tree.stats();del tree
        assert nodes.sum()==stats['evaluated_leaves']
        z=root[:,378:2346][ar[:,None],ids].astype(float);p=np.zeros_like(q)
        for g in np.unique(groups):
            selected=groups==g;f=plan['method']['parameters'][str(int(g))]
            p[selected],_=loss_gradient([f['alpha'],f['beta']],z[selected],q[selected],mask[selected],target[selected],'forward',return_policy=True)
        prediction=np.where(mask&(p==p.max(1)[:,None]),ids,1968).min(1)
        score=np.stack([-np.log(p[ar,target]),prediction==ids[ar,target],p.max(1)],axis=1)
        stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds,
                     unique_nonroot_nn_requests=unique_requests)
        tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,score=score,policy=p,ids=ids,mask=mask,root=root,nodes=nodes,stats=json.dumps(stats),game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]))
        tmp.replace(dest)
        print('Dynamic router',lo+n,'/',len(rows),stats['seconds'],flush=True)
    from .analyze_dynamic_router import main
    main()
    return dict(positions=len(rows),elapsed_seconds=time.monotonic()-start)
