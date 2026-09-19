"""Preregistered batch/order numerical audit, not a method-selection experiment."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from search.engine.service import ROOT,atomic
from search.engine.balanced_eval import digest
from search.engine.analyze_balanced import cellmean,bootstrap_deltas
from search.engine.growforest_native import load
from search.engine.scaled_count_native import load as backup_load
from search.engine.budget_policy import policy
from .collect import inventory,freeze,OUT,BUDGETS
from .handles import ShipHandles


def run(oracle,spec):
    size=spec['model'];plan=freeze();rows,feat,hashes=inventory('gold');tag=spec.get('audit_tag','');assert not tag or tag.replace('-','').isalnum()
    out=OUT/(f'order-{size}'+('-'+tag if tag else ''));out.mkdir(exist_ok=True)
    idx=np.array([i for cell in range(16) for i in [j for j,r in enumerate(rows) if r['cell']==cell][:64]])
    rows=[rows[i] for i in idx];feat=[feat[i] for i in idx];n=len(rows);k=max(len(r['legal']) for r in rows)
    ids=np.zeros((n,k),np.int32);mask=np.zeros((n,k),bool)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
    frozen=dict(sample=hashes,indices=idx.tolist(),seed=918225,block=128,budgets=BUDGETS,source=digest(Path(__file__)),plan=digest(OUT/'plan.json'),note='Two deterministic orders of1024 score-independently selected golden positions; same method/budgets, owned KV; no calibration/selection.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==frozen
    else:atomic(pp,frozen)
    module,reducer=load(),backup_load();outputs=[];root=[];start=time.monotonic()
    for version,order in enumerate([np.arange(n),np.random.default_rng(frozen['seed']).permutation(n)]):
        logits=np.zeros((n,2432));qs=np.zeros((n,k))
        for lo in range(0,n,128):
            path=out/f'v{version}-{lo:06d}.npz';take=order[lo:lo+128];part=[rows[i] for i in take]
            if not path.exists():
                oracle.reset();bridge=ShipHandles(oracle,[r['prefix'] for r in part],[feat[i] for i in take]);forced=np.array([len(r['legal'])==1 for r in part])
                tree=module.Tree([r['prefix'] for r in part],bridge.root_logits,np.where(forced,0,64).tolist(),[2.5]*len(part),4)
                for budget in BUDGETS:
                    if budget!=64:tree.grow(np.where(forced,0,budget).tolist())
                    while not tree.done:
                        h=tree.select()
                        if len(h):tree.update(bridge(h))
                compact=tree.compact();q=reducer.Backup(compact,1000,16.).reduce(np.log(.2),-.5,ids[take])[0];del tree,compact
                tmp=path.with_suffix('.partial')
                with tmp.open('wb') as f:np.savez_compressed(f,indices=take,root=bridge.root_logits,q=q)
                tmp.replace(path)
            with np.load(path) as f:np.testing.assert_array_equal(f['indices'],take);logits[take]=f['root'];qs[take]=f['q']
        outputs.append(policy(rows,logits,qs,ids,mask,np.array([f[-1,0] for f in feat]),plan['old_parameters']['unchanged']));root.append(logits)
    a,b=outputs;target=np.array([r['legal'].index(r['target']) for r in rows]);delta=np.log(a[np.arange(n),target])-np.log(b[np.arange(n),target]);cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows])
    boot=bootstrap_deltas(delta[:,None],cells,games)[:,:,0];m=cellmean(delta,cells)
    result=dict(positions=n,max_policy_gap=float(abs(a-b).max()),mean_policy_kl=float(np.mean((a*np.log(np.maximum(a,1e-300)/np.maximum(b,1e-300))).sum(1))),
        order2_minus_order1_macro=float(m.mean()),expert=float(m[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(),
        root_max_logit_gap=float(abs(root[0]-root[1]).max()),seconds=time.monotonic()-start,plan_sha256=digest(pp),
        interpretation='Numerical sensitivity diagnostic on1024 positions, not an estimated extra statistical variance term or a full fresh-sample confirmation.')
    atomic(out/'results.json',result);print('ORDER AUDIT',size,result,flush=True);return result
