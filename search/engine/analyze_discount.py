"""Test geometric-depth critic regularization on the exact existing trees."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT, atomic, digest
from .analyze_august import means
from .innovation import fit
from .discount_native import load, test


def main():
    start=time.monotonic();module=load();test(module)
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);ids=np.zeros((n,k),int);target=np.zeros(n,int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    lambdas=[0.,.25,.5,.75,1.];root=np.zeros((n,2432));q=np.zeros((len(lambdas),n,k));nodes=np.zeros(n)
    for lo in range(0,n,512):
        with np.load(ROOT/f'aug-selection-v1/zero_cp25/{lo:06d}.npz') as z:
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);kk=z['q'].shape[-1]
            root[lo:hi]=z['z'];nodes[lo:hi]=z['evaluated_nodes'][-1]
            data={key:z[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            for j,lam in enumerate(lambdas):
                v=module.reduce(data,1000,.1,lam)
                q[j,lo:hi]=v[np.arange(hi-lo)[:,None],ids[lo:hi]]
            np.testing.assert_allclose(q[-1,lo:hi,:kk][mask[lo:hi,:kk]],z['q'][-1][mask[lo:hi,:kk]],atol=3e-12,rtol=0)
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.);records={};losses={}
    for j,lam in enumerate(lambdas):
        name=f'lambda{lam:g}';features=np.stack([logits,q[j]],1)
        p,params=fit(features,mask,target,cells,fm,0.);loss=-p[ar,target];oof=np.full(n,np.nan);converged=[]
        for fold in range(3):
            val=fm&(cv==fold);v,info=fit(features,mask,target,cells,fm&(cv!=fold),0.)
            oof[val]=-v[ar[val],target[val]];converged.append(info['converged'])
        a=means(loss[~fm],cells[~fm]);b=means(oof[fm],cells[fm]);losses[name]=loss
        records[name]=dict(parameters=params,cv_converged=converged,training_equivalent_cm=None,mean_nodes=float(means(nodes,cells).mean()),
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
        print(name,records[name]['fit_game_cv'],a.mean(),a[3::4].mean(),flush=True)
    prior=json.loads((ROOT/'aug-selection-v1/results.json').read_text())['results']['zero_cp25_1000_elo']['confirmation']
    for key in ('macro_ce','expert_ce'):np.testing.assert_allclose(records['lambda1']['confirmation'][key],prior[key],atol=2e-6,rtol=0)
    selected={metric:min(records,key=lambda k:records[k]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(8317).multinomial(g,np.full(g,1/g),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name in records:
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-losses['lambda1'])[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(ROOT/'aug-selection-v1/discount.json',dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('discount.cpp'),Path(__file__).with_name('discount_native.py')]},
        stage='August potentially training-seen, game-CV selection. Recursive V=(1-lambda)*critic+lambda*soft-Bellman, terminal values exact. Same tree/prior/nodes; lambda0 shown at same cost for ablation, although standalone one-ply is cheaper. All five arms reported; CM pending golden.'))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
