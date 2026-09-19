"""Test expected outcomes under a value-tilted continuation policy on cached trees."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT, atomic, digest
from .analyze_august import means
from .innovation import fit
from .behavior_native import load, test
from .compact_native import load as load_soft


def main():
    start=time.monotonic();module=load();test(module)
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);ids=np.zeros((n,k),int);target=np.zeros(n,int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    lambdas=[.05,.1,.2,.4,1.];root=np.zeros((n,2432));q=np.zeros((len(lambdas),n,k));nodes=np.zeros(n)
    for lo in range(0,n,512):
        with np.load(ROOT/f'aug-coverage-v1/coverage_bernoulli/{lo:06d}.npz') as z:
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);kk=z['q'].shape[-1]
            root[lo:hi]=z['z'];nodes[lo:hi]=z['evaluated_nodes'][-1]
            data={key:z[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            for j,lam in enumerate(lambdas):
                v=load_soft().reduce(data,1000,.1,.1) if j==len(lambdas)-1 else module.reduce(data,1000,lam,lam)
                q[j,lo:hi]=v[np.arange(hi-lo)[:,None],ids[lo:hi]]
            np.testing.assert_allclose(q[-1,lo:hi,:kk][mask[lo:hi,:kk]],z['q'][-1][mask[lo:hi,:kk]],atol=1e-9,rtol=0)
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.);records={};losses={}
    for j,lam in enumerate(lambdas):
        name='soft_control' if j==len(lambdas)-1 else f'behavior_tau{lam:g}';features=np.stack([logits,q[j]],1)
        p,params=fit(features,mask,target,cells,fm,0.);loss=-p[ar,target];oof=np.full(n,np.nan);converged=[]
        for fold in range(3):
            val=fm&(cv==fold);v,info=fit(features,mask,target,cells,fm&(cv!=fold),0.)
            oof[val]=-v[ar[val],target[val]];converged.append(info['converged'])
        a=means(loss[~fm],cells[~fm]);b=means(oof[fm],cells[fm]);losses[name]=loss
        records[name]=dict(parameters=params,cv_converged=converged,training_equivalent_cm=None,mean_nodes=float(means(nodes,cells).mean()),
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
        print(name,records[name]['fit_game_cv'],a.mean(),a[3::4].mean(),flush=True)
    prior=json.loads((ROOT/'aug-coverage-v1/results.json').read_text())['results']['coverage_bernoulli_1000_elo']['confirmation']
    for key in ('macro_ce','expert_ce'):np.testing.assert_allclose(records['soft_control']['confirmation'][key],prior[key],atol=2e-6,rtol=0)
    selected={metric:min(records,key=lambda k:records[k]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(8317).multinomial(g,np.full(g,1/g),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name in records:
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-losses['soft_control'])[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(ROOT/'aug-coverage-v1/behavior.json',dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('behavior.cpp'),Path(__file__).with_name('behavior_native.py')]},
        stage='August potentially training-seen, game-CV selection. Expected outcome under a tilted human continuation policy instead of a KL-regularized Bellman objective. Same coverage tree/prior/nodes. Four temperatures and unchanged soft control; CM pending golden.'))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
