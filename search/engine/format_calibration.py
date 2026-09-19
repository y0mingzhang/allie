"""Partial pooling of format-by-rating calibration toward the Elo-only model."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .balanced_eval import ROOT, atomic, digest
from .analyze_august import means
from .innovation import fit as elo_fit


def fit(features,mask,target,cells,train,ridge):
    f=features[train];m=mask[train];c=cells[train];t=target[train];ar=np.arange(len(c))
    count=np.bincount(c,minlength=16);w=1/count[c];w/=w.sum();scales=np.array([.1,4.])
    def objective(x):
        theta=x.reshape(16,2)*scales;a=np.where(m,np.einsum('nd,ndk->nk',theta[c],f),-np.inf)
        lp=a-logsumexp(a,axis=1,keepdims=True);p=np.exp(lp)
        residual=np.einsum('nk,ndk->nd',p,f)-f[ar,:,t];grad=np.zeros((16,2));np.add.at(grad,c,w[:,None]*residual)
        deviation=(theta.reshape(4,4,2)-theta.reshape(4,4,2).mean(0,keepdims=True)).reshape(16,2)
        # The mean's derivative cancels because each Elo group's deviations sum to zero.
        grad+=ridge*deviation/scales**2/16
        return -w@lp[ar,t]+.5*ridge*np.square(deviation/scales).sum()/16,(grad*scales).ravel()
    # Optimize in regularizer units: otherwise alpha/beta's scale ratio makes
    # the strongest pooling case hit the iteration limit despite convexity.
    opt=minimize(objective,np.tile([10.,.25],16),jac=True,method='L-BFGS-B',bounds=[(4.,16.),(0.,10.)]*16,
        options=dict(maxiter=500,ftol=1e-12,gtol=1e-7))
    theta=opt.x.reshape(16,2)*scales;a=np.where(mask,np.einsum('nd,ndk->nk',theta[cells],features),-np.inf)
    return a-logsumexp(a,axis=1,keepdims=True),dict(theta=theta.tolist(),ridge=ridge,converged=bool(opt.success),iterations=int(opt.nit),message=str(opt.message))


def main():
    start=time.monotonic();rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);ids=np.zeros((n,k),int);target=np.zeros(n,int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    root=np.zeros((n,2432));q=np.zeros((n,k));nodes=np.zeros(n)
    for lo in range(0,n,512):
        with np.load(ROOT/f'aug-coverage-v1/coverage_bernoulli/{lo:06d}.npz') as z:
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);kk=z['q'].shape[-1]
            root[lo:hi]=z['z'];q[lo:hi,:kk]=z['q'][-1];nodes[lo:hi]=z['evaluated_nodes'][-1]
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.);features=np.stack([logits,q],1)
    records={};losses={}
    for name,ridge in [('elo',None),('format001',.001),('format01',.01),('format1',.1),('format10',1.)]:
        def train(t):return elo_fit(features,mask,target,cells,t,0.) if ridge is None else fit(features,mask,target,cells,t,ridge)
        p,params=train(fm);loss=-p[ar,target];oof=np.full(n,np.nan);converged=[]
        for fold in range(3):
            val=fm&(cv==fold);v,info=train(fm&(cv!=fold));oof[val]=-v[ar[val],target[val]];converged.append(info['converged'])
        a=means(loss[~fm],cells[~fm]);b=means(oof[fm],cells[fm]);losses[name]=loss
        records[name]=dict(parameters=params,cv_converged=converged,training_equivalent_cm=None,mean_nodes=float(means(nodes,cells).mean()),
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
        print(name,records[name]['fit_game_cv'],a.mean(),a[3::4].mean(),params['converged'],flush=True)
    prior=json.loads((ROOT/'aug-coverage-v1/results.json').read_text())['results']['coverage_bernoulli_1000_elo']['confirmation']
    for key in ('macro_ce','expert_ce'):np.testing.assert_allclose(records['elo']['confirmation'][key],prior[key],atol=2e-6,rtol=0)
    selected={metric:min(records,key=lambda k:records[k]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(8317).multinomial(g,np.full(g,1/g),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name in records:
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-losses['elo'])[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(ROOT/'aug-coverage-v1/format-calibration.json',dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,
        source_sha256=digest(Path(__file__)),stage='August potentially training-seen; same root-coverage trees/node cost. Cell alpha/beta partially pooled toward Elo means, known header format/rating only. All fitting inside game CV; all five arms reported; CM pending golden.'))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
