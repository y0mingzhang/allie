"""Does only the value information added by deeper search improve the prior?"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .balanced_eval import ROOT,atomic,digest
from .analyze_august import means


def fit(features,mask,target,cells,train,ridge):
    n,d,k=features.shape;g=cells%4;f=features[train];mt=mask[train];tt=target[train];gt=g[train];ar=np.arange(train.sum())
    count=np.bincount(cells[train],minlength=16);w=1/count[cells[train]];w/=w.sum()
    def objective(flat):
        theta=flat.reshape(4,d);a=np.where(mt,np.einsum('nd,ndk->nk',theta[gt],f),-np.inf)
        logp=a-logsumexp(a,axis=1,keepdims=True);p=np.exp(logp)
        residual=np.einsum('nk,ndk->nd',p,f)-f[ar,:,tt];grad=np.zeros((4,d));np.add.at(grad,gt,w[:,None]*residual)
        grad[:,2:]+=ridge*theta[:,2:]
        return -w@logp[ar,tt]+.5*ridge*np.square(theta[:,2:]).sum(),grad.ravel()
    bounds=([(.4,1.6),(0.,40.)]+[(-20.,20.)]*(d-2))*4
    opt=minimize(objective,np.tile([1.,1.]+[0.]*(d-2),4),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=300,ftol=1e-12,gtol=1e-7))
    theta=opt.x.reshape(4,d);a=np.where(mask,np.einsum('nd,ndk->nk',theta[g],features),-np.inf)
    logp=a-logsumexp(a,axis=1,keepdims=True)
    return logp,dict(theta=theta.tolist(),ridge=ridge,converged=bool(opt.success),iterations=int(opt.nit))


def main():
    start=time.monotonic();rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);ids=np.zeros((n,k),int);target=np.zeros(n,int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    root=np.zeros((n,2432));q=np.zeros((3,n,k));one=np.zeros((n,k))
    for lo in range(0,n,512):
        with np.load(ROOT/f'aug-selection-v1/zero_cp25/{lo:06d}.npz') as z:
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);kk=z['q'].shape[-1]
            root[lo:hi]=z['z'];q[:,lo:hi,:kk]=z['q']
            roots=z['roots'];parents=z['parent'];boot=z['boot'];moves=z['move'];terminal=z['terminal']
            rootmap={int(r):i for i,r in enumerate(roots)}
            one[lo:hi]=-boot[roots,None]
            for child in np.flatnonzero(np.isin(parents,roots)):
                parent=parents[child]
                if parent in rootmap:
                    i=rootmap[parent];at=np.flatnonzero(mask[lo+i]&(ids[lo+i]==moves[child]));assert len(at)==1
                    # bootstrap is already from this child's parent's perspective.
                    one[lo+i,at[0]]=boot[child] if terminal[child]<0 else (0. if terminal[child]==.5 else 1.)
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    menus=[('baseline',[q[-1]],0.),('deep_minus_one',[q[-1]-one],0.),('deep_plus_one',[q[-1],one],.0005),
           ('deep_plus64',[q[-1],q[0]],.0005),('deep_plus256',[q[-1],q[1]],.0005)]
    records={};losses={}
    for name,values,ridge in menus:
        features=np.stack([logits,*values],1);p,params=fit(features,mask,target,cells,fm,ridge);loss=-p[ar,target];oof=np.full(n,np.nan);converged=[]
        for fold in range(3):
            val=fm&(cv==fold);v,info=fit(features,mask,target,cells,fm&(cv!=fold),ridge);oof[val]=-v[ar[val],target[val]];converged.append(info['converged'])
        a=means(loss[~fm],cells[~fm]);b=means(oof[fm],cells[fm]);losses[name]=loss
        records[name]=dict(parameters=params,cv_converged=converged,training_equivalent_cm=None,
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
        print(name,records[name]['fit_game_cv'],records[name]['confirmation']['macro_ce'],records[name]['confirmation']['expert_ce'],flush=True)
    prior=json.loads((ROOT/'aug-selection-v1/results.json').read_text())['results']['zero_cp25_1000_elo']['confirmation']
    for key in ('macro_ce','expert_ce'):np.testing.assert_allclose(records['baseline']['confirmation'][key],prior[key],atol=2e-6,rtol=0)
    selected={metric:min(records,key=lambda k:records[k]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(8317).multinomial(g,np.full(g,1/g),size=2000).astype(float);den=w@count
    for name in records:
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-losses['baseline'])[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_vs_baseline_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(ROOT/'aug-selection-v1/innovation.json',dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,source_sha256=digest(Path(__file__)),
        stage='August potentially training-seen, fit-game CV selection. Same cp2.5 trees and1000sim node cost, no golden tuning. One-ply value comes from expanded root children; unseen actions get original root value. All five arms reported.'))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
