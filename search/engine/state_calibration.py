"""State-dependent strength of a fixed search correction; August game CV only.

Policy logits are alpha_g * logit + (beta_g + gamma dot feature) * Q.
Only gamma is penalized. Features come from root predictions and the finished
tree, never future human moves or outcomes. Shared gamma controls capacity.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from .balanced_eval import ROOT, atomic, digest
from .analyze_august import means


def calibrate(z,q,mask,target,cells,features,train,ridge):
    g=cells%4; mu=features[train].mean(0); sd=np.maximum(features[train].std(0),1e-6)
    x=np.clip((features-mu)/sd,-3,3); d=x.shape[1]; ar=np.arange(train.sum())
    gt=g[train];zt=z[train];qt=q[train];xt=x[train];mt=mask[train];tt=target[train]
    count=np.bincount(cells[train],minlength=16);w=1/count[cells[train]];w/=w.sum()
    def objective(theta):
        alpha=theta[:4][gt];beta=theta[4:8][gt]+xt@theta[8:]
        a=np.where(mt,alpha[:,None]*zt+beta[:,None]*qt,-np.inf)
        logp=a-logsumexp(a,axis=1,keepdims=True);p=np.exp(logp)
        rz=(p*zt).sum(1)-zt[ar,tt];rq=(p*qt).sum(1)-qt[ar,tt]
        loss=-w@logp[ar,tt]+.5*ridge*np.square(theta[8:]).sum()
        grad=np.r_[np.bincount(gt,weights=w*rz,minlength=4),np.bincount(gt,weights=w*rq,minlength=4),xt.T@(w*rq)+ridge*theta[8:]]
        return loss,grad
    opt=minimize(objective,np.r_[np.ones(4),np.ones(4),np.zeros(d)],jac=True,method='L-BFGS-B',
        bounds=[(.4,1.6)]*4+[(0,40)]*4+[(-10,10)]*d,options=dict(maxiter=400,ftol=1e-12,gtol=1e-7))
    assert np.isfinite(opt.fun)
    beta=opt.x[4:8][g]+x@opt.x[8:]
    p=softmax(np.where(mask,opt.x[:4][g,None]*z+beta[:,None]*q,-np.inf),axis=1)
    info=dict(theta=opt.x.tolist(),mean=mu.tolist(),scale=sd.tolist(),clip=3,ridge=ridge,
        converged=bool(opt.success),iterations=int(opt.nit),beta_min=float(beta.min()),beta_max=float(beta.max()))
    return p,info


def main():
    start=time.monotonic();rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);ids=np.zeros((n,k),int);target=np.zeros(n,int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    root=np.zeros((n,2432));q=np.zeros((n,k));cost=np.zeros(n)
    for lo in range(0,n,512):
        with np.load(ROOT/f'aug-selection-v1/zero_cp25/{lo:06d}.npz') as z:
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);kk=z['q'].shape[-1]
            root[lo:hi]=z['z'];q[lo:hi,:kk]=z['q'][-1];cost[lo:hi]=z['evaluated_nodes'][-1]
    logits=root[:,378:2346][ar[:,None],ids];logits=np.where(mask,logits,0.);prior=softmax(np.where(mask,logits,-np.inf),axis=1)
    tp=softmax(root[:,2350:2413],axis=1);seconds=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    v=softmax(root[:,2413:2416],axis=1)@np.array([1.,0.,-1.]);qm=(prior*q).sum(1)
    features=np.column_stack([np.log1p(tp@seconds),-(prior*np.log(np.maximum(prior,1e-300))).sum(1),np.abs(v),np.log(.01+np.sqrt((prior*(q-qm[:,None])**2).sum(1)))])
    menus=[('baseline',[],0.),('predtime',[0],.001),('entropy',[1],.001),('value_scale',[2,3],.001),('state',[0,1,2,3],.001),('state_ridge',[0,1,2,3],.01)]
    records={};losses={}
    for name,fields,ridge in menus:
        f=features[:,fields];p,params=calibrate(logits,q,mask,target,cells,f,fm,ridge);loss=-np.log(p[ar,target]);oof=np.full(n,np.nan);cv_info=[]
        for fold in range(3):
            val=fm&(cv==fold);v,info=calibrate(logits,q,mask,target,cells,f,fm&(cv!=fold),ridge)
            oof[val]=-np.log(v[ar[val],target[val]]);cv_info.append(info['converged'])
        a=means(loss[~fm],cells[~fm]);b=means(oof[fm],cells[fm]);losses[name]=loss
        records[name]=dict(fields=fields,parameters=params,cv_converged=cv_info,mean_nodes=float(means(cost,cells).mean()),training_equivalent_cm=None,
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
        print(name,records[name]['fit_game_cv'],records[name]['confirmation']['macro_ce'],records[name]['confirmation']['expert_ce'],flush=True)
    baseline=json.loads((ROOT/'aug-selection-v1/results.json').read_text())['results']['zero_cp25_1000_elo']['confirmation']
    for key in ('macro_ce','expert_ce'):np.testing.assert_allclose(records['baseline']['confirmation'][key],baseline[key],atol=2e-6,rtol=0)
    selected={metric:min(records,key=lambda k:records[k]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(99317).multinomial(g,np.full(g,1/g),size=2000).astype(float);den=w@count
    for name in records:
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-losses['baseline'])[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_vs_baseline_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    out=ROOT/'aug-selection-v1/state-calibration.json'
    atomic(out,dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,source_sha256=digest(Path(__file__)),
        stage='August possibly training-seen; same cpuct2.5 trees and node count. Conditional search correction only. Parameters and preprocessing refit inside each game CV fold. No golden tuning or CM conversion.'))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
