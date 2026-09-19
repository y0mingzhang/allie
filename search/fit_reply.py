"""Fit two policy-calibration coefficients on the pilot's fit games only.

The first grid's reply beta landed at its maximum (4). Fit the convex two-feature
logistic objective rather than choose another arbitrary grid boundary. Preserve the
previous selected.json and its ongoing confirmation unchanged.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    root=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'][:,378:2346]
                         for i in range(0,n,128)]).astype(np.float64)
    q=np.concatenate([np.load(ROOT/f'reply-pilot/{i:05d}.npz')['q2'] for i in range(0,n,64)])
    legal=np.isfinite(q);q=np.nan_to_num(q).astype(np.float64)
    # Constants per position cancel in the softmax; center for conditioning.
    root-=root.max(1,keepdims=True)
    fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
    target=np.array([r['target']-378 for r in rows]);ar=np.arange(n)
    outputs={}
    for name,subset in [('all',np.ones(n,bool)),('expert',expert),('nonexpert',~expert)]:
        mask=fit&subset;x=root[mask];v=q[mask];ok=legal[mask];y=target[mask];a=np.arange(len(y))
        def objective(par):
            z=np.where(ok,par[0]*x+par[1]*v,-np.inf);norm=logsumexp(z,axis=1)
            p=np.exp(z-norm[:,None])
            loss=np.mean(norm-z[a,y])
            grad=np.array([(p*x).sum(1).mean()-x[a,y].mean(),(p*v).sum(1).mean()-v[a,y].mean()])
            return loss,grad
        opt=minimize(objective,[1.,4.],jac=True,method='L-BFGS-B',bounds=[(.5,2.),(0.,32.)],
                     options=dict(ftol=1e-12,gtol=1e-8,maxiter=200))
        assert opt.success,opt.message
        # Finite-difference check guards the analytic gradient on real logits.
        at=np.array([.93,3.7]);_,grad=objective(at);eps=1e-5
        numerical=np.array([(objective(at+np.eye(2)[i]*eps)[0]-objective(at-np.eye(2)[i]*eps)[0])/(2*eps) for i in range(2)])
        assert np.max(np.abs(grad-numerical))<1e-7
        alpha,beta=map(float,opt.x)
        z=np.where(legal,alpha*root+beta*q,-np.inf);loss=logsumexp(z,axis=1)-z[ar,target]
        outputs[name]=dict(alpha=alpha,beta=beta,fit_positions=int(mask.sum()),
            fit_ce=float(loss[mask].mean()),confirmation_ce=float(loss[(~fit)&subset].mean()),
            confirmation_expert_ce=float(loss[(~fit)&expert].mean()))
    result=dict(method='legal softmax(alpha*move_logit + beta*two_ply_root_WDL)',
        scope='Fit on fold0 only. Both per-rating groups and common coefficients are frozen before expanded confirmation.',
        parameters=outputs,data_sha256=hashlib.sha256((ROOT/'dev.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'reply-pilot/calibrated-selection.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
