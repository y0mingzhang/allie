"""Fit horizon-specific calibration on pilot fit games; report held-out dev."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from deeper_pilot import ROOT,OUT


def main():
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    q=np.concatenate([np.load(OUT/f'{i:05d}.npz')['q'] for i in range(0,n,16)],axis=1).astype(float)
    root=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'][:,378:2346]
                         for i in range(0,n,128)]).astype(float)
    # D1/D2 independently reproduce the old generator; D3/D4 add continuations.
    old=np.concatenate([np.load(ROOT/f'reply-pilot/{i:05d}.npz')['q2'] for i in range(0,n,64)])
    assert np.allclose(q[1],old,atol=1e-7,equal_nan=True)
    legal=np.isfinite(q[0]);q=np.nan_to_num(q);root-=root.max(1,keepdims=True)
    fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
    target=np.array([r['target']-378 for r in rows]);ar=np.arange(n)
    result={}
    for name,features in [(f'depth{h+1}',[root,q[h]]) for h in range(4)]+[('depth2+depth4',[root,q[1],q[3]])]:
        x=np.stack(features,-1);xf=x[fit];mask=legal[fit];y=target[fit];a=np.arange(len(y))
        def objective(w):
            z=np.where(mask,np.einsum('nav,v->na',xf,w),-np.inf);norm=logsumexp(z,axis=1);p=np.exp(z-norm[:,None])
            return float(np.mean(norm-z[a,y])),np.einsum('na,nav->v',p,xf)/len(y)-xf[a,y].mean(0)
        opt=minimize(objective,[1.]+[3.]*(len(features)-1),jac=True,method='L-BFGS-B',
            bounds=[(.5,2.)]+[(0.,32.)]*(len(features)-1),options=dict(ftol=1e-12,gtol=1e-8,maxiter=200))
        assert opt.success,opt.message
        z=np.where(legal,np.einsum('nav,v->na',x,opt.x),-np.inf);loss=logsumexp(z,axis=1)-z[ar,target]
        result[name]=dict(coefficients=opt.x.tolist(),
            metrics={label:dict(ce=float(loss[m].mean()),expert_ce=float(loss[m&expert].mean()),
                accuracy=float((z.argmax(1)==target)[m].mean()),expert_accuracy=float((z.argmax(1)==target)[m&expert].mean()))
                for label,m in [('fit',fit),('confirmation',~fit)]})
    # Keep a single horizon as the predeclared promoted family; the mixture is an
    # exploratory diagnostic with one extra parameter, not an automatic winner.
    chosen=min((f'depth{i}' for i in range(1,5)),key=lambda k:result[k]['metrics']['fit']['ce'])
    report=dict(stage='Existing small dev pilot; all parameters fit on fold0',selected=chosen,results=result,
        generator_plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=OUT/'results.json'
    if path.exists():assert json.loads(path.read_text())==report
    else:path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
