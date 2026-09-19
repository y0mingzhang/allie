"""Test nonlinear value corrections using existing two-ply predictions only.

The auxiliary value is expected score (win + half draw), not a win probability.
Compare linear score, multiplicative score weighting, score odds, and a bounded
state-dependent scale. All coefficients use pilot fit games; no new GPU calls.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    root=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'] for i in range(0,n,128)]).astype(np.float64)
    value=np.concatenate([np.load(ROOT/f'reply-pilot/{i:05d}.npz')['q2'] for i in range(0,n,64)])
    legal=np.isfinite(value);value=np.nan_to_num(value).astype(np.float64);v=np.clip(value,1e-3,1-1e-3)
    rv=softmax(root[:,2413:2416],axis=1)@np.array([1.,.5,0.]);move=root[:,378:2346]
    features=dict(linear=value,log_score=np.log(v),score_odds=np.log(v)-np.log1p(-v),
                  uncertainty_scaled=value/np.maximum(.2,np.sqrt(rv*(1-rv)))[:,None])
    target=np.array([r['target']-378 for r in rows]);ar=np.arange(n);fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows]);result={}
    for name,feature in features.items():
        x=np.stack([move,feature],-1);xf=x[fit];mask=legal[fit];y=target[fit];a=np.arange(len(y))
        def objective(w):
            z=np.where(mask,np.einsum('nav,v->na',xf,w),-np.inf);norm=logsumexp(z,axis=1);p=np.exp(z-norm[:,None])
            return float(np.mean(norm-z[a,y])),np.einsum('na,nav->v',p,xf)/len(y)-xf[a,y].mean(0)
        opt=minimize(objective,[1.,1.],jac=True,method='L-BFGS-B',bounds=[(.5,2.),(0.,32.)],options=dict(ftol=1e-12,gtol=1e-8,maxiter=200))
        assert opt.success,opt.message
        z=np.where(legal,np.einsum('nav,v->na',x,opt.x),-np.inf);loss=logsumexp(z,axis=1)-z[ar,target];correct=z.argmax(1)==target
        result[name]=dict(alpha=float(opt.x[0]),beta=float(opt.x[1]),
            metrics={label:dict(ce=float(loss[m].mean()),expert_ce=float(loss[m&expert].mean()),
                               accuracy=float(correct[m].mean()),expert_accuracy=float(correct[m&expert].mean())) for label,m in [('fit',fit),('confirmation',~fit)]})
    chosen=min(result,key=lambda k:result[k]['metrics']['fit']['ce'])
    report=dict(stage='Cached development pilot; coefficients and family selected on fit CE only',selected=chosen,results=result,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'reply-pilot/utility-selection.json'
    if path.exists():assert json.loads(path.read_text())==report
    else:path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
