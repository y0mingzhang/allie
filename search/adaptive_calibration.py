"""Cheap cached calibration of value disagreement and root uncertainty/time.

Choose the family by three-way game CV within pilot fit games, then report the
unchanged pilot check. Pack legal actions before fitting to avoid dense vocabulary
work. No golden scores, new inference, or observed human future are inputs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax

from allie_mcts import expected_seconds

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def optimize(x,legal,target,mask):
    xf=x[mask];ok=legal[mask];y=target[mask];a=np.arange(len(y));k=x.shape[-1]
    def fun(w):
        z=np.where(ok,np.einsum('nav,v->na',xf,w),-np.inf);norm=logsumexp(z,axis=1);p=np.exp(z-norm[:,None])
        return float(np.mean(norm-z[a,y])),np.einsum('na,nav->v',p,xf)/len(y)-xf[a,y].mean(0)
    opt=minimize(fun,[1.,6.]+[0.]*(k-2),jac=True,method='L-BFGS-B',
        bounds=[(.5,2.),(0.,32.)]+[(-16.,16.)]*(k-2),options=dict(ftol=1e-12,gtol=1e-8,maxiter=200))
    assert opt.success,opt.message
    return opt.x


def main():
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    root=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'] for i in range(0,n,128)]).astype(float)
    q={name:np.concatenate([np.load(ROOT/f'reply-pilot/{i:05d}.npz')[name] for i in range(0,n,64)])
       for name in ('q1','q2')}
    width=max(len(r['legal']) for r in rows);idx=np.zeros((n,width),int);legal=np.zeros((n,width),bool);target=np.zeros(n,int)
    for i,row in enumerate(rows):
        actions=row['legal'];idx[i,:len(actions)]=actions;legal[i,:len(actions)]=True;target[i]=actions.index(row['target'])
    ar=np.arange(n)[:,None];move=root[ar,idx];v1=q['q1'][ar,np.maximum(0,idx-378)];v2=q['q2'][ar,np.maximum(0,idx-378)]
    v1=np.nan_to_num(v1);v2=np.nan_to_num(v2);move-=np.max(np.where(legal,move,-np.inf),1,keepdims=True)
    lp=np.where(legal,move,-np.inf);lp-=logsumexp(lp,axis=1,keepdims=True)
    entropy=-np.sum(np.exp(lp)*np.where(legal,lp,0),axis=1);time=np.log1p(expected_seconds(root))
    fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows]);splits=np.array([int(hashlib.sha256(('calibration-cv:'+r['game']).encode()).hexdigest()[:8],16)%3 for r in rows])
    # Fixed physical scales avoid leaking held-out observations into CV features.
    tf=np.clip(time/np.log(16.)-1,-2,2);ef=np.clip(entropy/2.-1,-2,2)
    features=dict(two_ply=[move,v2],value_disagreement=[move,v2,v2-v1],predicted_time=[move,v2,v2*tf[:,None]],
                  policy_entropy=[move,v2,v2*ef[:,None]])
    result={}
    for name,parts in features.items():
        x=np.stack(parts,-1);cv_loss=[];cv_n=[]
        for fold in range(3):
            train=fit&(splits!=fold);val=fit&(splits==fold);w=optimize(x,legal,target,train)
            z=np.where(legal[val],np.einsum('nav,v->na',x[val],w),-np.inf)
            loss=logsumexp(z,axis=1)-z[np.arange(val.sum()),target[val]];cv_loss.extend(loss.tolist());cv_n.append(int(val.sum()))
        w=optimize(x,legal,target,fit);z=np.where(legal,np.einsum('nav,v->na',x,w),-np.inf)
        loss=logsumexp(z,axis=1)-z[np.arange(n),target]
        result[name]=dict(coefficients=w.tolist(),cv_ce=float(np.mean(cv_loss)),cv_positions=cv_n,
            metrics={label:dict(ce=float(loss[m].mean()),expert_ce=float(loss[m&expert].mean()),
                accuracy=float((z.argmax(1)==target)[m].mean()),expert_accuracy=float((z.argmax(1)==target)[m&expert].mean()))
                for label,m in [('fit',fit),('confirmation',~fit)]})
    selected=min(result,key=lambda k:result[k]['cv_ce'])
    report=dict(stage='Existing dev pilot; family selected by game CV inside fit fold',selected=selected,results=result,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'reply-pilot/adaptive-selection.json'
    if path.exists():assert json.loads(path.read_text())==report
    else:path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
