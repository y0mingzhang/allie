"""Fit next-stage policy corrections on fold0 only, then report fold1."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, softmax

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    root=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'] for i in range(0,n,128)]).astype(np.float64)
    q1=np.concatenate([np.load(ROOT/f'reply-pilot/{i:05d}.npz')['q1'] for i in range(0,n,64)])
    q2=np.concatenate([np.load(ROOT/f'reply-pilot/{i:05d}.npz')['q2'] for i in range(0,n,64)])
    legal=np.isfinite(q1);move=root[:,378:2346]
    rootvalue=softmax(root[:,2413:2416],axis=1)@np.array([1.,.5,0.])
    timegate=softmax(root[:,2350:2413],axis=1)[:,6:].sum(1)
    target=np.array([r['target']-378 for r in rows]);ar=np.arange(n)
    fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
    order=np.argsort(np.where(legal,move,-np.inf),axis=1)[:,::-1]
    ranks=np.empty_like(order);np.put_along_axis(ranks,order,np.arange(1968)[None,:],axis=1)
    baseline=logsumexp(np.where(legal,move,-np.inf),axis=1)-move[ar,target]
    basefit=float(baseline[fit].mean())
    def evaluate(p):
        q=(1-p['reply_mix'])*q1+p['reply_mix']*q2
        advantage=np.where(legal&(ranks<p['k']),q-rootvalue[:,None],0.)
        gate=timegate if p['gate']=='time' else np.ones(n)
        enabled=expert if p['expert_only'] else np.ones(n,bool)
        x=move/np.where(enabled,p['temperature'],1.)[:,None]+p['beta']*(gate*enabled)[:,None]*advantage
        x=np.where(legal,x,-np.inf)
        return logsumexp(x,axis=1)-x[ar,target],x.argmax(1)==target
    trials=[];predictions={}
    # Freeze temperature at the earlier cheap-control choice; avoid another
    # large temperature sweep on the same small development sample.
    for temperature in (1.,):
        for beta in (.5,1.,2.,4.):
            for mix in (0.,.5,1.):
                for k in (8,1968):
                    for gate in ('none','time'):
                        for expert_only in (False,True):
                            p=dict(temperature=temperature,beta=beta,reply_mix=mix,k=k,gate=gate,expert_only=expert_only)
                            nll,_=evaluate(p)
                            score=float(nll[fit&expert].mean()+max(0,nll[fit].mean()-basefit))
                            trials.append((score,p))
    best=min(trials,key=lambda x:x[0]);same1=min((t for t in trials if t[1]['reply_mix']==0),key=lambda x:x[0])
    result={}
    for name,choice in [('selected',best),('best_one_ply',same1)]:
        nll,correct=evaluate(choice[1]);result[name]=dict(parameters=choice[1],fit_objective=choice[0])
        for fold in (0,1):
            mask=fit if fold==0 else ~fit;e=mask&expert
            result[name][f'fold{fold}']=dict(ce=float(nll[mask].mean()),expert_ce=float(nll[e].mean()),
                accuracy=float(correct[mask].mean()),expert_accuracy=float(correct[e].mean()))
    result['trials']=len(trials);result['data_sha256']=hashlib.sha256((ROOT/'dev.json').read_bytes()).hexdigest()
    result['limitation']='Development-only selection; confirmation folded by game. Golden remains untouched.'
    frozen=ROOT/'reply-pilot/selected.json'
    if frozen.exists():assert json.loads(frozen.read_text())==result
    else:frozen.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
