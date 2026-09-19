"""Dev-only horizon extension. Fit on fold0; report every preset arm on fold1."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();out=Path(a.directory).resolve()
    assert out.is_relative_to(ROOT.resolve())
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    plan=json.loads((out/'engine-plan.json').read_text());bs=plan['roots_per_batch']
    z=[];q=[];cost=[]
    for lo in range(0,n,bs):
        with np.load(out/f'{lo:06d}.npz') as f:
            assert list(f['game'])==[r['game'] for r in rows[lo:lo+bs]]
            assert list(f['ply'])==[r['ply'] for r in rows[lo:lo+bs]]
            z.append(f['root']);q.append(f['q']);cost.append(json.loads(str(f['stats'])))
    z=np.concatenate(z)[:,378:2346].astype(float);q=np.concatenate(q,axis=1).astype(float)
    ids=np.zeros((n,max(len(r['legal']) for r in rows)),int);legal=np.zeros_like(ids,bool);y=np.zeros(n,int)
    for i,r in enumerate(rows):
        ids[i,:len(r['legal'])]=np.array(r['legal'])-378;legal[i,:len(r['legal'])]=True
        y[i]=r['legal'].index(r['target'])
    ar=np.arange(n);z=z[ar[:,None],ids];q=np.nan_to_num(q[:,ar[:,None],ids]);fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
    assert np.isfinite(q).all()
    def fit_policy(features,bounds):
        x=features[fit];mask=legal[fit];yy=y[fit];aa=np.arange(fit.sum())
        def loss(w):
            logits=np.where(mask,x@w,-np.inf);normalizer=logsumexp(logits,axis=1);p=np.exp(logits-normalizer[:,None])
            return float((normalizer-logits[aa,yy]).mean()),np.einsum('na,nak->k',p,x)/len(yy)-x[aa,yy].mean(0)
        result=minimize(loss,np.array([1.,*([0.]*(features.shape[-1]-1))]),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-12,gtol=1e-8))
        assert result.success,result.message
        return result.x
    records={};losses={}
    def score(name,x,w):
        s=np.where(legal,x@w,-np.inf);loss=logsumexp(s,axis=1)-s[ar,y];correct=s.argmax(1)==y;losses[name]=loss
        records[name]=dict(coefficients=list(map(float,w)),metrics={label:dict(ce=float(loss[m].mean()),expert_ce=float(loss[m&expert].mean()),accuracy=float(correct[m].mean()),expert_accuracy=float(correct[m&expert].mean())) for label,m in [('fit',fit),('confirmation',~fit)]})
    score('legal',z[:,:,None],np.array([1.]))
    for depth in range(1,len(q)+1):
        x=np.stack([z,q[depth-1]],axis=-1);w=fit_policy(x,[(.5,2.),(0.,32.)]);score(f'depth{depth}',x,w)
    # Predeclared discrepancy arm: deep-vs-shallow disagreement can correct leaf bias.
    x=np.stack([z,q[-1],q[-1]-q[1]],axis=-1)
    w=fit_policy(x,[(.5,2.),(0.,32.),(-32.,32.)]);score('deep_plus_disagreement',x,w)
    games=np.array([r['game'] for r in rows]);u,ix=np.unique(games[~fit],return_inverse=True)
    weights=np.random.default_rng(196921).multinomial(len(u),np.full(len(u),1/len(u)),size=2000)
    for name,v in records.items():
        v['paired_delta_vs_legal']={}
        for label,mask in [('overall',np.ones(n,bool)),('expert',expert)]:
            m=mask[~fit];delta=(losses[name]-losses['legal'])[~fit]
            numer=np.bincount(ix,weights=delta*m,minlength=len(u));denom=np.bincount(ix,weights=m,minlength=len(u))
            draws=(weights@numer)/(weights@denom)
            v['paired_delta_vs_legal'][label]=dict(delta=float(delta[m].mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
    result=dict(stage='Reused small development game split; all arms reported, not golden macro',
                positions=int((~fit).sum()),expert_positions=int((~fit&expert).sum()),results=records,
                cost=dict(seconds=sum(v['seconds'] for v in cost),new_tokens=sum(v['new_tokens'] for v in cost)),
                plan_sha256=hashlib.sha256((out/'engine-plan.json').read_bytes()).hexdigest(),
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                training_equivalent_cm=None,cm_note='Golden scaling laws cannot be applied to blitz-only development metrics.')
    tmp=out/'results.partial';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'results.json')
    for k,v in records.items():print(k,v['metrics']['confirmation'])


if __name__=='__main__':main()
