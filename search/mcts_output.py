"""Separate output-policy regularization from tree traversal/budget.

Fixed lambda solves an explicit KL problem. Existing tree values can be reused,
but these output-only comparisons do not test a different tree-selection rule.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax,logsumexp

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def reverse_kl(prior,values,lam):
    assert lam>0
    prior=np.asarray(prior,np.float64);q=np.asarray(values,np.float64)
    assert (prior>0).all() and np.isclose(prior.sum(),1.)
    lo=np.max(q+lam*prior);hi=np.max(q)+lam
    for _ in range(80):
        alpha=(lo+hi)/2;p=lam*prior/(alpha-q)
        if p.sum()>1:lo=alpha
        else:hi=alpha
    p=lam*prior/((lo+hi)/2-q)
    return p/p.sum()


def policy(logits,values,legal,lam,direction):
    out=np.zeros_like(logits,dtype=np.float64)
    for i,ids in enumerate(legal):
        prior=softmax(logits[i,ids].astype(np.float64))
        if direction=='prior_to_search':
            out[i,ids]=reverse_kl(prior,values[i,ids],lam)
        else:
            assert direction=='search_to_prior'
            out[i,ids]=softmax(logits[i,ids].astype(np.float64)+values[i,ids]/lam)
    return out


def main():
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    logits=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'][:,378:2346] for i in range(0,n,128)])
    legal=[np.array(r['legal'])-378 for r in rows];target=np.array([r['target']-378 for r in rows]);ar=np.arange(n)
    fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows]);output={}
    # Freeze the comparison's lambda on FIXED52 fit games, then apply unchanged
    # to both tree types. Do not select a separate lambda from adaptive results.
    trees={name:np.concatenate([np.load(ROOT/f'mcts-pilot/{name}-{i:05d}.npz')['values'] for i in range(0,n,128)])
           for name in ('fixed_matched','adaptive50')}
    for direction in ('prior_to_search','search_to_prior'):
        trials=[]
        for lam in (.0625,.125,.25,.5,1.,2.,4.):
            p=policy(logits,trees['fixed_matched'],legal,lam,direction)
            loss=-np.log(p[ar,target]);trials.append((float(loss[fit&expert].mean()),lam))
        lam=min(trials)[1];results={}
        for name,q in trees.items():
            p=policy(logits,q,legal,lam,direction);loss=-np.log(p[ar,target]);correct=p.argmax(1)==target
            results[name]={}
            for label,mask in [('fit',fit),('confirmation',~fit)]:
                results[name][label]=dict(ce=float(loss[mask].mean()),expert_ce=float(loss[mask&expert].mean()),
                    accuracy=float(correct[mask].mean()),expert_accuracy=float(correct[mask&expert].mean()))
        output[direction]=dict(lambda_selected=lam,fit_trials=trials,results=results)
    record=dict(stage='Output-only development ablation; trees unchanged, golden unopened',results=output)
    path=ROOT/'mcts-pilot/output-ablation.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(record))
    else:path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
