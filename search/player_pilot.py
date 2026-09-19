"""Bayesian current-form inference from the player's already observed moves.

The root human move is never supplied to the inference routine. Counterfactual
rating prompts are hypotheses about latent performance, not changed observations.
Model weights stay fixed. Hyperparameters use only existing development fold0.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from client import Oracle
from prompt_pilot import transform

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
OUT=ROOT/'player-pilot'
DELTAS=(-600,-300,-100,0,100,300,600)
HISTORY=(8,16,1024)


def trace(prefixes,oracle):
    # Prefixes end BEFORE the target move. All teacher-forced targets below
    # are strictly earlier moves that would already be known in a live game.
    pred=oracle(prefixes,all_tokens=True,compact=True)
    roots=[];scores=[];counts=[];offset=0
    for p in prefixes:
        logits=pred[offset:offset+len(p)];offset+=len(p)
        roots.append(logits[-1,378:2346])
        mover=(len(p)-11)%2
        # Exclude the first five moves of both players from the evidence.
        j=np.array([j for j in range(21,len(p)) if (j-11)%2==mover],dtype=int)
        if len(j):
            x=logits[j-1,378:2346].astype(np.float64);target=np.asarray(p)[j]-378
            assert ((target>=0)&(target<1968)).all()
            lp=x[np.arange(len(j)),target]-logsumexp(x,axis=1)
        else:lp=np.empty(0)
        scores.append([float(lp[-k:].sum()) for k in HISTORY]);counts.append([min(len(j),k) for k in HISTORY])
    assert offset==len(pred)
    return np.stack(roots),np.asarray(scores),np.asarray(counts)


def main():
    OUT.mkdir(exist_ok=True);path=ROOT/'dev.json';rows=json.loads(path.read_text())['positions'];oracle=Oracle()
    manifest=dict(data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint=oracle.ready['checkpoint_sha256'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transform_sha256=hashlib.sha256(Path(__file__).with_name('prompt_pilot.py').read_bytes()).hexdigest(),
        deltas=list(DELTAS),history=list(HISTORY),scope='Existing pilot fit fold0; causal prefix evidence only',
        evidence='Raw move likelihood of the ROOT MOVER past moves, excluding first five full moves')
    plan=OUT/'plan.json'
    if plan.exists():assert json.loads(plan.read_text())==manifest
    else:plan.write_text(json.dumps(manifest,indent=2)+'\n')
    raw=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'][:,378:2346] for i in range(0,len(rows),128)])
    variants=[];evidence=[]
    for delta in DELTAS:
        path=OUT/f'{delta:+d}.npz';started=time.monotonic()
        if not path.exists():
            r=[];s=[];c=[]
            for lo in range(0,len(rows),16):
                prefixes=[transform(p['prefix'],'mover',delta) for p in rows[lo:lo+16]]
                root,score,count=trace(prefixes,oracle);r.append(root);s.append(score);c.append(count)
            roots=np.concatenate(r)
            if delta==0:assert np.array_equal(roots,raw),'Trace must preserve direct predictions'
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez(f,root=roots,evidence=np.concatenate(s),counts=np.concatenate(c),seconds=time.monotonic()-started)
            tmp.replace(path);print('delta',delta,'seconds',time.monotonic()-started,flush=True)
        with np.load(path) as z:variants.append(z['root'].astype(np.float64));evidence.append(z['evidence'])
    variants=np.stack(variants);evidence=np.stack(evidence)
    legal=np.zeros_like(raw,dtype=bool)
    for i,p in enumerate(rows):legal[i,np.asarray(p['legal'])-378]=True
    logits=np.where(legal[None],variants,-np.inf);lp=logits-logsumexp(logits,axis=2,keepdims=True)
    target=np.array([p['target']-378 for p in rows]);ar=np.arange(len(rows));fit=np.array([p['fold']==0 for p in rows]);expert=np.array([p['cell']%4==3 for p in rows])
    base=-lp[DELTAS.index(0),ar,target];trials=[]
    for width in (100.,300.,600.):
        logprior=-.5*(np.asarray(DELTAS)/width)**2
        for h,k in enumerate(HISTORY):
            for temper in (0.,.25,.5,1.):
                weights=logprior[:,None]+temper*evidence[:,:,h]
                weights-=logsumexp(weights,axis=0,keepdims=True)
                mix=logsumexp(lp+weights[:,:,None],axis=0);loss=-mix[ar,target];correct=mix.argmax(1)==target
                result=dict(parameters=dict(prior_sd=width,history_own_moves=k,evidence_temperature=temper),
                    fit_ce=float(loss[fit].mean()),fit_expert_ce=float(loss[fit&expert].mean()),
                    confirmation_ce=float(loss[~fit].mean()),confirmation_expert_ce=float(loss[~fit&expert].mean()),
                    confirmation_accuracy=float(correct[~fit].mean()),confirmation_expert_accuracy=float(correct[(~fit)&expert].mean()))
                result['objective']=result['fit_expert_ce']+max(0,result['fit_ce']-float(base[fit].mean()))
                trials.append(result)
    baseline=dict(parameters=dict(kind='unchanged_legal_policy'),fit_ce=float(base[fit].mean()),fit_expert_ce=float(base[fit&expert].mean()),
        confirmation_ce=float(base[~fit].mean()),confirmation_expert_ce=float(base[(~fit)&expert].mean()),objective=float(base[fit&expert].mean()))
    selected=min(trials+[baseline],key=lambda x:x['objective'])
    report=dict(stage='Development pilot; no golden CM claim',selected=selected,baseline=baseline,trials=trials)
    (OUT/'selected.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()
