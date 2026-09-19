"""Allie-like reverse-KL root output inside the current mixture/calibration stack."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .diff_backup_native import load
from .fit_policy import fit, loss_gradient

from .analyze_player_search import FACTORS
from .analyze_august import means
from .conditional_mixture import objective as gate_objective
from .temperature_stack import evaluate as temperature


def components(z,q,mask,params,group):
    result=np.zeros((len(FACTORS),*mask.shape))
    for g in np.unique(group):
        take=group==g;p=params[str(int(g))]
        for i,factor in enumerate(FACTORS):
            result[i,take],_=loss_gradient([p['alpha'],p['beta']*factor],z[take],q[take],mask[take],np.zeros(take.sum(),int),'reverse',return_policy=True)
    assert np.isfinite(result).all() and (result[:,mask]>0).all()
    np.testing.assert_allclose(result.sum(2),1.,atol=1e-12)
    return result


def main():
    start=time.monotonic();budgets=[128,256,512,1000]
    out=ROOT/'aug-budget-reverse-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-expanded-search-v1'
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n,ar=len(rows),np.arange(len(rows));group=cells%4
    parent=ROOT/'aug-budget-surface-v2/policies.npz'
    plan=dict(budgets=budgets,sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        parent_policies_sha256=digest(parent),source_sha256=digest(Path(__file__)),
        fit_policy_sha256=digest(Path(__file__).with_name('fit_policy.py')),
        recipe='Same cached root coverage trees, soft Bellman Q and per-budget fits, changing root regularization to reverse KL(p||pi). Refit Elo alpha/beta, all0.001 strength gate and state0.01 temperature, independently inside three game folds. Full-support factor mixture [.25,.5,1,2,4].',
        limitation='Not released traversal, no inference changes or extra NN nodes. August training-seen separate fit/confirmation; CM pending golden. Reuses parent Q/root arrays exactly.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    with np.load(parent) as f:
        for name,x in [('games',games),('ids',ids),('mask',mask),('target',target),('budgets',budgets)]:np.testing.assert_array_equal(f[name],x)
        root,q,nodes=(f[k] for k in ('root','q','nodes'))
    inventory=ROOT/'aug-tune-v1';manifest=json.loads((inventory/'manifest.json').read_text())
    for name in ('strat.npz','feats.npz'):assert digest(inventory/name)==manifest['files_sha256'][name]
    with np.load(inventory/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(inventory/'feats.npz') as f:feats=f['feats']
    seconds=[]
    for r in rows:
        rr,cc=r['row'],r['column'];assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(r['prefix']):cc],r['prefix']);seconds.append(feats[rr,cc-1,0])
    seconds=np.array(seconds);known=seconds>=0
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    prior=softmax(np.where(mask,logits,-np.inf),axis=1)
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)];predtime=np.log1p(softmax(root[:,2350:2413],axis=1)@tc)
    records,losses,all_params,policy_versions={}, {}, {}, []
    for bi,b in enumerate(budgets):
        qm=(prior*q[bi]).sum(1);qstd=np.sqrt((prior*(q[bi]-qm[:,None])**2).sum(1))
        features=np.column_stack([entropy,np.log(.01+qstd),[len(r['prefix'])-11 for r in rows],predtime,
            np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known])
        versions=[];oofs={k:np.full(n,np.nan) for k in ('fixed','gate','temperature')};all_params[str(b)]=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            params={}
            for g in range(4):
                tr=train&(group==g);ct=np.bincount(cells[tr],minlength=16)
                params[str(g)]=fit(logits[tr],q[bi,tr],mask[tr],target[tr],'reverse',1/ct[cells[tr]])
                assert params[str(g)]['converged']
            comp=components(logits,q[bi],mask,params,group)
            fixed=np.einsum('a,ank->nk',softmax(-.5*np.log(FACTORS)**2),comp)
            mean=features[train].mean(0);scale=np.maximum(features[train].std(0),1e-6)
            x=np.c_[np.ones(n),np.clip((features-mean)/scale,-3,3)]
            ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
            pt=comp[:,ar,target].T
            opt=minimize(lambda t:gate_objective(t,x[train],pt[train],w,.001),np.zeros(8),jac=True,
                method='L-BFGS-B',bounds=[(-2,2)]*8,options=dict(ftol=1e-11,gtol=1e-7,maxiter=200));assert opt.success,opt
            mu=x@opt.x;mu[~known]=0.
            gw=softmax(-.5*np.log(FACTORS)[None,:]**2+mu[:,None]*np.log(FACTORS),axis=1)
            gp=np.einsum('na,ank->nk',gw,comp)
            logp=np.where(mask,np.log(np.maximum(gp,1e-300)),0.)
            tx=x[:,:5]
            tp=minimize(lambda t:temperature(t,tx[train],logp[train],mask[train],target[train],w,.01),np.zeros(5),jac=True,
                method='L-BFGS-B',bounds=[(-1,1)]*5,options=dict(ftol=1e-11,gtol=1e-7,maxiter=250));assert tp.success,tp
            final=temperature(tp.x,tx,logp,mask,target);versions.append(final)
            all_params[str(b)].append(dict(fold=fold,root=params,mean=mean.tolist(),scale=scale.tolist(),gate=opt.x.tolist(),temperature=tp.x.tolist()))
            for key,p in [('fixed',fixed),('gate',gp),('temperature',final)]:
                ll=-np.log(p[ar,target])
                if fold is None:losses[f'{b}_{key}']=ll
                else:oofs[key][fm&(cv==fold)]=ll[fm&(cv==fold)]
        policy_versions.append(np.array(versions,dtype=np.float32))
        for key in ('fixed','gate','temperature'):
            ce,cc=means(losses[f'{b}_{key}'][~fm],cells[~fm]),means(oofs[key][fm],cells[fm])
            records[f'{b}_{key}']=dict(budget=b,stage=key,parameters=all_params[str(b)][0],training_equivalent_cm=None,
                mean_nodes=float(nodes[bi].mean()),expert_mean_nodes=float(nodes[bi,cells%4==3].mean()),
                confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
                fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
            print(b,key,ce.mean(),ce[3::4].mean(),nodes[bi].mean(),flush=True)
    old=json.loads((ROOT/'aug-budget-surface-v2/results.json').read_text())
    drift={name:{field:{m:r[field][m]-old['results'][name][field][m] for m in ('macro_ce','expert_ce')} for field in ('confirmation','fit_game_cv')} for name,r in records.items()}
    with (out/'policies.npz').open('wb') as f:
        np.savez_compressed(f,budgets=budgets,policy=np.array(policy_versions),q=q,root=root,nodes=nodes,ids=ids,mask=mask,target=target,cells=cells,games=games,fit=fm,cv=cv)
    atomic(out/'results.json',dict(stage=plan['recipe'],results=records,fold_parameters=all_params,analysis_seconds=time.monotonic()-start,
        plan_sha256=digest(pp),policies_sha256=digest(out/'policies.npz'),reverse_minus_forward_ce=drift))


def run(oracle,spec):
    from .fit_policy import test
    test();main()
    return dict(output=str(ROOT/'aug-budget-reverse-v1'))

if __name__=='__main__':main()
