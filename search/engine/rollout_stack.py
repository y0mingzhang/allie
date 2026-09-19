"""Diagnose Monte Carlo noise and combine rollout with searched values on fit games."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from .service import ROOT,atomic
from .balanced_eval import digest
from .retrieval_common import data
from .analyze_august import means

def fit(features,mask,target,weights):
    def objective(theta):
        logits=np.einsum('nkd,d->nk',features,theta);logits=np.where(mask,logits,-np.inf)
        lp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(lp);ar=np.arange(len(p));w=weights/weights.sum()
        grad=np.einsum('n,nkd,nk->d',w,features,p)-np.einsum('n,nd->d',w,features[ar,target])
        return float(w@(-lp[ar,target])),grad
    dim=features.shape[-1];initial=[1,1]+[0]*(dim-2)
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(.4,1.6),(0,40)]+[(-40,40)]*(dim-2),options=dict(ftol=1e-11,gtol=1e-7,maxiter=300))
    assert result.success,result
    return result.x

def main():
    start=time.monotonic();out=ROOT/'aug-rollout-v1';plan=json.loads((out/'plan.json').read_text());d=data()
    rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];mask=d['mask'];target=d['target'];n=len(rows);ar=np.arange(n);k=mask.shape[1]
    root=np.zeros((n,2432));qs=np.zeros((len(plan['budgets']),n,k));var=np.zeros_like(qs);cost=np.zeros((len(plan['budgets']),n))
    for lo in range(0,n,plan['roots_per_batch']):
        with np.load(out/'human_rollout'/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1];np.testing.assert_array_equal(z['game'],games[lo:hi])
            root[lo:hi]=z['z'];qs[:,lo:hi,:kk]=z['q'];var[:,lo:hi,:kk]=z['variance'];cost[:,lo:hi]=z['evaluated_nodes']
    cq=np.zeros((n,k));cz=np.zeros((n,2432))
    for lo in range(0,n,1024):
        with np.load(ROOT/'aug-adaptive-root-v1/quota_static'/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1];cq[lo:hi,:kk]=z['q'][-1];cz[lo:hi]=z['z']
    logits=cz[:,378:2346][ar[:,None],d['ids']];logits=np.where(mask,logits,0.)
    prior=softmax(np.where(mask,logits,-np.inf),axis=1)
    diagnostics={};variants={'search':(np.stack([logits,cq],axis=-1),d['cost'])}
    for j,h in enumerate(plan['budgets']):
        q=qs[j];mu=(prior*q).sum(1,keepdims=True);between=(prior*(q-mu)**2).sum(1)
        estimator_var=var[j]/(plan['replicas']-1)
        noise=(prior*(1-prior)*estimator_var).sum(1);signal=np.maximum(between-noise,0.)
        shrink=signal[:,None]/np.maximum(signal[:,None]+estimator_var,1e-30)
        denoised=mu+shrink*(q-mu)
        diagnostics[str(h)]=dict(policy_weighted_mean_estimator_variance=float((prior*estimator_var).sum(1).mean()),
            observed_action_variance=float(between.mean()),estimated_noise_in_action_variance=float(noise.mean()),
            positive_debiased_signal_fraction=float((signal>0).mean()),
            note='Four rollout samples; assumes independent sampling across root actions. At horizon1 all replicas share the exact same critic, so variance0 measures no rollout noise, not critic accuracy.')
        if h>1:
            variants[f'search_plus_rollout{h}']=(np.stack([logits,cq,q],-1),d['cost']+cost[j])
            variants[f'search_plus_denoised{h}']=(np.stack([logits,cq,denoised],-1),d['cost']+cost[j])
    records={};losses={}
    for name,(x,nodes) in variants.items():
        def predict(train):
            p=np.zeros((n,k));parameters={}
            for group in range(4):
                tr=train&(cells%4==group);pred=cells%4==group;count=np.bincount(cells[tr],minlength=16)
                theta=fit(x[tr],mask[tr],target[tr],1/count[cells[tr]]);parameters[str(group)]=theta.tolist()
                a=np.einsum('nkd,d->nk',x[pred],theta);p[pred]=softmax(np.where(mask[pred],a,-np.inf),axis=1)
            return p,parameters
        p,parameters=predict(fm);loss=-np.log(p[ar,target]);oof=np.full(n,np.nan)
        for f in range(3):
            pp,_=predict(fm&(cv!=f));val=fm&(cv==f);oof[val]=-np.log(pp[ar[val],target[val]])
        conf=means(loss[~fm],cells[~fm]);cc=means(oof[fm],cells[fm])
        records[name]=dict(parameters=parameters,training_equivalent_cm=None,mean_nodes=float(nodes.mean()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        losses[name]=loss
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['search'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['confirmation_delta_vs_search']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August paired fit-game CV and confirmation, potentially training-seen; no golden CM. Conservative combined-tree cost.',
        fit_cv_selected=selected,results=records,rollout_noise_diagnostics=diagnostics,seconds=time.monotonic()-start,
        source_sha256=digest(Path(__file__)),sample_sha256=plan['sample_sha256'])
    atomic(out/'stack.json',report)
    with (out/'stack-scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,x in records.items():print(name,x['confirmation']['macro_ce'],x['confirmation']['expert_ce'],x['mean_nodes'],flush=True)
    print('SELECTED',selected,'DIAGNOSTICS',diagnostics,flush=True)
if __name__=='__main__':main()
