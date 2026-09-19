"""Re-fit the selected small output-calibration pipeline inside each game fold."""
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from .fit_policy import fit
from .analyze_expanded import components
from .analyze_player_search import FACTORS
from .conditional_mixture import objective as gate_objective
from .temperature_stack import evaluate as temperature
from .analyze_august import means


def fit_stack(rows,root,q,ids,mask,target,cells,fm,cv,seconds):
    n=len(rows);ar=np.arange(n);groups=cells%4
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    prior=softmax(np.where(mask,logits,-np.inf),axis=1)
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    mean=(prior*q).sum(1);spread=np.sqrt((prior*(q-mean[:,None])**2).sum(1))
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    known=seconds>=0
    features=np.column_stack([entropy,np.log(.01+spread),[len(r['prefix'])-11 for r in rows],
        np.log1p(softmax(root[:,2350:2413],axis=1)@tc),
        np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known])
    versions=[];parameters=[];oof=np.full(n,np.nan)
    for fold,train in [(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]:
        params={}
        for g in range(4):
            tr=train&(groups==g);ct=np.bincount(cells[tr],minlength=16)
            params[str(g)]=fit(logits[tr],q[tr],mask[tr],target[tr],'forward',1/ct[cells[tr]])
            assert params[str(g)]['converged']
        comp=components(logits,q,mask,params,groups)
        mu=features[train].mean(0);scale=np.maximum(features[train].std(0),1e-6)
        x=np.c_[np.ones(n),np.clip((features-mu)/scale,-3,3)]
        ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
        pt=comp[:,ar,target].T
        gate=minimize(lambda t:gate_objective(t,x[train],pt[train],w,.001),np.zeros(8),jac=True,
            method='L-BFGS-B',bounds=[(-2,2)]*8,options=dict(ftol=1e-11,gtol=1e-7,maxiter=200))
        assert gate.success,gate
        location=x@gate.x;location[~known]=0.
        weights=softmax(-.5*np.log(FACTORS)[None,:]**2+location[:,None]*np.log(FACTORS),axis=1)
        p=np.einsum('na,ank->nk',weights,comp)
        logp=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
        temp=minimize(lambda t:temperature(t,x[train,:5],logp[train],mask[train],target[train],w,.01),np.zeros(5),jac=True,
            method='L-BFGS-B',bounds=[(-1,1)]*5,options=dict(ftol=1e-11,gtol=1e-7,maxiter=250))
        assert temp.success,temp
        final=temperature(temp.x,x[:,:5],logp,mask,target)
        ll=-np.log(final[ar,target])
        if fold is None:loss=ll
        else:oof[fm&(cv==fold)]=ll[fm&(cv==fold)]
        parameters.append(dict(fold=fold,root=params,mean=mu.tolist(),scale=scale.tolist(),gate=gate.x.tolist(),temperature=temp.x.tolist()))
        versions.append(final)
    ce,cc=means(loss[~fm],cells[~fm]),means(oof[fm],cells[fm])
    return dict(parameters=parameters,training_equivalent_cm=None,
        confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
        fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean()))),loss,versions
