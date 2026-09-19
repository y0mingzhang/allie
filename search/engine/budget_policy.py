"""The frozen calibrated policy shared by fixed and adaptive budget runs."""
import numpy as np
from scipy.special import softmax
from .analyze_expanded import components
from .analyze_player_search import FACTORS
from .temperature_stack import evaluate


def policy(rows, root, q, ids, mask, seconds, params):
    n=len(rows);ar=np.arange(n);root=np.asarray(root,dtype=float)
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    prior=softmax(np.where(mask,logits,-np.inf),axis=1)
    mean=(prior*q).sum(1)
    spread=np.sqrt((prior*(q-mean[:,None])**2).sum(1))
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    known=seconds>=0
    f=np.column_stack([entropy,np.log(.01+spread),[len(r['prefix'])-11 for r in rows],
        np.log1p(softmax(root[:,2350:2413],axis=1)@tc),
        np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known])
    x=np.c_[np.ones(n),np.clip((f-params['mean'])/params['scale'],-3,3)]
    comp=components(logits,q,mask,params['root'],np.array([r['cell']%4 for r in rows]))
    mu=x@np.array(params['gate']);mu[~known]=0.
    weight=softmax(-.5*np.log(FACTORS)[None,:]**2+mu[:,None]*np.log(FACTORS),axis=1)
    p=np.einsum('na,ank->nk',weight,comp)
    logp=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
    p=evaluate(np.array(params['temperature']),x[:,:5],logp,mask,np.zeros(n,int))
    assert np.isfinite(p).all() and (p[mask]>0).all()
    np.testing.assert_allclose(p.sum(1),1.,atol=1e-13)
    return p


def route(cells, feat, params, budgets):
    from .value_of_compute import choose
    category=np.eye(4)[cells%4][:,1:]
    raw=category if params['kind']=='elo' else np.c_[category,np.eye(4)[cells//4][:,1:],feat]
    x=np.c_[np.ones(len(cells)),np.clip((raw-params['mean'])/params['scale'],-3,3)]
    return choose(x,np.array(params['coef']),params['penalty'],budgets)
