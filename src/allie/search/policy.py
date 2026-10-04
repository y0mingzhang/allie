"""Calibrated output policy, Elo budget routing, and the Allie method's KL-regularized output."""
import numpy as np
from scipy.special import softmax, logsumexp
from .board import predicted_seconds
FACTORS=np.array([.25,.5,1.,2.,4.])

def components(z,q,mask,params,group):
    a=np.array([params[str(g)]['alpha'] for g in group]);b=np.array([params[str(g)]['beta'] for g in group])
    logits=a[None,:,None]*z[None,:,:]+(FACTORS[:,None]*b[None,:])[:,:,None]*q[None,:,:]
    return softmax(np.where(mask[None,:,:],logits,-np.inf),axis=2)

def temperature(theta, x, logp, mask):
    eta = np.clip(1+x@theta, .5, 2.)
    lp = np.where(mask, eta[:, None]*logp, -np.inf)
    lp -= logsumexp(lp, axis=1, keepdims=True)
    return np.exp(lp)

def policy(rows, root, q, ids, mask, seconds, params):
    n=len(rows);ar=np.arange(n);root=np.asarray(root,dtype=float)
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    prior=softmax(np.where(mask,logits,-np.inf),axis=1)
    mean=(prior*q).sum(1)
    spread=np.sqrt((prior*(q-mean[:,None])**2).sum(1))
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    known=seconds>=0
    f=np.column_stack([entropy,np.log(.01+spread),[len(r['prefix'])-11 for r in rows],
        np.log1p(predicted_seconds(root)),
        np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known])
    x=np.c_[np.ones(n),np.clip((f-params['mean'])/params['scale'],-3,3)]
    comp=components(logits,q,mask,params['root'],np.array([r['cell']%4 for r in rows]))
    mu=x@np.array(params['gate']);mu[~known]=0.
    weight=softmax(-.5*np.log(FACTORS)[None,:]**2+mu[:,None]*np.log(FACTORS),axis=1)
    p=np.einsum('na,ank->nk',weight,comp)
    logp=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
    p=temperature(np.array(params['temperature']),x[:,:5],logp,mask)
    assert np.isfinite(p).all() and (p[mask]>0).all()
    np.testing.assert_allclose(p.sum(1),1.,atol=1e-13)
    return p

def route(cells, params, budgets):
    """Index into budgets per position, from the mover's Elo band alone."""
    assert params['kind']=='elo'
    x=np.c_[np.ones(len(cells)),np.clip((np.eye(4)[cells%4][:,1:]-params['mean'])/params['scale'],-3,3)]
    coef=np.array(params['coef'])
    assert np.isfinite(x).all() and np.isfinite(coef).all()
    return np.argmin(x@coef + params['penalty']*np.asarray(budgets)[None, :]/1000., axis=1)

def output(logits, values, legal, alpha, beta):
    """argmax_pi E_pi Q - KL(prior || pi) / beta, prior = softmax(alpha * logits) over legal moves.

    beta is independent of simulation count; beta=0 is the prior. Solved by FP64 bisection.
    """
    logits, values = np.asarray(logits, np.float64), np.asarray(values, np.float64)
    legal = np.asarray(legal, bool)
    assert logits.shape == values.shape == legal.shape and (legal.sum(1) > 0).all()
    assert beta >= 0 and alpha > 0
    z = np.where(legal, alpha * logits, -np.inf)
    logp = z - logsumexp(z, axis=1, keepdims=True)
    if beta == 0:
        return np.exp(logp)
    prior = np.exp(logp)
    q = np.where(legal, values, -np.inf)
    lam = 1. / beta
    lo = (q + lam * prior).max(1)
    hi = q.max(1) + lam
    for _ in range(80):
        mid = (lo + hi) / 2
        p = lam * prior / (mid[:, None] - q)
        above = p.sum(1) > 1
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    p = lam * prior / (((lo + hi) / 2)[:, None] - q)
    return p / p.sum(1, keepdims=True)
