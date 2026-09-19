"""Fast differentiable calibration of forward/reverse KL search policies.

Only two fitted scalars. Reverse-KL gradients use implicit differentiation of
the normalization equation, avoiding repeated finite-difference solves.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def loss_gradient(theta,z,q,mask,target,direction,weights=None,return_policy=False):
    alpha,beta=theta;z=np.where(mask,np.asarray(z,float),0.);q=np.where(mask,np.asarray(q,float),0.)
    ar=np.arange(len(z));weights=np.ones(len(z))/len(z) if weights is None else weights/weights.sum()
    if direction=='forward':
        a=np.where(mask,alpha*z+beta*q,-np.inf);logp=a-logsumexp(a,axis=1,keepdims=True);p=np.exp(logp)
        loss=-logp[ar,target]
        grad=np.array([weights@((p*z).sum(1)-z[ar,target]),weights@((p*q).sum(1)-q[ar,target])])
    else:
        assert direction=='reverse'
        a=np.where(mask,alpha*z,-np.inf);logprior=a-logsumexp(a,axis=1,keepdims=True);prior=np.exp(logprior)
        v=beta*q;lo=np.where(mask,v+prior,-np.inf).max(1);hi=np.where(mask,v,-np.inf).max(1)+1.
        for _ in range(48):
            mid=(lo+hi)/2;d=np.where(mask,mid[:,None]-v,1.)
            above=(prior/d).sum(1)>1;lo=np.where(above,mid,lo);hi=np.where(above,hi,mid)
        d=np.where(mask,((lo+hi)/2)[:,None]-v,1.);p=prior/d
        # F(t,alpha,beta)=sum prior/(t-beta*q)-1=0.
        weighted=prior/(d*d);den=weighted.sum(1)
        dt_alpha=(p*(z-(prior*z).sum(1)[:,None])).sum(1)/den
        dt_beta=(weighted*q).sum(1)/den
        loss=-logprior[ar,target]+np.log(d[ar,target])
        grad=np.array([weights@(-z[ar,target]+(prior*z).sum(1)+dt_alpha/d[ar,target]),
                       weights@((dt_beta-q[ar,target])/d[ar,target])])
    if return_policy:return p,loss
    return float(weights@loss),grad


def fit(z,q,mask,target,direction,weights=None):
    objective=lambda x:loss_gradient(x,z,q,mask,target,direction,weights)
    result=minimize(objective,[1.,1.],jac=True,method='L-BFGS-B',bounds=[(.4,1.6),(0.,40.)],
                    options=dict(maxiter=100,ftol=1e-11,gtol=1e-7))
    assert np.isfinite(result.fun),result
    return dict(alpha=float(result.x[0]),beta=float(result.x[1]),converged=bool(result.success),iterations=int(result.nit))


def test():
    from .adaptive_policy import output
    rng=np.random.default_rng(829);z=rng.normal(size=(23,11));q=rng.uniform(-1,1,size=z.shape)
    mask=rng.random(z.shape)>.2;mask[:,0]=True;target=np.zeros(len(z),int)
    for direction in ('forward','reverse'):
        for theta in ([.91,0.],[.84,4.2],[1.17,25.]):
            p,nll=loss_gradient(theta,z,q,mask,target,direction,return_policy=True)
            ref=output(z,q,mask,*theta,direction=direction)
            np.testing.assert_allclose(p,ref,atol=2e-11,rtol=2e-11)
            f,g=loss_gradient(theta,z,q,mask,target,direction)
            np.testing.assert_allclose(f,nll.mean(),atol=1e-12)
            if theta[1]>0:
                finite=[]
                for j in range(2):
                    a=np.array(theta);b=a.copy();a[j]+=1e-5;b[j]-=1e-5
                    finite.append((loss_gradient(a,z,q,mask,target,direction)[0]-loss_gradient(b,z,q,mask,target,direction)[0])/2e-5)
                np.testing.assert_allclose(g,finite,atol=2e-7,rtol=2e-6)
            assert np.all(p[mask]>0) and np.all(p[~mask]==0)
    print('PASS policy identity, beta=0 limit, analytic implicit gradients, full legal support')


if __name__=='__main__':test()
