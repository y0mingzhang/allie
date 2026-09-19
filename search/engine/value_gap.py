"""Same-node test of nonlinear value-gap calibration after the search mixture.

A linear value tilt need not capture how humans treat several nearly equivalent
moves versus clearly inferior moves. This tests a small smooth piecewise curve;
it updates calibration scalars only, keeping model weights and trees fixed.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_august import means


def objective(theta,x,logbase,mask,target,weights=None,ridge=0.):
    logits=np.where(mask,logbase+np.einsum('nkd,d->nk',x,theta),-np.inf)
    logp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(logp)
    if weights is None:return p
    ar=np.arange(len(target));residual=np.einsum('nk,nkd->nd',p,x)-x[ar,target]
    return float(-weights@logp[ar,target]+.5*ridge*(theta@theta)),weights@residual+ridge*theta


def test():
    rng=np.random.default_rng(7181);x=rng.normal(size=(30,7,5))
    mask=rng.random((30,7))>.2;mask[:,0]=True;target=np.zeros(30,int)
    base=np.where(mask,rng.normal(size=(30,7)),0.);theta=rng.normal(size=5)*.1;w=np.ones(30)/30
    loss,grad=objective(theta,x,base,mask,target,w,.01);finite=[]
    for j in range(5):
        a,b=theta.copy(),theta.copy();a[j]+=1e-6;b[j]-=1e-6
        finite.append((objective(a,x,base,mask,target,w,.01)[0]-objective(b,x,base,mask,target,w,.01)[0])/2e-6)
    np.testing.assert_allclose(grad,finite,atol=1e-9)
    np.testing.assert_allclose(objective(np.zeros(5),x,base,mask,target),softmax(np.where(mask,base,-np.inf),axis=1),atol=1e-15)
    print('PASS value-gap residual gradient, masking and zero-correction identity',flush=True)


def main():
    test();start=time.monotonic();out=ROOT/'aug-value-gap-v1';out.mkdir(exist_ok=True)
    surface=ROOT/'aug-budget-surface-v2';d=read('aug-tune-expanded-v1')
    cells,games,fm,cv,mask,target=(d[k] for k in ('cells','games','fit','cv','mask','target'))
    n=len(cells);ar=np.arange(n)
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games)
        parents=f['policy'][-1].astype(float);q=f['q'][-1];nodes=f['nodes'][-1]
    menus=[('parent','parent',0.,False)]
    for kind in ('linear','curve'):
        for separate in (False,True):
            for ridge in (.001,.01):menus.append((kind+('_elo' if separate else '_shared')+str(ridge),kind,ridge,separate))
    plan=dict(source_sha256=digest(Path(__file__)),surface_sha256=digest(surface/'results.json'),
        sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),policies_sha256=digest(surface/'policies.npz'),
        knots=[.025,.1,.3],smooth_width=.025,menus=menus,
        formula='pi_new proportional to pi_parent*exp(theta*basis). Basis: log parent policy, Q-maxQ, plus smooth hinges .025*softplus((Q-maxQ+knot)/.025). Parent is final conditional-strength+temperature stack. Shared or Elo-group coefficients, ridge regularized.',
        validation='All scalars and feature scales refitted inside3 fit-game folds using independently fitted parent policies; all arms scored on disjoint August confirmation. No neural weights changed; no new model nodes.',
        hypothesis='A human-choice correction may penalize clearly inferior moves differently from near-equal alternatives; the linear residual control distinguishes curvature from generic recalibration. This is a calibration hypothesis, not an established behavioral law.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    gap=q-np.where(mask,q,-np.inf).max(1)[:,None];gap=np.where(mask,gap,0.)
    hinges=[np.logaddexp(0.,(gap+k)/.025)*.025 for k in plan['knots']]
    records,losses={},{}
    for name,kind,ridge,separate in menus:
        oof=np.full(n,np.nan);versions=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            p=parents[vi];p/=p.sum(1,keepdims=True)
            logbase=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
            if kind=='parent':new=p;params={}
            else:
                xx=np.stack([logbase,gap,*hinges] if kind=='curve' else [logbase,gap],axis=2)
                # Per-query shifts cancel in softmax; center to improve conditioning.
                xx-=np.einsum('nk,nkd->nd',p,xx)[:,None,:]
                x=np.where(mask[:,:,None],xx,0.)
                params={};new=np.zeros_like(p);groups=cells%4 if separate else np.zeros(n,int)
                for g in np.unique(groups):
                    tr=train&(groups==g);take=groups==g
                    ct=np.bincount(cells[tr],minlength=16);w=1/ct[cells[tr]];w/=w.sum()
                    scale=np.sqrt(w@np.einsum('nk,nkd->nd',p[tr],x[tr]**2));scale=np.maximum(scale,1e-6)
                    xs=x/scale
                    opt=minimize(lambda t:objective(t,xs[tr],logbase[tr],mask[tr],target[tr],w,ridge),np.zeros(x.shape[2]),jac=True,
                        method='L-BFGS-B',bounds=[(-3,3)]*x.shape[2],options=dict(ftol=1e-11,gtol=1e-7,maxiter=250))
                    assert opt.success,opt
                    new[take]=objective(opt.x,xs[take],logbase[take],mask[take],target[take])
                    params[str(g)]=dict(theta=opt.x.tolist(),scale=scale.tolist())
            assert np.isfinite(new).all() and (new[mask]>0).all()
            np.testing.assert_allclose(new.sum(1),1.,atol=1e-13)
            ll=-np.log(new[ar,target]);versions.append(dict(fold=fold,groups=params))
            if fold is None:losses[name]=ll
            else:oof[fm&(cv==fold)]=ll[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(kind=kind,ridge=ridge,separate_elo=separate,parameters=versions,training_equivalent_cm=None,
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),mean_nodes=float(nodes.mean()))
        print(name,ce.mean(),ce[3::4].mean(),records[name]['fit_game_cv'],flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(95184).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count
    for name,rec in records.items():
        delta=losses[name]-losses['parent'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['delta_vs_parent']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp)))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
