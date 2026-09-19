"""Cross-fitted scalar policy/value calibration from frozen hidden-state PCs.

Only low-dimensional calibration coefficients are fitted. Checkpoint weights,
search trees, and the human-move labels used by the underlying model stay fixed.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_august import means


def objective(theta,x,lp,q,mask,target,w=None,ridge=0.,joint=True):
    d=x.shape[1]
    raw=1+x@theta[:d]; alpha=np.clip(raw,.5,2.)
    value=x@theta[d:] if joint else np.zeros(len(x))
    beta=np.clip(value,-3.,3.)
    z=np.where(mask,alpha[:,None]*lp+beta[:,None]*q,-np.inf)
    logp=z-logsumexp(z,axis=1,keepdims=True);p=np.exp(logp)
    if w is None:return p
    ar=np.arange(len(x))
    ga=x.T@(w*((p*lp).sum(1)-lp[ar,target])*(raw>.5)*(raw<2.))
    if joint:
        gb=x.T@(w*((p*q).sum(1)-q[ar,target])*(value>-3)*(value<3))
        grad=np.r_[ga,gb]
    else:grad=ga
    return float(-w@logp[ar,target]+.5*ridge*(theta@theta)),grad+ridge*theta


def test():
    rng=np.random.default_rng(8192);x=np.c_[np.ones(23),rng.normal(size=(23,3))]
    mask=rng.random((23,7))>.2;mask[:,0]=True
    lp=np.where(mask,rng.normal(size=(23,7)),0.);q=rng.normal(size=lp.shape)
    target=np.zeros(23,int);w=np.ones(23)/23
    for joint in (False,True):
        theta=rng.normal(size=4*(1+joint))*.02
        _,g=objective(theta,x,lp,q,mask,target,w,.01,joint)
        fd=[]
        for j in range(len(theta)):
            a,b=theta.copy(),theta.copy();a[j]+=1e-6;b[j]-=1e-6
            fd.append((objective(a,x,lp,q,mask,target,w,.01,joint)[0]-objective(b,x,lp,q,mask,target,w,.01,joint)[0])/2e-6)
        np.testing.assert_allclose(g,fd,atol=1e-9,rtol=1e-6)
        np.testing.assert_allclose(objective(np.zeros_like(theta),x,lp,q,mask,target,joint=joint),softmax(np.where(mask,lp,-np.inf),axis=1),atol=1e-15)
    print('PASS hidden calibration masking, gradient, zero correction identity',flush=True)


def main():
    test();start=time.monotonic();out=ROOT/'aug-hidden-calibration-v2';out.mkdir(exist_ok=True)
    d=read('aug-tune-expanded-v1')
    cells,games,fm,cv,mask,target=(d[k] for k in ('cells','games','fit','cv','mask','target'))
    n=len(cells);ar=np.arange(n);surface=ROOT/'aug-budget-surface-v2'
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games);np.testing.assert_array_equal(f['target'],target)
        parents=f['policy'][-1].astype(float);q=f['q'][-1];nodes=f['nodes'][-1]
    hp=ROOT/'aug-retrieval-large-v1/roots.npz'
    with np.load(hp) as f:
        np.testing.assert_array_equal(f['game'],games);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in d['rows']])
        hidden=f['hidden'].astype(float)
    assert np.isfinite(hidden).all()
    menus=[('parent',0,0.,False),('global_joint',0,.01,True)]
    for pcs in (8,32):
        for joint in (False,True):
            for ridge in (.01,.1):menus.append((f"pc{pcs}_{'joint' if joint else 'temp'}_{ridge}",pcs,ridge,joint))
    plan=dict(source_sha256=digest(Path(__file__)),sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        policies_sha256=digest(surface/'policies.npz'),hidden_sha256=digest(hp),menus=menus,
        method='Residual temperature alpha=clip(1+x*a,.5,2), value coefficient beta=clip(x*b,-3,3); policy proportional to parent^alpha * exp(beta * centered_Q / fitted_global_Q_scale). x is intercept plus up to32 whitened hidden-state PCs clipped to3. PCA and all calibration coefficients fitted strictly inside each training-game fold, macro cell weighted.',
        validation='Same4 independently fitted parents as budget surface1000, three fit-game CV folds, disjoint August confirmation. Potentially model-training-seen. No golden selection. No neural-weight updates. Fixed checkpoint hidden states are causal pre-target activations.',
        cost='Cached hidden states currently require1 additional full-prefix call per query, conservatively charged. Search nodes unchanged. Integrated root export could remove that call but is not credited here.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    folds=[]
    for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
        ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
        mean=w@hidden[train];center=hidden-mean
        cov=(center[train].T*w)@center[train]
        eig,vec=np.linalg.eigh(cov);order=np.argsort(eig)[::-1][:32]
        basis=vec[:,order]/np.sqrt(np.maximum(eig[order],1e-8))
        feat=np.clip(center@basis,-3,3)
        p=parents[vi];p/=p.sum(1,keepdims=True);lp=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
        qq=np.where(mask,q-(p*q).sum(1)[:,None],0.)
        scale=max(float(np.sqrt(w@((p[train]*qq[train]**2).sum(1)))),1e-6)
        folds.append((fold,train,w,feat,lp,qq/scale,p))
        np.savez(out/f'pca-{vi}.npz',mean=mean,basis=basis,scale=scale,train=train)
    records,losses={},{}
    for name,pcs,ridge,joint in menus:
        oof=np.full(n,np.nan);params=[]
        for vi,(fold,train,w,feat,lp,qq,parent) in enumerate(folds):
            x=np.c_[np.ones(n),feat[:,:pcs]]
            theta=np.zeros(x.shape[1]*(1+joint))
            if name!='parent':
                opt=minimize(lambda t:objective(t,x[train],lp[train],qq[train],mask[train],target[train],w,ridge,joint),theta,jac=True,
                    method='L-BFGS-B',bounds=[(-2,2)]*len(theta),options=dict(ftol=1e-11,gtol=1e-7,maxiter=300))
                assert opt.success,opt;theta=opt.x
                p=objective(theta,x,lp,qq,mask,target,joint=joint)
            else:p=parent
            assert np.isfinite(p).all() and (p[mask]>0).all()
            np.testing.assert_allclose(p.sum(1),1.,atol=1e-13)
            loss=-np.log(p[ar,target]);params.append(dict(fold=fold,theta=theta.tolist(),pca_sha256=digest(out/f'pca-{vi}.npz')))
            if fold is None:losses[name]=loss
            else:oof[fm&(cv==fold)]=loss[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,pcs=pcs,joint=joint,ridge=ridge,training_equivalent_cm=None,
            mean_nodes=float(nodes.mean())+int(pcs>0),extra_full_prefix_queries=int(pcs>0),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
        print(name,records[name]['fit_game_cv'],ce.mean(),ce[3::4].mean(),flush=True)
    ref=json.loads((surface/'results.json').read_text())['results']['1000_temperature']
    # Surface stores float32 policies; expected roundoff in normalized targets is <1e-7.
    for section in ('confirmation','fit_game_cv'):
        for metric in ('macro_ce','expert_ce'):
            np.testing.assert_allclose(records['parent'][section][metric],ref[section][metric],atol=1e-7,rtol=0)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(81819).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['parent'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['delta_vs_parent']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp)))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
