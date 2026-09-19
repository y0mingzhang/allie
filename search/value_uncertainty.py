"""Marginalize uncertainty along observed search refinement; frozen model only.

Successive search changes are an instability proxy, NOT posterior samples or
ground-truth errors. Compare zero-mean marginalization to signed drift and a
diagonal, second-order softmax correction on the same 1000-node policies.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from .lapse_screen import ROOT,digest,atomic,means

GX,GW=np.polynomial.hermite.hermgauss(9)
GX=GX*np.sqrt(2);GW=GW/np.sqrt(np.pi)


def policy(p,d,mask,kind,a):
    if a==0:return p.copy()
    logp=np.where(mask,np.log(np.maximum(p,1e-300)),-np.inf)
    if kind=='rank1':
        # Exact quadrature of the declared rank-one Gaussian, not of an
        # independently calibrated uncertainty posterior.
        return sum(w*softmax(logp+a*x*d,axis=1) for x,w in zip(GX,GW))
    if kind=='diagonal':
        # E softmax(z+e), e_i independent: relative 2nd-order correction is
        # .5[(1-2p_i)var_i + common_term]. Normalization removes common_term.
        correction=.5*(1-2*p)*d*d
        return softmax(logp+a*correction,axis=1)
    assert kind=='drift'
    return softmax(logp+a*d,axis=1)


def test():
    rng=np.random.default_rng(733);p=softmax(rng.normal(size=(5,4)),axis=1)
    d=rng.normal(size=p.shape);mask=np.ones_like(p,bool)
    for kind in ('rank1','diagonal','drift'):
        np.testing.assert_array_equal(policy(p,d,mask,kind,0),p)
        q=policy(p,d,mask,kind,1);np.testing.assert_allclose(q.sum(1),1,atol=1e-14)
        assert (q>0).all()
    np.testing.assert_allclose(policy(p,d,mask,'rank1',.7),policy(p,-d,mask,'rank1',.7),atol=1e-15)
    # Independent central finite differences verify the diagonal formula locally.
    z=np.log(p);sigma=np.abs(d);eps=1e-3;exact=np.zeros_like(p)
    for j in range(p.shape[1]):
        e=np.zeros_like(p);e[:,j]=eps*sigma[:,j]
        exact+=(softmax(z+e,axis=1)+softmax(z-e,axis=1)-2*p)/2
    np.testing.assert_allclose(policy(p,sigma,mask,'diagonal',eps**2)-p,exact,atol=2e-12)
    print('PASS exact zero, full support, normalization, sign symmetry, independent second derivative',flush=True)


def main():
    test();start=time.monotonic();out=ROOT/'aug-value-uncertainty-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-budget-surface-v2';sp=source/'policies.npz';meta=source/'results.json'
    with np.load(sp) as f:
        parents=f['policy'][-1].astype(float);delta=f['q'][-1]-f['q'][-2]
        mask,target,cells,games,fm,cv,nodes=(f[k] for k in ('mask','target','cells','games','fit','cv','nodes'))
        nodes=nodes[-1]
    n,k=mask.shape;ar=np.arange(n);assert not set(games[fm])&set(games[~fm])
    params=json.loads(meta.read_text())['fold_parameters']['1000']
    menus=[('parent',None,False)]+[(kind+('_elo' if by else ''),kind,by) for kind in ('rank1','diagonal','drift') for by in (False,True)]
    plan=dict(sources=dict(script=digest(Path(__file__)),surface=digest(sp),calibration=digest(meta)),menus=menus,
        proxy='d=beta*(Q1000-Q512), centered under the parent. Beta and parent use their matching game-fold fits. d estimates instability, not an independently validated posterior variance.',
        methods='rank1: E_Z softmax(log p + a*d*Z), Z standard Gaussian,9point quadrature,a[0,3]. diagonal: normalized second-order correction exp(.5*a*(1-2p)*d^2),a[0,4]. drift: softmax(log p+a*d),a[-2,2]. Each shared or4Elo groups. Zero exactly recovers parent. All parameters fit by equal-cell CE within the original game folds.',
        validation='Select by game CV; report every arm on reused August development. No golden access; no external memory; no neural weight updates. Q512 is a free prefix snapshot of the same1000 tree, so no extra NN queries.',
        interpretation='This is a conditional predictive-uncertainty hypothesis. Diagonal formula is a local approximation. No claim that search samples are independent or the proxy is a calibrated error bar.')
    pp=out/'plan.json';plan=json.loads(json.dumps(plan))
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    records,losses={},{}
    for name,kind,by in menus:
        group=cells%4 if by else np.zeros(n,int);oof=np.full(n,np.nan);fitted=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            p=np.where(mask,np.maximum(parents[vi],1e-300),0.);p/=p.sum(1,keepdims=True)
            beta=np.array([params[vi]['root'][str(g)]['beta'] for g in cells%4])
            d=beta[:,None]*delta;d-=np.sum(p*d,axis=1,keepdims=True);d=np.where(mask,d,0.)
            pred=p.copy();coeff={}
            for g in np.unique(group):
                tr=train&(group==g);take=group==g
                count=np.bincount(cells[tr],minlength=16);w=1/count[cells[tr]];w/=w.sum()
                a=0.
                if kind is not None:
                    pt,dt,mt,tt=p[tr],d[tr],mask[tr],target[tr];ix=np.arange(tr.sum())
                    objective=lambda x:float(-w@np.log(np.maximum(policy(pt,dt,mt,kind,x)[ix,tt],1e-300)))
                    bounds={'rank1':(0.,3.),'diagonal':(0.,4.),'drift':(-2.,2.)}[kind]
                    opt=minimize_scalar(objective,bounds=bounds,method='bounded',options=dict(xatol=1e-6));assert opt.success
                    a=min((0.,float(opt.x),*bounds),key=objective)
                    pred[take]=policy(p[take],d[take],mask[take],kind,a)
                coeff[str(int(g))]=a
            np.testing.assert_allclose(pred.sum(1),1,atol=1e-13);assert (pred[mask]>0).all()
            ll=-np.log(pred[ar,target]);fitted.append(dict(fold=fold,coefficients=coeff))
            if fold is None:losses[name]=ll
            else:oof[fm&(cv==fold)]=ll[fm&(cv==fold)]
        a,b=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(parameters=fitted,training_equivalent_cm=None,mean_nodes=float(nodes.mean()),
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
        print(name,records[name]['confirmation']['macro_ce'],records[name]['confirmation']['expert_ce'],flush=True)
    selected={m:min(records,key=lambda key:records[key]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    draws=np.random.default_rng(91343).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@count;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['parent'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_parent']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    result=dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,plan_sha256=digest(pp))
    atomic(out/'results.json',result)
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
