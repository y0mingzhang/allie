"""Residual rather than raw-frequency retrieval, with fit-only scalar calibration."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from .service import ROOT,atomic
from .balanced_eval import digest
from .analyze_august import means
from .analyze_retrieval import distribution
from .retrieval_common import data

def policy(base,empirical,expected,ok,strength,kind):
    if kind=='linear_residual':
        p=np.maximum(base+strength*(empirical-expected),.01*base)
    elif kind=='log_ratio':
        # One prior pseudocount: empirical and expected distributions equally
        # weighted; the ratio is 1 when observed and predicted neighbors agree.
        delta=np.log(np.maximum((empirical+expected)/(2*np.maximum(expected,1e-30)),1e-30))
        p=base*np.exp(strength*np.clip(delta,-30,30))
    else:raise ValueError(kind)
    p/=p.sum(1,keepdims=True)
    p[~ok]=base[~ok]
    return p

def fit_strength(base,empirical,expected,ok,kind,target,cells,train):
    ar=np.arange(len(base));count=np.bincount(cells[train],minlength=16);w=1/count[cells[train]];w/=w.sum()
    obj=lambda a:float(w@(-np.log(policy(base,empirical,expected,ok,a,kind)[ar[train],target[train]])))
    # Linear clipping can be nonconvex. A fixed bounded coarse grid plus local
    # refinement around its best point prevents silently assuming unimodality.
    grid=np.linspace(0,2,33);scores=[obj(a) for a in grid];j=int(np.argmin(scores))
    if j==0:return 0.
    lo,hi=grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]
    result=minimize_scalar(obj,bounds=(lo,hi),method='bounded',options=dict(xatol=1e-8));assert result.success
    return float(min([0.,float(grid[j]),float(result.x),2.],key=obj))

def test():
    rng=np.random.default_rng(1228);p=rng.dirichlet(np.ones(5),size=20);e=rng.dirichlet(np.ones(5),size=20)
    ok=np.ones(20,bool);ok[0]=False
    for kind in ('linear_residual','log_ratio'):
        np.testing.assert_allclose(policy(p,e,e,ok,1.3,kind),p,atol=1e-15)
        for a in (0.,.5,2.):
            z=policy(p,e,p,ok,a,kind);assert np.isfinite(z).all() and (z>0).all()
            np.testing.assert_allclose(z.sum(1),1.)
            np.testing.assert_array_equal(z[0],p[0])
        np.testing.assert_allclose(policy(p,e,p,ok,0.,kind),p,atol=1e-15)
    print('PASS residual zero-effect, zero-strength, normalization, positive support, empty fallback')

def main():
    test();start=time.monotonic();out=ROOT/'aug-retrieval-residual-v2';plan=json.loads((out/'plan.json').read_text())
    worker=json.loads((out/'worker.json').read_text());assert worker['plan_sha256']==digest(out/'plan.json')
    with np.load(out/'expected.npz') as z:expected=z['expected'];eids=z['ids'];emask=z['mask']
    with np.load(ROOT/'aug-retrieval-v1/neighbors.npz') as z:sim=z['similarity'];lab=z['label']
    d=data();cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];target=d['target'];n=len(target);ar=np.arange(n)
    np.testing.assert_array_equal(eids,np.where(d['mask'],d['ids']+378,0));np.testing.assert_array_equal(emask,d['mask'])
    controls=d['controls'];records={};losses={}
    def record(name,p,oof,params,base,kernel,kind):
        loss=-np.log(p[ar,target]);conf=means(loss[~fm],cells[~fm]);cvce=means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,base=base,kernel=kernel,kind=kind,training_equivalent_cm=None,
            mean_nodes=float(d['cost'].mean()) if base=='search' else 0.,
            fit_game_cv=dict(macro_ce=float(cvce.mean()),expert_ce=float(cvce[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        losses[name]=loss
    for base,versions in controls.items():
        p,params=versions[0];oof=np.full(n,np.nan)
        for f in range(3):
            val=fm&(cv==f);pp=versions[f+1][0];oof[val]=-np.log(pp[ar[val],target[val]])
        record(base,p,oof,dict(base=params,strength=0.),base,None,None)
        for j,kernel in enumerate(plan['kernels']):
            emp,ok=distribution(sim,lab,d['ids'],d['mask'],kernel['k'],kernel['temperature']);ex=expected[j].astype(float)
            np.testing.assert_allclose(ex[ok].sum(1),1.,atol=2e-6)
            for kind in plan['methods']:
                strength=fit_strength(p,emp,ex,ok,kind,target,cells,fm)
                mixed=policy(p,emp,ex,ok,strength,kind);cvloss=np.full(n,np.nan)
                for f in range(3):
                    pp=versions[f+1][0];val=fm&(cv==f)
                    a=fit_strength(pp,emp,ex,ok,kind,target,cells,fm&(cv!=f));pm=policy(pp,emp,ex,ok,a,kind)
                    cvloss[val]=-np.log(pm[ar[val],target[val]])
                name=f"{base}_{kind}_k{kernel['k']}"
                record(name,mixed,cvloss,dict(base=params,strength=strength),base,kernel,kind)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses[rec['base']];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['confirmation_delta_vs_base']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August fit-game CV; potentially training-seen, no golden CM.',fit_cv_selected=selected,results=records,analysis_seconds=time.monotonic()-start,
        sample_sha256=digest(ROOT/'aug-tune-v1/sample.json'),source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('retrieval_common.py'),Path(__file__).with_name('analyze_retrieval.py')]})
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,rec in records.items():print(name,rec['confirmation']['macro_ce'],rec['confirmation']['expert_ce'],'strength',rec['parameters']['strength'],flush=True)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
