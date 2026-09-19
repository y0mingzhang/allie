"""Fit fixed-model within-game residual correction on game-disjoint August folds."""
import time,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from .service import ROOT,atomic
from .balanced_eval import digest
from .analyze_august import means
from .retrieval_common import data

def policy(p,c,mask,gamma):
    z=np.log(np.maximum(p,1e-300))+gamma*c
    z=np.where(mask,z,-np.inf)
    return softmax(z,axis=1)

def strength(p,c,mask,target,cells,train):
    count=np.bincount(cells[train],minlength=16)
    weight=1/count[cells[train]];weight/=weight.sum()
    ar=np.flatnonzero(train)
    objective=lambda g:float(weight@(-np.log(policy(p[train],c[train],mask[train],g)[np.arange(len(ar)),target[train]])))
    fit=minimize_scalar(objective,bounds=(0.,100.),method='bounded',options={'xatol':1e-8})
    assert fit.success
    return float(min([0.,100.,float(fit.x)],key=objective))

def test():
    rng=np.random.default_rng(277);p=rng.dirichlet(np.ones(5),size=20)
    mask=np.ones_like(p,bool);c=rng.normal(size=p.shape)
    np.testing.assert_allclose(policy(p,c,mask,0),p,atol=1e-15)
    np.testing.assert_allclose(policy(p,np.zeros_like(c),mask,55),p,atol=1e-15)
    np.testing.assert_allclose(policy(p,c,mask,100).sum(1),1,atol=1e-15)
    mask[:,-1]=False;q=policy(p,c,mask,10)
    assert (q[:,-1]==0).all() and (q[:,:-1]>0).all()
    print('PASS residual normalization, zero strength, empty residual, legal support')

def main():
    test();start=time.monotonic();out=ROOT/'aug-history-residual-v1'
    plan=json.loads((out/'plan.json').read_text());worker=json.loads((out/'worker.json').read_text())
    assert worker['plan_sha256']==digest(out/'plan.json')
    assert worker['features_sha256']==digest(out/'features.npz')
    d=data();cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];target=d['target'];mask=d['mask'];n=len(target);ar=np.arange(n)
    with np.load(out/'features.npz') as z:
        corrections=z['correction'].astype(float)
        np.testing.assert_array_equal(z['ids'],d['ids']);np.testing.assert_array_equal(z['mask'],mask)
        np.testing.assert_array_equal(z['game'],games);np.testing.assert_array_equal(z['ply'],[r['ply'] for r in d['rows']])
    records={};losses={}
    def record(name,p,oof,params,base,variant):
        loss=-np.log(p[ar,target]);conf=means(loss[~fm],cells[~fm]);cvce=means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,base=base,variant=variant,training_equivalent_cm=None,
            mean_nodes=float(d['cost'].mean()) if base=='search' else 0.,
            fit_game_cv=dict(macro_ce=float(cvce.mean()),expert_ce=float(cvce[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        losses[name]=loss
    for base,versions in d['controls'].items():
        p,params=versions[0];oof=np.full(n,np.nan)
        for f in range(3):
            val=fm&(cv==f);pp=versions[f+1][0];oof[val]=-np.log(pp[ar[val],target[val]])
        record(base,p,oof,dict(base=params,strength=0),base,None)
        for j,variant in enumerate(plan['variants']):
            c=corrections[j];g=strength(p,c,mask,target,cells,fm);mixed=policy(p,c,mask,g);oof=np.full(n,np.nan)
            for f in range(3):
                pp=versions[f+1][0];val=fm&(cv==f)
                gf=strength(pp,c,mask,target,cells,fm&(cv!=f));pm=policy(pp,c,mask,gf)
                oof[val]=-np.log(pm[ar[val],target[val]])
            record(base+'_'+variant,mixed,oof,dict(base=params,strength=g),base,variant)
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float)
    den=w@count;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses[rec['base']];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['confirmation_delta_vs_base']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August fit-game CV; potentially training-seen. Golden CM pending.',fit_cv_selected=selected,
        results=records,analysis_seconds=time.monotonic()-start,worker=worker,
        sample_sha256=digest(ROOT/'aug-tune-v1/sample.json'),
        source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('retrieval_common.py')]})
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,rec in records.items():print(name,rec['confirmation'],rec['parameters']['strength'],flush=True)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
