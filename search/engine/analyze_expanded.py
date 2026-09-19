"""Larger August development comparison, including convex search-strength mixtures."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .fit_policy import fit,loss_gradient
from .analyze_august import means
from .analyze_player_search import FACTORS

def components(z,q,mask,params,group):
    a=np.array([params[str(g)]['alpha'] for g in group]);b=np.array([params[str(g)]['beta'] for g in group])
    logits=a[None,:,None]*z[None,:,:]+(FACTORS[:,None]*b[None,:])[:,:,None]*q[None,:,:]
    return softmax(np.where(mask[None,:,:],logits,-np.inf),axis=2)

def learn(p,target,cells,train):
    n=len(target);pt=p[:,np.arange(n),target][:,train]
    counts=np.bincount(cells[train],minlength=16);w=1/counts[cells[train]];w/=w.sum()
    def objective(x):
        prob=np.maximum(x@pt,1e-300)
        return float(w@(-np.log(prob))),-(pt*(w/prob)[None,:]).sum(1)
    r=minimize(objective,np.ones(5)/5,jac=True,method='SLSQP',bounds=[(0,1)]*5,
        constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(5))],options=dict(maxiter=100,ftol=1e-12))
    assert r.success,r
    x=np.clip(r.x,0,1);x/=x.sum()
    assert objective(x)[0]<=objective(np.array([0,0,1,0,0]))[0]+1e-7
    return x

def test():
    rng=np.random.default_rng(961);p=rng.dirichlet(np.ones(3),size=(5,160))
    # The labels prefer component0 by construction; fit is a proper full-support mixture.
    target=p[0].argmax(1);cells=np.arange(160)%16
    w=learn(p,target,cells,np.ones(160,bool));q=np.einsum('a,ank->nk',w,p)
    assert np.isfinite(q).all() and (q>0).all()
    np.testing.assert_allclose(q.sum(1),1,atol=1e-14)
    assert w[0]>.5
    print('PASS convex mixture normalization, full support, likelihood direction and control inclusion')

def main():
    test();start=time.monotonic();out=ROOT/'aug-expanded-search-v1';sample='aug-tune-expanded-v1'
    original=json.loads((out/'plan.json').read_text());worker=json.loads((out/'worker.json').read_text())
    plan=dict(parent_plan_sha256=digest(out/'plan.json'),sample_sha256=digest(ROOT/sample/'sample.json'),
        widths=[0,.5,1.],learned='One shared5-component simplex, convex CE fit on fitgames only and refit inside3-waygameCV. Factors [.25,.5,1,2,4]. No neural parameters.',
        methods=original['methods'],budgets=original['budgets'],
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('sample_data.py'),Path(__file__).with_name('fit_policy.py')]},
        selection='All candidate arm definitions fixed before expanded losses. Same Elo-calibrated control is refit within every fold. Width0=single original search strength. Identical nodes for all mixtures on a fixed budget/backup.')
    pp=out/'analysis-plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    d=read(sample);rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];ids=d['ids'];mask=d['mask'];target=d['target'];n=len(rows);ar=np.arange(n);group=cells%4
    root=np.zeros((n,2432),np.float32);q=np.zeros((len(original['budgets']),len(original['methods']),n,mask.shape[1]));cost=np.zeros((len(original['budgets']),n))
    for lo in range(0,n,original['roots_per_batch']):
        with np.load(out/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1]
            np.testing.assert_array_equal(z['game'],games[lo:hi]);np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk]);np.testing.assert_array_equal(z['mask'],mask[lo:hi,:kk])
            root[lo:hi]=z['z'];q[:,:,lo:hi,:kk]=z['q'];cost[:,lo:hi]=z['evaluated_nodes']
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids].astype(float),0);del root
    records={};losses={}
    for b,budget in enumerate(original['budgets']):
        for j,method in enumerate(original['methods']):
            versions=[]
            for fold,train in [(None,fm)]+[(f,fm&(cv!=f)) for f in range(3)]:
                params={}
                for g in range(4):
                    tr=train&(group==g);counts=np.bincount(cells[tr],minlength=16)
                    fp=fit(logits[tr],q[b,j,tr],mask[tr],target[tr],'forward',1/counts[cells[tr]]);assert fp['converged'];params[str(g)]=fp
                p=components(logits,q[b,j],mask,params,group)
                weights={'single':np.array([0,0,1,0,0.]),'sigma05':softmax(-.5*(np.log(FACTORS)/.5)**2),'sigma10':softmax(-.5*np.log(FACTORS)**2),'learned':learn(p,target,cells,train)}
                versions.append((params,{name:(np.einsum('a,ank->nk',w,p),w) for name,w in weights.items()}))
            for mixture in versions[0][1]:
                p,w=versions[0][1][mixture];loss=-np.log(p[ar,target]);oof=np.full(n,np.nan)
                for f in range(3):
                    val=fm&(cv==f);pp_,_=versions[f+1][1][mixture];oof[val]=-np.log(pp_[ar[val],target[val]])
                ce=means(loss[~fm],cells[~fm]);cc=means(oof[fm],cells[fm]);name=f'{method}{budget}_{mixture}'
                records[name]=dict(method=method,budget=budget,mixture=mixture,parameters=versions[0][0],weights=w.tolist(),training_equivalent_cm=None,
                    mean_nodes=float(means(cost[b],cells).mean()),expert_mean_nodes=float(means(cost[b],cells)[3::4].mean()),
                    fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),
                    confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()))
                losses[name]=loss
            print('Expanded calibrated',method,budget,flush=True)
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,r in records.items():
        for ref in [f"constant{r['budget']}_single",f"{r['method']}{r['budget']}_single"]:
            delta=losses[name]-losses[ref];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
            r['delta_vs_'+ref]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='16384 August moves, same inventory and game folds; potentially training-seen. Golden CM pending.',results=records,fit_cv_selected=selected,
        analysis_seconds=time.monotonic()-start,analysis_plan_sha256=digest(out/'analysis-plan.json'),worker=worker)
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,r in records.items():print(name,r['confirmation']['macro_ce'],r['confirmation']['expert_ce'],r['mean_nodes'],r['weights'],flush=True)
    print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
