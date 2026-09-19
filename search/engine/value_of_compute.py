"""Learn where another model evaluation buys human-move likelihood.

The router observes only root inputs and the first 128 simulations. Targets are
paired losses of longer searches on fit games. It never sees the played move,
long-search features, or future clocks when choosing a query's budget.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_august import means


def choose(x, coef, penalty, budgets):
    assert np.isfinite(x).all() and np.isfinite(coef).all()
    return np.argmin(x@coef + penalty*np.asarray(budgets)[None, :]/1000., axis=1)


def solve(x, losses, weights, ridge, budgets, cap):
    assert np.isfinite(x).all() and np.isfinite(losses).all()
    # Regression on paired differences removes most shared position difficulty.
    y=losses-losses[:, :1]
    reg=np.eye(x.shape[1])*ridge
    reg[0,0]=1e-10
    coef=np.linalg.solve(x.T@(weights[:,None]*x)+reg,x.T@(weights[:,None]*y))
    low,high=0.,1.
    cost=lambda p:float(weights@np.asarray(budgets)[choose(x,coef,p,budgets)])
    while cost(high)>cap:high*=2
    if cost(0)<=cap:high=0.
    else:
        for _ in range(60):
            mid=(low+high)/2
            if cost(mid)>cap:low=mid
            else:high=mid
    assert np.isfinite(coef).all() and cost(high)<=cap+1e-8
    return coef,high


def features(rows, root, q128, p128, mask, ids, seconds):
    p128=np.asarray(p128,dtype=np.float64)
    ar=np.arange(len(rows))
    z=np.where(mask,root[:,378:2346][ar[:,None],ids],-np.inf)
    prior=softmax(z,axis=1)
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    mean=(prior*q128).sum(1)
    spread=np.sqrt((prior*(q128-mean[:,None])**2).sum(1))
    kl=(p128*(np.log(np.maximum(p128,1e-300))-np.log(np.maximum(prior,1e-300)))).sum(1)
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    think=np.log1p(softmax(root[:,2350:2413],axis=1)@tc)
    known=seconds>=0
    return np.column_stack([entropy,np.log(.01+spread),np.log1p(kl),
        [len(r['prefix'])-11 for r in rows],think,
        np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known])


def design(cells, feat, train, kind):
    # Elo and format are available at inference; never use the target move.
    category=np.eye(4)[cells%4][:,1:]
    if kind=='elo':raw=category
    else:raw=np.c_[category,np.eye(4)[cells//4][:,1:],feat]
    mean=raw[train].mean(0);scale=np.maximum(raw[train].std(0),1e-6)
    return np.c_[np.ones(len(raw)),np.clip((raw-mean)/scale,-3,3)],mean,scale


def main():
    start=time.monotonic()
    out=ROOT/'aug-value-of-compute-v2';out.mkdir(exist_ok=True)
    surface=ROOT/'aug-budget-surface-v2'
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n=len(rows);ar=np.arange(n)
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games)
        p,q,root,nodes,budgets=(f[k] for k in ('policy','q','root','nodes','budgets'))
    seconds=[]
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text())
    for fn in ('strat.npz','feats.npz'):assert digest(inv/fn)==manifest['files_sha256'][fn]
    with np.load(inv/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(inv/'feats.npz') as f:feat=f['feats']
    for r in rows:
        rr,cc=r['row'],r['column']
        assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(r['prefix']):cc],r['prefix'])
        seconds.append(feat[rr,cc-1,0])
    seconds=np.array(seconds)
    menus=[('fixed'+str(b),'fixed',int(b)) for b in budgets]
    menus += [(kind+str(ridge),kind,ridge) for kind in ('elo','state') for ridge in (.01,.1,1.)]
    plan=dict(source_sha256=digest(Path(__file__)),surface_sha256=digest(surface/'results.json'),
        policies_sha256=digest(surface/'policies.npz'),sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        budgets=budgets.tolist(),menus=menus,nominal_cap=512,
        features='Elo group, format, root prior entropy,128-sim Q spread,128-sim policy KL from prior,ply,predicted thinking time,actual pre-move remaining clock and missingness. No longer-search features or target move used by router.',
        objective='Weighted ridge regression of paired CE differences relative to128; choose argmin predicted CE+lambda*nominal simulations. Lambda fitted on training queries only to average at most512 simulations. Report realized NN requests separately.',
        validation='Every component calibration and router re-fitted inside3 game folds; all methods scored on disjoint August confirmation. Earlier same-fit policy calibration supplies router training targets, a limitation addressed by held-out outer game CV. No golden tuning.',
        semantics='Cached-prefix experiment only; eventual promotion requires actual live mixed-budget run and batch drift audit.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    records,losses,choices,params={},{},{},{}
    for name,kind,setting in menus:
        params[name]=[];oof=np.full(n,np.nan);oofcost=np.full(n,np.nan)
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            ll=-np.log(np.maximum(p[:,vi,ar,target],1e-300)).T
            if kind=='fixed':
                selected=np.full(n,int(np.flatnonzero(budgets==setting)[0]))
                pars=dict(fold=fold,budget=setting)
            else:
                feat=features(rows,root,q[0],p[0,vi],mask,ids,seconds)
                x,mean,scale=design(cells,feat,train,kind)
                ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
                coef,penalty=solve(x[train],ll[train],w,setting,budgets,512)
                selected=choose(x,coef,penalty,budgets)
                pars=dict(fold=fold,kind=kind,ridge=setting,mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),penalty=penalty)
            params[name].append(pars)
            loss=ll[ar,selected];cost=nodes[selected,ar]
            if fold is None:losses[name]=loss;choices[name]=selected
            else:
                take=fm&(cv==fold);oof[take]=loss[take];oofcost[take]=cost[take]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        sel=choices[name]
        records[name]=dict(parameters=params[name],training_equivalent_cm=None,
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean()),mean_nodes=float(oofcost[fm].mean())),
            mean_nodes=float(nodes[sel,ar][~fm].mean()),expert_mean_nodes=float(nodes[sel,ar][~fm&(cells%4==3)].mean()),
            nominal_mean=float(budgets[sel][~fm].mean()),budget_fractions={str(b):float((budgets[sel][~fm]==b).mean()) for b in budgets})
        print(name,records[name]['confirmation']['macro_ce'],records[name]['confirmation']['expert_ce'],records[name]['mean_nodes'],records[name]['fit_game_cv'],flush=True)
    eligible=[k for k in records if k!='fixed1000']
    selected={m:min(eligible,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    weights=np.random.default_rng(849123).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=weights@count
    for name,rec in records.items():
        delta=losses[name]-losses['fixed512'];point=means(delta[~fm],cells[~fm])
        sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=weights@sums/den
        rec['delta_vs_fixed512']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp)))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),choice=np.stack(list(choices.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
