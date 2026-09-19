"""Marginalize limited-search beliefs rather than only varying root sharpness."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_august import means
from .value_of_compute import features,design


def objective(theta,x,small,big,weights,ridge):
    w=expit(x@theta);prob=big+w*(small-big)
    loss=-weights@np.log(prob)+.5*ridge*np.square(theta[1:]).sum()
    grad=x.T@(weights*(big-small)/prob*w*(1-w));grad[1:]+=ridge*theta[1:]
    return float(loss),grad


def test():
    rng=np.random.default_rng(2792);x=np.c_[np.ones(51),rng.normal(size=(51,4))]
    a,b=rng.uniform(.001,.5,size=(2,51));w=np.ones(51)/51;t=rng.normal(size=5)*.1
    _,g=objective(t,x,a,b,w,.01);fd=[]
    for j in range(5):
        aa,bb=t.copy(),t.copy();aa[j]+=1e-6;bb[j]-=1e-6
        fd.append((objective(aa,x,a,b,w,.01)[0]-objective(bb,x,a,b,w,.01)[0])/2e-6)
    np.testing.assert_allclose(g,fd,atol=1e-9)
    print('PASS depth-mixture gradient',flush=True)


def main():
    test();start=time.monotonic();out=ROOT/'aug-depth-mixture-v1';out.mkdir(exist_ok=True)
    surface=ROOT/'aug-budget-surface-v2';d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n=len(rows);ar=np.arange(n)
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games)
        policies=f['policy'].astype(float);q=f['q'];root=f['root'];nodes=f['nodes'][-1];budgets=f['budgets']
    policies/=policies.sum(-1,keepdims=True)
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text())
    assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:feats=f['feats']
    seconds=np.array([feats[r['row'],r['column']-1,0] for r in rows])
    menus=[('parent',None,None,0.)]
    for bi,b in enumerate(budgets[:-1]):
        for kind in ('static','elo','state'):menus.append((str(b)+'_'+kind,bi,kind,.01))
    plan=dict(source_sha256=digest(Path(__file__)),sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        policies_sha256=digest(surface/'policies.npz'),menus=menus,
        formula='pi=(1-sigmoid(x theta))*pi1000+sigmoid(x theta)*pi_b, b128/256/512; independently calibrated parent policies per game fold. Ratings/format, pre-move clock, predicted time,128-only state signals and cheap-vs-deep KL can condition the latent deliberation depth. All methods pay the full1000 search.',
        fit='Every gate re-fit inside3 game folds, best of2 fixed starts by training objective. All methods on August disjoint confirmation. Zero-weight hard parent retained exactly.',
        hypothesis='Changing root strength beta cannot recover an action whose subjective ranking changed with search depth. A mixture of partial-search beliefs may model limited deliberation. Older small-data/old-tree budget mixtures failed; this retry uses improved root-coverage trees, larger August game sample and legitimate clock signals.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    records,losses={},{}
    for name,bi,kind,ridge in menus:
        oof=np.full(n,np.nan);parameters=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            big=policies[-1,vi];params={}
            if kind is None:p=big
            else:
                small=policies[bi,vi]
                feat=features(rows,root,q[0],policies[0,vi],mask,ids,seconds)
                disagreement=(small*(np.log(np.maximum(small,1e-300))-np.log(np.maximum(big,1e-300)))).sum(1)
                feat=np.c_[feat,np.log1p(disagreement)]
                if kind=='static':x=np.ones((n,1));mu=[];scale=[]
                else:x,mu,scale=design(cells,feat,train,kind)
                ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
                pt,bt=small[ar,target],big[ar,target];opts=[]
                for intercept in (-2.,0.):
                    init=np.zeros(x.shape[1]);init[0]=intercept
                    opt=minimize(lambda t:objective(t,x[train],pt[train],bt[train],w,ridge),init,jac=True,
                        method='L-BFGS-B',bounds=[(-12,12)]+[(-3,3)]*(len(init)-1),options=dict(ftol=1e-11,gtol=1e-7,maxiter=250))
                    assert opt.success,opt;opts.append(opt)
                opt=min(opts,key=lambda o:o.fun);weight=expit(x@opt.x)
                p=weight[:,None]*small+(1-weight[:,None])*big
                params=dict(kind=kind,theta=opt.x.tolist(),mean=np.asarray(mu).tolist(),scale=np.asarray(scale).tolist(),mean_weight=float(weight.mean()))
            np.testing.assert_allclose(p.sum(1),1,atol=1e-13)
            ll=-np.log(p[ar,target]);parameters.append(dict(fold=fold,parameters=params))
            if fold is None:losses[name]=ll
            else:oof[fm&(cv==fold)]=ll[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(parameters=parameters,training_equivalent_cm=None,mean_nodes=float(nodes.mean()),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
        print(name,ce.mean(),ce[3::4].mean(),records[name]['fit_game_cv'],flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(73634).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count
    for name,rec in records.items():
        delta=losses[name]-losses['parent'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['delta_vs_parent']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp)))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
