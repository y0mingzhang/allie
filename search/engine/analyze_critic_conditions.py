"""Fit small value-only corrections; all tree/parent policy scores remain fixed."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .critic_conditions import VARIANTS
from .value_gap import objective
from .analyze_august import means


def main():
    start=time.monotonic();out=ROOT/'aug-critic-conditions-v1'
    assert (out/'worker.json').exists(),'wait for complete GPU probe'
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','mask','target'))
    n=len(rows);ar=np.arange(n);surface=ROOT/'aug-budget-surface-v2'
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games)
        parent=f['policy'][-1].astype(float);basecost=f['nodes'][-1]
    values={};extra={}
    for name in VARIANTS:
        value=np.zeros(mask.shape);nodes=np.zeros(n)
        for path in sorted((out/name).glob('[0-9]*.npz')):
            with np.load(path) as f:
                lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
                k=f['wdl'].shape[1];wdl=f['wdl']
                np.testing.assert_allclose(wdl.sum(2)[mask[lo:hi,:k]],1.,atol=1e-12)
                value[lo:hi,:k]=wdl[:,:,0]-wdl[:,:,2];nodes[lo:hi]=f['nodes']
        values[name]=value;extra[name]=nodes
    menus=[('parent',[])]
    for name in VARIANTS:menus.append((name,[name]))
    for name in VARIANTS[1:]:menus.append((name+'_and_actual',['actual',name]))
    plan=dict(gpu_plan_sha256=digest(out/'plan.json'),surface_sha256=digest(surface/'results.json'),
        source_sha256=digest(Path(__file__)),loss_source_sha256=digest(Path(__file__).with_name('value_gap.py')),
        menus=menus,ridge=.01,
        fit='Residual log policy correction from log parent and one/two candidate-child value features. Elo-group scalars, ridge .01, feature scales estimated only on training games. Parent and every correction refitted inside3 game folds.',
        semantics='The actual human-policy prior and prior search remain fixed; changed ratings used only in auxiliary VALUE queries. All extra nonterminal child calls and full-root queries charged conservatively; actual-rating children could potentially be reused from parent tree but no such implementation claim here. No external engine or future labels.')
    plan=json.loads(json.dumps(plan));pp=out/'analysis-plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    records,losses={},{}
    for name,fields in menus:
        oof=np.full(n,np.nan);versions=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            p=parent[vi];p=p/p.sum(1,keepdims=True)
            logbase=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
            params={}
            if not fields:new=p
            else:
                x=np.stack([logbase,*[values[f] for f in fields]],axis=2)
                x-=np.einsum('nk,nkd->nd',p,x)[:,None,:];x=np.where(mask[:,:,None],x,0.)
                new=np.zeros_like(p)
                for g in range(4):
                    tr=train&(cells%4==g);take=cells%4==g
                    ct=np.bincount(cells[tr],minlength=16);w=1/ct[cells[tr]];w/=w.sum()
                    scale=np.maximum(np.sqrt(w@np.einsum('nk,nkd->nd',p[tr],x[tr]**2)),1e-6)
                    xx=x/scale
                    opt=minimize(lambda t:objective(t,xx[tr],logbase[tr],mask[tr],target[tr],w,.01),np.zeros(x.shape[2]),jac=True,
                        method='L-BFGS-B',bounds=[(-3,3)]*x.shape[2],options=dict(ftol=1e-11,gtol=1e-7,maxiter=200))
                    assert opt.success,opt
                    new[take]=objective(opt.x,xx[take],logbase[take],mask[take],target[take])
                    params[str(g)]=dict(theta=opt.x.tolist(),scale=scale.tolist())
            assert np.isfinite(new).all() and (new[mask]>0).all()
            np.testing.assert_allclose(new.sum(1),1.,atol=1e-13)
            ll=-np.log(new[ar,target]);versions.append(dict(fold=fold,groups=params))
            if fold is None:losses[name]=ll
            else:oof[fm&(cv==fold)]=ll[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        cost=basecost.copy()
        for field in fields:cost+=extra[field]
        records[name]=dict(fields=fields,parameters=versions,training_equivalent_cm=None,
            mean_nodes=float(cost.mean()),extra_full_prefix_queries=len(fields),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
        print(name,ce.mean(),ce[3::4].mean(),records[name]['fit_game_cv'],flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(738146).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count
    for name,rec in records.items():
        delta=losses[name]-losses['parent'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['delta_vs_parent']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp)))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
