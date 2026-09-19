"""Action-class calibration after frozen model-only search."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .native_board import MOVES
from .action_features_native import load,NAMES
from .value_gap import objective,test
from .analyze_august import means

def main():
    test();start=time.monotonic();out=ROOT/'aug-action-calibration-v1';out.mkdir(exist_ok=True)
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','mask','target'))
    n=len(rows);ar=np.arange(n);surface=ROOT/'aug-budget-surface-v2'
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games);parents=f['policy'][-1].astype(float);nodes=f['nodes'][-1]
    menus=[('parent','basic',0.,False)]
    for kind in ('basic','full'):
        for separate in (False,True):
            for ridge in (.01,.1):menus.append((kind+('_elo' if separate else '_shared')+str(ridge),kind,ridge,separate))
    plan=dict(sources={p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(s) for s in ('action_features_native.py','action_features.cpp','value_gap.py')]]},
        sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),parent_sha256=digest(surface/'policies.npz'),menus=menus,feature_names=NAMES,minimum_scale=.1,
        formula='pi_new proportional to pi_parent*exp(theta*rule_features). Basic features: moving piece type, capture, check, castle, promotion, forced reply. Full features split victim type and add relative destination rank, file centrality, distance, opponent legal-move count. Per-query centering cancels in normalization; feature scales fitted on training rows only.',
        validation='Ridge coefficients shared or per Elo band, fitted inside each game fold with independently fitted parent. All variants on disjoint August confirmation; CV chooses before confirmation. No neural-weight updates or external evaluation engine. Rules reconstruct prefix only; no target/future used to compute features.',
        cost='No neural calls beyond parent. CPU candidate feature generation and calibration time reported separately.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    begin=time.monotonic();full=load().encode([r['prefix'] for r in rows],[r['legal'] for r in rows],MOVES)
    generation_seconds=time.monotonic()-begin;assert full.shape==(*mask.shape,20)
    basic=np.concatenate([full[:,:,:6],full[:,:,6:12].sum(2,keepdims=True),full[:,:,12:15],full[:,:,19:20]],axis=2)
    np.savez_compressed(out/'features.npz',features=full,game=games,ply=[r['ply'] for r in rows])
    records,losses={},{}
    for name,kind,ridge,separate in menus:
        oof=np.full(n,np.nan);params=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            p=parents[vi];p/=p.sum(1,keepdims=True);base=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
            coefficients={}
            if name=='parent':new=p
            else:
                feature=basic if kind=='basic' else full
                x=feature-np.einsum('nk,nkd->nd',p,feature)[:,None,:];x=np.where(mask[:,:,None],x,0.)
                new=np.zeros_like(p);groups=cells%4 if separate else np.zeros(n,int)
                for g in np.unique(groups):
                    tr=train&(groups==g);take=groups==g;ct=np.bincount(cells[tr],minlength=16);w=1/ct[cells[tr]];w/=w.sum()
                    scale=np.maximum(np.sqrt(w@np.einsum('nk,nkd->nd',p[tr],x[tr]**2)),.1)
                    xx=x/scale
                    opt=minimize(lambda t:objective(t,xx[tr],base[tr],mask[tr],target[tr],w,ridge),np.zeros(x.shape[2]),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*x.shape[2],options=dict(ftol=1e-11,gtol=1e-7,maxiter=250))
                    assert opt.success,opt
                    new[take]=objective(opt.x,xx[take],base[take],mask[take],target[take])
                    coefficients[str(g)]=dict(theta=opt.x.tolist(),scale=scale.tolist())
            assert np.isfinite(new).all() and (new[mask]>0).all()
            np.testing.assert_allclose(new.sum(1),1.,atol=1e-13)
            loss=-np.log(new[ar,target]);params.append(dict(fold=fold,groups=coefficients))
            if fold is None:losses[name]=loss
            else:oof[fm&(cv==fold)]=loss[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,kind=kind,ridge=ridge,separate_elo=separate,training_equivalent_cm=None,mean_nodes=float(nodes.mean()),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
        print(name,records[name]['fit_game_cv'],ce.mean(),ce[3::4].mean(),flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(18771).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['parent'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_parent']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,feature_generation_seconds=generation_seconds,plan_sha256=digest(pp)))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
