"""Cross-fit layer readout contrasts against identical frozen search outputs."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .value_gap import objective,test
from .analyze_august import means

def main():
    test();start=time.monotonic();out=ROOT/'aug-layer-lens-v1'
    worker=json.loads((out/'worker.json').read_text());assert worker['plan_sha256']==digest(out/'plan.json')
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','mask','target'))
    n,k=mask.shape;ar=np.arange(n);z=np.zeros((4,n,k));surface=ROOT/'aug-budget-surface-v2'
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game'])
            np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
            z[:,lo:hi,:f['logits'].shape[2]]=f['logits']
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games);parents=f['policy'][-1].astype(float);nodes=f['nodes'][-1]
    prob=softmax(np.where(mask[None],z,-np.inf),axis=2)
    mix=(prob[:3]+prob[-1][None])/2
    log=lambda p:np.log(np.maximum(p,1e-300))
    js=.5*(prob[:3]*(log(prob[:3])-log(mix))+prob[-1][None]*(log(prob[-1][None])-log(mix))).sum(2)
    chosen=js.argmax(0);dynamic=z[chosen,ar]
    feature=dict(mature=z[-1,:,:,None],layer2=np.stack([z[-1],z[0]],axis=2),layer4=np.stack([z[-1],z[1]],axis=2),
        layer6=np.stack([z[-1],z[2]],axis=2),dynamic_js=np.stack([z[-1],dynamic],axis=2),all_layers=z.transpose(1,2,0))
    menus=[('parent','mature',0.)]+[(kind+str(ridge),kind,ridge) for kind in feature for ridge in (.001,.01)]
    plan=dict(sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('value_gap.py')]},
        worker_sha256=digest(out/'worker.json'),parent_sha256=digest(surface/'policies.npz'),menus=menus,
        formula='pi_new proportional to parent*exp(sum theta_j*logit_lens_j). Mature-only is the calibration control. Two-feature arms combine mature with an early readout; negative early coefficient means layer contrast. Dynamic_JS chooses early layer by maximal legal-policy JS divergence from mature, without targets. All-layer arm has4 coefficients. No zero-probability plausibility truncation.',
        validation='Ridge coefficients and scales fitted inside each of3 game CV folds using independent parent fits, then all arms on August confirmation. No model-weight updates; no golden tuning. Parent identity and eager-vs-serving numerical differences explicitly controlled by keeping the same parent, rather than replacing it with eager root output.',
        cost='One extra eager full-prefix call and4 readouts for every non-parent arm including mature-only control; equal cost for mechanism comparison.')
    plan=json.loads(json.dumps(plan));pp=out/'analysis-plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    records,losses={},{}
    for name,kind,ridge in menus:
        oof=np.full(n,np.nan);params=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            parent=parents[vi];parent/=parent.sum(1,keepdims=True);base=np.where(mask,log(parent),0.)
            coefficients={}
            if name=='parent':p=parent
            else:
                xx=feature[kind]
                x=xx-np.einsum('nk,nkd->nd',parent,xx)[:,None,:];x=np.where(mask[:,:,None],x,0.)
                ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
                scale=np.maximum(np.sqrt(w@np.einsum('nk,nkd->nd',parent[train],x[train]**2)),.05)
                x/=scale
                opt=minimize(lambda t:objective(t,x[train],base[train],mask[train],target[train],w,ridge),np.zeros(x.shape[2]),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*x.shape[2],options=dict(ftol=1e-11,gtol=1e-7,maxiter=250))
                assert opt.success,opt
                p=objective(opt.x,x,base,mask,target);coefficients=dict(theta=opt.x.tolist(),scale=scale.tolist())
            assert np.isfinite(p).all() and (p[mask]>0).all()
            np.testing.assert_allclose(p.sum(1),1.,atol=1e-13)
            loss=-np.log(p[ar,target]);params.append(dict(fold=fold,**coefficients))
            if fold is None:losses[name]=loss
            else:oof[fm&(cv==fold)]=loss[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,kind=kind,ridge=ridge,training_equivalent_cm=None,mean_nodes=float(nodes.mean())+int(name!='parent'),
            extra_full_prefix_queries=int(name!='parent'),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
        print(name,records[name]['fit_game_cv'],ce.mean(),ce[3::4].mean(),params[0],flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(72185).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        for ref in ('parent','mature0.01'):
            delta=losses[name]-losses[ref];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
            rec['delta_vs_'+ref]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,analysis_plan_sha256=digest(pp),worker=worker,dynamic_layer_fractions=np.bincount(chosen,minlength=3).tolist()))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
