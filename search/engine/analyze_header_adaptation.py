"""Bayesian header-gap adaptation on strictly past moves, with static controls."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize_scalar
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_august import means

OFFSETS=np.array([-400.,-200.,0.,200.,400.])


def posterior(evidence,width,power,history):
    ll=evidence[:,:,:history].sum(2).T
    prior=-.5*(OFFSETS/width)**2
    return softmax(prior[None,:]+power*ll,axis=1)


def tilted(kappa,base,ratio,mask,target,weights=None):
    z=np.where(mask,base+kappa*ratio,-np.inf);lp=z-logsumexp(z,axis=1,keepdims=True)
    if weights is None:return np.exp(lp)
    return float(-weights@lp[np.arange(len(target)),target])


def test():
    rng=np.random.default_rng(786)
    e=rng.normal(size=(5,20,32));e[:,:,8:]=0.
    prior=posterior(e,200,0,8)
    np.testing.assert_allclose(prior,posterior(np.zeros_like(e),200,1,32),atol=1e-15)
    ep=e.copy();ep[4]+=1
    assert (posterior(ep,200,1,8)[:,4]>posterior(e,200,1,8)[:,4]).all()
    p=rng.dirichlet(np.ones(7),size=20);mask=np.ones_like(p,bool)
    np.testing.assert_allclose(tilted(0,np.log(p),rng.normal(size=p.shape),mask,np.zeros(20,int)),p,atol=1e-15)
    print('PASS no-evidence/static prior, posterior direction, zero tilt identity',flush=True)


def main():
    test();start=time.monotonic();out=ROOT/'aug-header-adaptation-v1'
    worker=json.loads((out/'worker.json').read_text());assert worker['plan_sha256']==digest(out/'plan.json')
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','mask','target'))
    n,k=mask.shape;ar=np.arange(n)
    pi=np.zeros((5,n,k));evidence=np.zeros((5,n,32));counts=np.zeros(n,int)
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game'])
            np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
            pi[:,lo:hi,:f['probability'].shape[2]]=f['probability'];evidence[:,lo:hi]=f['evidence'];counts[lo:hi]=f['counts']
    assert (pi[:,mask]>0).all();np.testing.assert_allclose(pi.sum(2),1.,atol=1e-13)
    surface=ROOT/'aug-budget-surface-v2'
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games);parents=f['policy'][-1].astype(float);nodes=f['nodes'][-1]
    menus=[('parent',None,0,0)]
    for width in (200,400):
        menus.append((f'static{width}',width,0,8))
        for power in (.1,.3,1.):
            for history in (8,32):menus.append((f'adapt{width}_p{power}_h{history}',width,power,history))
    plan=dict(source_sha256=digest(Path(__file__)),worker_sha256=digest(out/'worker.json'),
        source_plan_sha256=digest(out/'plan.json'),surface_sha256=digest(surface/'policies.npz'),menus=menus,
        formula='Posterior(delta) proportional to exp(-0.5*(delta/width)^2 + power*sum(last H past own move loglik(delta))). pi_mix=sum posterior*pi_delta. pi_final proportional to parent * (pi_mix/pi_delta0)^kappa, kappa in[0,2] fit inside each training-game fold. Static controls power0 with identical five prefills.',
        validation='Refit kappa within each of3 fit-game folds, evaluate all arms on disjoint August confirmation. Parent calibrations were independently fitted on same folds. Prior width/power/history selected only by fit CV, not confirmation. No current target or future at inference. No neural-weight changes.',
        attribution='The LM already sees past moves; any adaptation win is additional use of evidence, not newly available information. Offset absorbs own or opponent rating gap errors. Report posterior entropy, recent-history size and gains by number of own moves seen.',
        cost='Five extra full-prefix queries per position, charged equally to static and adaptive variants. Search nodes unchanged, prefix token cost explicit.')
    plan=json.loads(json.dumps(plan));pp=out/'analysis-plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    records,losses={},{};own_moves=np.array([(len(r['prefix'])-11)//2 for r in rows])
    for name,width,power,history in menus:
        weights=posterior(evidence,width,power,history) if width else np.eye(5)[np.full(n,2)]
        mix=np.einsum('nv,vnk->nk',weights,pi)
        ratio=np.where(mask,np.log(np.maximum(mix,1e-300))-np.log(np.maximum(pi[2],1e-300)),0.)
        oof=np.full(n,np.nan);params=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            p=parents[vi];p/=p.sum(1,keepdims=True);base=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
            ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
            kappa=0.
            if width:
                opt=minimize_scalar(lambda a:tilted(a,base[train],ratio[train],mask[train],target[train],w),bounds=(0,2),method='bounded',options=dict(xatol=1e-7))
                assert opt.success,opt
                candidates=[0.,float(opt.x),2.]
                kappa=min(candidates,key=lambda a:tilted(a,base[train],ratio[train],mask[train],target[train],w))
            p=tilted(kappa,base,ratio,mask,target);ll=-np.log(p[ar,target]);params.append(dict(fold=fold,kappa=kappa))
            if fold is None:losses[name]=ll
            else:oof[fm&(cv==fold)]=ll[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        entropy=-(weights*np.log(np.maximum(weights,1e-300))).sum(1)
        records[name]=dict(parameters=params,width=width,power=power,history=history,training_equivalent_cm=None,
            mean_nodes=float(nodes.mean())+5*bool(width),extra_full_prefix_queries=5*int(bool(width)),
            mean_extra_prefill_tokens=float(np.mean([len(r['prefix']) for r in rows]))*5*bool(width),
            posterior_entropy_mean=float(entropy[~fm].mean()),posterior_entropy_quantiles=np.quantile(entropy[~fm],[0,.1,.5,.9,1]).tolist(),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
        print(name,records[name]['fit_game_cv'],ce.mean(),ce[3::4].mean(),params[0],flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(41912).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        refs=['parent']
        if rec['power']>0:refs.append(f"static{rec['width']}")
        for ref in refs:
            delta=losses[name]-losses[ref];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
            rec['delta_vs_'+ref]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
            by_ply={}
            for label,take in [('lt10',own_moves<10),('10to30',(own_moves>=10)&(own_moves<=30)),('gt30',own_moves>30)]:
                take&=~fm;cell_values=[float(delta[take&(cells==c)].mean()) if np.any(take&(cells==c)) else None for c in range(16)]
                by_ply[label]=dict(positions=int(take.sum()),per_cell_delta=cell_values,macro_delta=None if None in cell_values else float(np.mean(cell_values)),
                    expert_delta=None if any(cell_values[c] is None for c in (3,7,11,15)) else float(np.mean([cell_values[c] for c in (3,7,11,15)])))
            rec['delta_vs_'+ref]['own_moves_seen']=by_ply
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,analysis_plan_sha256=digest(pp),worker=worker))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
