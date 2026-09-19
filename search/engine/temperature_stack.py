"""Test residual uncertainty calibration after the conditional strength mixture."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_expanded import components
from .analyze_player_search import FACTORS
from .analyze_august import means


def evaluate(theta, x, logp, mask, target, weights=None, ridge=0.):
    raw = 1+x@theta
    eta = np.clip(raw, .5, 2.)
    lp = np.where(mask, eta[:, None]*logp, -np.inf)
    lp -= logsumexp(lp, axis=1, keepdims=True)
    if weights is None: return np.exp(lp)
    p, ar = np.exp(lp), np.arange(len(target))
    residual = (p*logp).sum(1)-logp[ar, target]
    grad = x.T@(weights*residual*((raw>.5)&(raw<2.)))+ridge*theta
    return float(-weights@lp[ar, target]+.5*ridge*np.square(theta).sum()), grad


def test():
    rng = np.random.default_rng(143)
    logp = np.log(rng.dirichlet(np.ones(6), size=31))
    x = np.c_[np.ones(31), rng.normal(size=(31, 3))]
    mask = np.ones(logp.shape, bool); target = rng.integers(0, 6, 31)
    weights = np.ones(31)/31; theta = np.array([.1, .07, -.08, .06])
    _, g = evaluate(theta, x, logp, mask, target, weights, .01)
    fd = []
    for j in range(4):
        a, b = theta.copy(), theta.copy(); a[j] += 1e-6; b[j] -= 1e-6
        fd.append((evaluate(a,x,logp,mask,target,weights,.01)[0]-evaluate(b,x,logp,mask,target,weights,.01)[0])/2e-6)
    np.testing.assert_allclose(g, fd, atol=1e-9, rtol=1e-6)
    np.testing.assert_allclose(evaluate(np.zeros(4),x,logp,mask,target), np.exp(logp), atol=2e-16)
    print('PASS clipped temperature gradient and identity', flush=True)


def main():
    test(); start=time.monotonic()
    out=ROOT/'aug-temperature-stack-v1'; out.mkdir(exist_ok=True)
    stack_path=ROOT/'aug-expanded-stack-v1/results.json'
    gate_path=ROOT/'aug-conditional-mixture-v1/results.json'
    stack=json.loads(stack_path.read_text()); gate=json.loads(gate_path.read_text())
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n,ar=len(rows),np.arange(len(rows))
    root=np.zeros((n,2432)); q=np.zeros(mask.shape); nodes=np.zeros(n)
    for lo in range(0,n,1024):
        with np.load(ROOT/'aug-expanded-search-v1'/f'{lo:06d}.npz') as f:
            hi=lo+len(f['game']); kk=f['ids'].shape[1]
            np.testing.assert_array_equal(f['game'],games[lo:hi]);root[lo:hi]=f['z'];q[lo:hi,:kk]=f['q'][-1,1];nodes[lo:hi]=f['evaluated_nodes'][-1]
    inventory=ROOT/'aug-tune-v1'; manifest=json.loads((inventory/'manifest.json').read_text())
    for name in ('strat.npz','feats.npz'):assert digest(inventory/name)==manifest['files_sha256'][name]
    with np.load(inventory/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(inventory/'feats.npz') as f:feats=f['feats']
    seconds=[]
    for r in rows:
        rr,cc=r['row'],r['column'];assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(r['prefix']):cc],r['prefix']);seconds.append(feats[rr,cc-1,0])
    seconds=np.array(seconds); known=seconds>=0
    z=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    prior=softmax(np.where(mask,z,-np.inf),axis=1);qm=(prior*q).sum(1)
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    qstd=np.sqrt((prior*(q-qm[:,None])**2).sum(1))
    tp=softmax(root[:,2350:2413],axis=1); time_centers=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    features=np.column_stack([entropy,np.log(.01+qstd),[len(r['prefix'])-11 for r in rows],np.log1p(tp@time_centers),
        np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known])
    menus=[('fixed',[],None),('global',[],0.)]
    for name,fields in [('state',[0,1,2,3]),('clock',[4,5,6]),('all',list(range(7)))]:
        for ridge in (.001,.01):menus.append((name+str(ridge),fields,ridge))
    plan=dict(sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),stack_sha256=digest(stack_path),gate_sha256=digest(gate_path),
        source_sha256=digest(Path(__file__)),menus=menus,
        formula='After frozen-fold all0.001 strength mixture pi, apply pi_new proportional to pi^clip(1+theta*x,.5,2). Separates uncertainty in prior/action probabilities from latent search strength. No neural-weight updates.',
        fit='Feature normalization and residual temperature fit inside each game-fold on independently fitted parent gate/components. All arms evaluated on disjoint August confirmation; no golden selection. Clock is pre-move.',
        cost='No extra neural queries; same1000 trees. Original parent is included exactly.')
    plan=json.loads(json.dumps(plan)); pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    parent=[]
    for version,entry in enumerate(stack['fold_parameters']['subtree']):
        comp=components(z,q,mask,entry['parameters'],cells%4)
        g=gate['results']['all0.001']['parameters'][version];assert g['fold']==entry['fold']
        xx=np.c_[np.ones(n),np.clip((features[:,g['fields']]-g['mean'])/g['scale'],-3,3)]
        mu=xx@np.array(g['theta']);mu[~known]=0
        w=softmax(-.5*np.log(FACTORS)[None,:]**2+mu[:,None]*np.log(FACTORS),axis=1)
        p=np.einsum('na,ank->nk',w,comp)
        parent.append(np.where(mask,np.log(np.maximum(p,1e-300)),0.))
    records,losses={},{}
    for name,fields,ridge in menus:
        oof=np.full(n,np.nan); params=[]
        for version,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            f=features[:,fields];mean=f[train].mean(0);scale=np.maximum(f[train].std(0),1e-6)
            x=np.c_[np.ones(n),np.clip((f-mean)/scale,-3,3)]
            count=np.bincount(cells[train],minlength=16);w=1/count[cells[train]];w/=w.sum()
            theta=np.zeros(x.shape[1])
            if ridge is not None:
                opt=minimize(lambda t:evaluate(t,x[train],parent[version][train],mask[train],target[train],w,ridge),theta,jac=True,
                    method='L-BFGS-B',bounds=[(-1,1)]*len(theta),options=dict(ftol=1e-11,gtol=1e-7,maxiter=250))
                assert opt.success,opt;theta=opt.x
            p=evaluate(theta,x,parent[version],mask,target);loss=-np.log(p[ar,target])
            params.append(dict(fold=fold,theta=theta.tolist(),fields=fields,mean=mean.tolist(),scale=scale.tolist(),ridge=ridge))
            if fold is None:losses[name]=loss
            else:oof[fm&(cv==fold)]=loss[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,training_equivalent_cm=None,mean_nodes=float(nodes.mean()),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
        print(name,records[name]['fit_game_cv'],ce.mean(),ce[3::4].mean(),flush=True)
    for field in ('confirmation','fit_game_cv'):
        for metric in ('macro_ce','expert_ce'):
            np.testing.assert_allclose(records['fixed'][field][metric],gate['results']['all0.001'][field][metric],atol=1e-12,rtol=0)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['fixed'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['delta_vs_fixed']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(stage=plan['fit'],results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp)))
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
