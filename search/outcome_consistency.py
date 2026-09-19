"""Enforce agreement between root WDL and policy-weighted child WDL."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax,logsumexp
from .engine.service import ROOT,atomic
from .engine.balanced_eval import digest
from .engine.retrieval_common import data
from .engine.analyze_august import means

def consistency(base,child,desired,iterations):
    p=base.copy()
    for _ in range(iterations):
        implied=(p[:,:,None]*child).sum(1)
        factor=(child*(desired/np.maximum(implied,1e-30))[:,None,:]).sum(-1)
        p*=factor;p/=p.sum(1,keepdims=True)
    return p

def policy(base,corrected,strength,mask):
    delta=np.log(np.maximum(corrected,1e-100))-np.log(np.maximum(base,1e-100))
    logits=np.where(mask,np.log(np.maximum(base,1e-100))+strength*delta,-np.inf)
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))

def test():
    rng=np.random.default_rng(223);p=rng.dirichlet(np.ones(7),size=19)
    child=rng.dirichlet(np.ones(3),size=(19,7));root=(p[:,:,None]*child).sum(1)
    for it in (1,4,16):np.testing.assert_allclose(consistency(p,child,root,it),p,atol=1e-14)
    fixed=np.broadcast_to(root[:,None,:],child.shape);desired=rng.dirichlet(np.ones(3),size=19)
    np.testing.assert_allclose(consistency(p,fixed,desired,16),p,atol=1e-14)
    corr=consistency(p,child,desired,4);np.testing.assert_allclose(corr.sum(1),1.)
    np.testing.assert_allclose(policy(p,corr,0,np.ones_like(p,bool)),p)
    assert (corr>0).all()
    print('PASS outcome consistency fixed points, uninformative-critic invariance, normalization and no-change limit')

def main():
    test();start=time.monotonic();out=ROOT/'aug-outcome-consistency-v1';out.mkdir(exist_ok=True)
    d=data();rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];target=d['target'];mask=d['mask'];n=len(rows);ar=np.arange(n)
    child=np.zeros((*mask.shape,3));root=np.zeros((n,3));nodes=np.zeros(n)
    src=ROOT/'aug-child-features-v1';sp=json.loads((src/'plan.json').read_text())
    assert sp['sample_sha256']==digest(ROOT/'aug-tune-v1/sample.json')
    for lo in range(0,n,sp['roots_per_batch']):
        with np.load(src/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);np.testing.assert_array_equal(z['game'],games[lo:hi]);f=z['features'];kk=f.shape[1]
            q=f[:,:,4];draw=f[:,:,5];child[lo:hi,:kk]=np.stack([(1-draw+q)/2,draw,(1-draw-q)/2],-1)
            root[lo:hi]=softmax(z['root'][:,2413:2416].astype(float),axis=1);nodes[lo:hi]=z['nodes']
    child=np.where(mask[:,:,None],np.clip(child,0,1),0.)
    np.testing.assert_allclose(child.sum(-1)[mask],1.,atol=1e-12)
    plan=dict(iterations=[1,4,16],sample_sha256=digest(ROOT/'aug-tune-v1/sample.json'),source_sha256=digest(Path(__file__)),
        input='Root and all legal child predicted WDL, child Win/Loss flipped to root-mover perspective; terminal WDL exact.',
        method='p_new(a)=p(a)*sum_o P_root(o)*P_child(o|a)/sum_b p(b)*P_child(o|b). Repeat1/4/16 times, then fit bounded interpolation in log probability (strength0..8). Consistent heads leave p unchanged.',
        fit='Fit-only game CV within August, with direct/search base refit in every fold. Scalar per arm, no true game outcomes or future moves.')
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:atomic(path,plan)
    records={};losses={}
    def record(name,p,oof,params,base,iterations):
        loss=-np.log(p[ar,target]);conf=means(loss[~fm],cells[~fm]);cc=means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,base=base,iterations=iterations,training_equivalent_cm=None,
            mean_nodes=float(nodes.mean()+(d['cost'].mean() if base=='search' else 0.)),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        losses[name]=loss
    for base,versions in d['controls'].items():
        p,params=versions[0];oof=np.full(n,np.nan)
        for fold in range(3):
            val=fm&(cv==fold);pp=versions[fold+1][0];oof[val]=-np.log(pp[ar[val],target[val]])
        record(base,p,oof,dict(base=params,strength=0.),base,0);records[base]['mean_nodes']=float(d['cost'].mean()) if base=='search' else 0.
        for iterations in plan['iterations']:
            def fit_predict(p,train):
                corr=consistency(p,child,root,iterations);count=np.bincount(cells[train],minlength=16);w=1/count[cells[train]];w/=w.sum()
                objective=lambda a:float(w@(-np.log(policy(p,corr,a,mask)[ar[train],target[train]])))
                opt=minimize_scalar(objective,bounds=(0.,8.),method='bounded',options=dict(xatol=1e-8));assert opt.success
                a=float(min([0.,float(opt.x),8.],key=objective));return policy(p,corr,a,mask),a
            pp,a=fit_predict(p,fm);oof=np.full(n,np.nan)
            for fold in range(3):
                cp,_=fit_predict(versions[fold+1][0],fm&(cv!=fold));val=fm&(cv==fold);oof[val]=-np.log(cp[ar[val],target[val]])
            record(base+f'_iter{iterations}',pp,oof,dict(base=params,strength=a),base,iterations)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    report=dict(stage='August model-only inference; disjoint parameter-fit/confirmation games, potentially training-seen, no golden CM.',
        fit_cv_selected=selected,results=records,seconds=time.monotonic()-start,plan_sha256=digest(path),
        cost_caveat='Conservative stack cost counts all-legal children plus search separately. A fused execution could share those child calls, but has not been timed.')
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,x in records.items():print(name,x['confirmation']['macro_ce'],x['confirmation']['expert_ce'],x['parameters']['strength'],flush=True)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
