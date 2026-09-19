"""Fit retrieval mixing on August games, with game CV and matched controls."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from .service import ROOT,atomic
from .balanced_eval import digest
from .fit_policy import fit,loss_gradient
from .analyze_august import means

def distribution(sim,labels,ids,mask,k,temp):
    sim=np.asarray(sim[:,:k],float);labels=labels[:,:k]
    valid=np.isfinite(sim)&(labels>=0)
    maximum=np.where(valid,sim,-np.inf).max(1);maximum=np.where(np.isfinite(maximum),maximum,0.)
    weights=np.exp(np.where(valid,(sim-maximum[:,None])/temp,-np.inf))
    p=np.zeros_like(ids,dtype=float)
    for j in range(k):
        p+=weights[:,j,None]*((labels[:,j,None]==ids)&mask)
    den=p.sum(1);nonempty=den>0
    p[nonempty]/=den[nonempty,None]
    assert np.isfinite(p).all() and (p[~mask]==0).all()
    return p,nonempty

def fit_mix(base,knn,nonempty,target,cells,train):
    ar=np.arange(len(base));a=base[ar,target];b=knn[ar,target]
    b=np.where(nonempty,b,a)
    counts=np.bincount(cells[train],minlength=16);w=1/counts[cells[train]];w/=w.sum()
    objective=lambda lam:float(w@(-np.log((1-lam)*a[train]+lam*b[train])))
    opt=minimize_scalar(objective,bounds=(0.,.95),method='bounded',options=dict(xatol=1e-9))
    assert opt.success
    return float(min([0.,float(opt.x),.95],key=objective))

def test():
    ids=np.array([[1,4,8],[2,7,0]]);mask=np.array([[1,1,1],[1,1,0]],bool)
    sim=np.array([[.8,.7,.6],[-np.inf,-np.inf,-np.inf]])
    lab=np.array([[1,1,8],[-1,-1,-1]])
    p,ok=distribution(sim,lab,ids,mask,3,.1)
    w=np.exp(np.array([.8,.7,.6])/.1);w/=w.sum()
    np.testing.assert_allclose(p[0],[w[0]+w[1],0.,w[2]])
    assert ok.tolist()==[True,False] and p[1].sum()==0
    cells=np.arange(16);b=np.tile([.7,.3],(16,1));knn=np.tile([.9,.1],(16,1));target=np.zeros(16,int)
    assert fit_mix(b,knn,np.ones(16,bool),target,cells,np.ones(16,bool))==.95
    assert fit_mix(b,knn,np.zeros(16,bool),target,cells,np.ones(16,bool))==0.
    print('PASS repeated-neighbor aggregation, masked padding, empty fallback and mixture endpoints')

def main():
    test();start=time.monotonic();out=ROOT/'aug-retrieval-v1'
    plan=json.loads((out/'plan.json').read_text());worker=json.loads((out/'worker.json').read_text())
    assert worker['plan_sha256']==digest(out/'plan.json')
    with np.load(out/'neighbors.npz') as zz:neighbor={k:zz[k] for k in zz.files}
    with np.load(out/'roots.npz') as zz:query_z=zz['z']
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions']
    assert digest(ROOT/'aug-tune-v1/sample.json')==plan['sample_sha256']
    n=len(rows);ar=np.arange(n);cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    np.testing.assert_array_equal(neighbor['game'],games);np.testing.assert_array_equal(neighbor['ply'],[r['ply'] for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):
        ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    control=ROOT/'aug-adaptive-root-v1';cp=json.loads((control/'plan.json').read_text())
    assert cp['sample_sha256']==plan['sample_sha256']
    root=np.zeros((n,2432));q=np.zeros((n,k));cost=np.zeros(n)
    for lo in range(0,n,cp['roots_per_batch']):
        with np.load(control/'quota_static'/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1]
            np.testing.assert_array_equal(z['game'],games[lo:hi]);np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk])
            root[lo:hi]=z['z'];q[lo:hi,:kk]=z['q'][cp['budgets'].index(1000)];cost[lo:hi]=z['evaluated_nodes'][cp['budgets'].index(1000)]
    logits=root[:,378:2346][ar[:,None],ids]
    group=cells%4
    def base_fit(q,train):
        p=np.zeros_like(q);params={}
        for g in range(4):
            take=train&(group==g);pred=group==g;count=np.bincount(cells[take],minlength=16)
            f=fit(logits[take],q[take],mask[take],target[take],'forward',1/count[cells[take]])
            assert f['converged'];params[str(g)]=f
            p[pred],_=loss_gradient([f['alpha'],f['beta']],logits[pred],q[pred],mask[pred],target[pred],'forward',return_policy=True)
        return p,params
    controls={}
    for name,bq in [('direct',np.zeros_like(q)),('search',q)]:
        controls[name]=[base_fit(bq,fm)]+[base_fit(bq,fm&(cv!=f)) for f in range(3)]
    records={};losses={}
    def record(name,p,oof,params,base,kernel):
        loss=-np.log(p[ar,target]);conf=means(loss[~fm],cells[~fm]);cvce=means(oof[fm],cells[fm]);acc=means((p.argmax(1)==target)[~fm],cells[~fm])
        records[name]=dict(parameters=params,base=base,kernel=kernel,training_equivalent_cm=None,
            mean_nodes=float(cost.mean()) if base=='search' else 0.,
            fit_game_cv=dict(macro_ce=float(cvce.mean()),expert_ce=float(cvce[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist(),macro_accuracy=float(acc.mean()),expert_accuracy=float(acc[3::4].mean())))
        losses[name]=loss
    for base,versions in controls.items():
        p,params=versions[0];oof=np.full(n,np.nan)
        for f in range(3):
            val=fm&(cv==f);pp=versions[f+1][0];oof[val]=-np.log(pp[ar[val],target[val]])
        record(base,p,oof,dict(base=params,mixing=0.),base,None)
        for kernel in plan['kernels']:
            knn,ok=distribution(neighbor['similarity'],neighbor['label'],ids,mask,kernel['k'],kernel['temperature'])
            lam=fit_mix(p,knn,ok,target,cells,fm);kp=np.where(ok[:,None],knn,p)
            mixed=(1-lam)*p+lam*kp
            cvloss=np.full(n,np.nan)
            for f in range(3):
                pp=versions[f+1][0];val=fm&(cv==f);train=fm&(cv!=f)
                ll=fit_mix(pp,knn,ok,target,cells,train);pk=np.where(ok[:,None],knn,pp)
                pm=(1-ll)*pp+ll*pk;cvloss[val]=-np.log(pm[ar[val],target[val]])
            name=f"{base}_k{kernel['k']}_t{kernel['temperature']}"
            record(name,mixed,cvloss,dict(base=params,mixing=lam),base,kernel)
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,rec in records.items():
        reference=rec['base'];delta=losses[name]-losses[reference];point=means(delta[~fm],cells[~fm])
        sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['confirmation_delta_vs_base']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August potentially training-seen; fit-game CV selection, disjoint confirmation, no golden CM.',
        fit_cv_selected=selected,results=records,analysis_seconds=time.monotonic()-start,
        query_root_drift_vs_control=dict(max_abs=float(np.max(np.abs(query_z-root))),mean_abs=float(np.mean(np.abs(query_z-root)))),
        source_sha256=digest(Path(__file__)),sample_sha256=plan['sample_sha256'],
        caveat='Extra human-game datastore, not pure search. Base calibration independently re-fit within every CV training fold; final coefficients fit only fold0. Search base remains unchanged cached static control; feature query numerical drift is not credited to retrieval.')
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name in records:print(name,records[name]['confirmation']['macro_ce'],records[name]['confirmation']['expert_ce'],'lambda',records[name]['parameters']['mixing'],flush=True)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
