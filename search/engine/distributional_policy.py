"""Same-node WDL/draw preference and outcome-consistency experiments on August."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from .service import ROOT,atomic
from .analyze_august import means


def consistency(prior,q,root):
    """One Bayesian mixture update; coherent predictions leave prior unchanged."""
    marginal=(prior[:,:,None]*q).sum(1)
    ratio=(q*(root/np.maximum(marginal,1e-12))[:,None,:]).sum(2)
    return np.log(np.maximum(ratio,1e-12))


def test():
    rng=np.random.default_rng(632);p=softmax(rng.normal(size=(12,7)),axis=1)
    q=softmax(rng.normal(size=(12,7,3)),axis=2);r=(p[:,:,None]*q).sum(1)
    np.testing.assert_allclose(consistency(p,q,r),0,atol=1e-14)
    r=softmax(rng.normal(size=(12,3)),axis=1)
    np.testing.assert_allclose((p*np.exp(consistency(p,q,r))).sum(1),1,atol=1e-12)


def main():
    test();start=time.monotonic();folder=ROOT/'aug-search-v1'
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);target=np.zeros(n,int);ids=np.zeros((n,k),int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    root=np.zeros((n,2432),float);wdl=np.zeros((2,n,k,3),float);qs=np.zeros((2,3,n,k),float);visits=np.zeros((2,n,k),int);cost=np.zeros((2,n))
    for lo in range(0,n,128):
        with np.load(folder/f'mcts-{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1];assert list(z['game'])==list(games[lo:hi])
            root[lo:hi]=z['root'];wdl[:,lo:hi,:kk]=z['wdl'][2:];visits[:,lo:hi,:kk]=z['visits'][2:];cost[:,lo:hi]=z['evaluated_nodes'][2:]
            qs[:,:,lo:hi,:kk]=z['q'][2:,:3]
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    prior=softmax(np.where(mask,logits,-np.inf),axis=1);rv=softmax(root[:,2413:2416],axis=1)
    prior_loss=-np.log(prior[ar,target]);records={};losses={}
    def calibrate(x,bounds,group,train,pred):
        out=np.zeros((n,k));params={}
        for g in np.unique(group):
            a=train&(group==g);b=pred&(group==g);count=np.bincount(cells[a],minlength=16)
            weight=1/count[cells[a]];weight/=weight.sum();xf=x[a];mf=mask[a];yf=target[a]
            def objective(theta):
                z=np.einsum('nkd,d->nk',xf,theta);z=np.where(mf,z,-np.inf)
                lp=z-logsumexp(z,axis=1,keepdims=True);p=np.exp(lp)
                value=-weight@lp[np.arange(len(yf)),yf]
                grad=np.einsum('n,nk,nkd->d',weight,p,xf)-np.einsum('n,nd->d',weight,xf[np.arange(len(yf)),yf])
                return value,grad
            opt=minimize(objective,[1.]+[0.]*(x.shape[-1]-1),jac=True,method='L-BFGS-B',bounds=bounds,
                         options=dict(maxiter=120,ftol=1e-11,gtol=1e-7))
            assert np.isfinite(opt.fun);params[str(int(g))]=dict(theta=opt.x.tolist(),converged=bool(opt.success))
            out[b]=softmax(np.where(mask[b],np.einsum('nkd,d->nk',x[b],opt.x),-np.inf),axis=1)
        return out,params
    for b,budget in enumerate((256,1000)):
        # Unvisited MCTS actions do not have an outcome prediction. For the
        # consistency step alone they inherit root WDL; scalar Q remains zero.
        qdist=np.where((visits[b]>0)[:,:,None],wdl[b],rv[:,None,:])
        np.testing.assert_allclose(qdist.sum(2),1,atol=2e-7)
        coherence=consistency(prior,qdist,rv);draw=qdist[:,:,1]
        features=dict(z=logits,q=qs[b,0],draw=draw,risk=draw*(rv[:,0]-rv[:,2])[:,None],
            consistency=coherence,expectation=qs[b,1],soft=qs[b,2])
        menus=[('q',),('q','draw'),('q','draw','risk'),('q','consistency'),('q','draw','consistency'),
               ('consistency',),('q','expectation'),('q','soft')]
        for menu in menus:
            fields=('z',*menu);x=np.stack([features[f] for f in fields],axis=2)
            bounds=[(.4,1.6)]+[(-20.,20.) if f in ('draw','risk') else ((-4.,4.) if f=='consistency' else (0.,40.)) for f in menu]
            for label,group in [('global',np.zeros(n,int)),('elo',cells%4),('format',cells//4)]:
                name=f'wdl{budget}_'+('_'.join(menu))+'_'+label
                p,params=calibrate(x,bounds,group,fm,np.ones(n,bool));nll=-np.log(p[ar,target]);losses[name]=nll
                oof=np.full(n,np.nan)
                for fold in range(3):
                    val=fm&(cv==fold);cp,_=calibrate(x,bounds,group,fm&(cv!=fold),val)
                    oof[val]=-np.log(cp[ar[val],target[val]])
                c=means(oof[fm],cells[fm]);d=means(nll[~fm],cells[~fm]);acc=means((p.argmax(1)==target)[~fm],cells[~fm])
                records[name]=dict(features=fields,parameters=params,training_equivalent_cm=None,
                    mean_nodes=float(means(cost[b],cells).mean()),expert_mean_nodes=float(means(cost[b],cells)[3::4].mean()),
                    fit_game_cv=dict(macro_ce=float(c.mean()),expert_ce=float(c[3::4].mean())),
                    confirmation=dict(macro_ce=float(d.mean()),expert_ce=float(d[3::4].mean()),cells=d.tolist(),
                        macro_accuracy=float(acc.mean()),expert_accuracy=float(acc[3::4].mean())))
            print('WDL calibrated',budget,menu,flush=True)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1
    count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(613813).multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    for name in set(selected.values())|{'wdl1000_q_global','wdl256_q_global'}:
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-prior_loss)[~fm]);delta=(w@sums)/den
        records[name]['confirmation_delta_vs_legal_ci95']=dict(macro=np.quantile(delta.mean(1),[.025,.975]).tolist(),expert=np.quantile(delta[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August potentially training-seen development; same-node feature tests; no golden CM.',
        input_sha256=hashlib.sha256((ROOT/'aug-tune-v1/sample.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fit_cv_selected=selected,results=records,
        analysis_seconds=time.monotonic()-start)
    atomic(folder/'distributional-results.json',report)
    with (folder/'distributional-scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,fit=fm,games=games)
    for name in set(selected.values())|{'wdl1000_q_global','wdl256_q_global'}:print(name,records[name]['confirmation'],flush=True)
    print('WDL FIT-CV selection',selected,report['analysis_seconds'],flush=True)


if __name__=='__main__':main()
