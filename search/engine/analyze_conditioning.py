"""Counterfactual-strength guidance, alone and stacked with the fixed soft backup."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .service import ROOT,atomic
from .analyze_august import means
from .compact_native import load
from .conditioning_pilot import VARIANTS


def main():
    start=time.monotonic();folder=ROOT/'aug-conditioning-v1';n=4096
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];assert len(rows)==n
    ar=np.arange(n);cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);target=np.zeros(n,int);ids=np.zeros((n,k),int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    logits={name:np.load(folder/f'{name}.npz')['logits'].astype(float) for name,_,_ in VARIANTS}
    base=logits['base'];q=np.zeros_like(base);cost=np.zeros(n)
    compact=load();study=ROOT/'aug-compact-v1';plan=json.loads((study/'plan.json').read_text());bs=plan['spec']['roots_per_batch']
    for lo in range(0,n,bs):
        with np.load(study/f'{lo:06d}.npz') as z:
            data={key:z[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi])
            q[lo:hi]=compact.reduce(data,1000,.1,.1)[np.arange(hi-lo)[:,None],ids[lo:hi]]
            cost[lo:hi]=z['evaluated_nodes'][list(z['budgets']).index(1000)]
    def fit_predict(x,bounds,group,train,pred):
        out=np.zeros((n,k));params={}
        for g in np.unique(group):
            a=train&(group==g);b=pred&(group==g);count=np.bincount(cells[a],minlength=16)
            w=1/count[cells[a]];w/=w.sum();xf=x[a];mf=mask[a];yf=target[a]
            def objective(theta):
                z=np.where(mf,xf@theta,-np.inf);lp=z-logsumexp(z,axis=1,keepdims=True);p=np.exp(lp)
                return -w@lp[np.arange(len(yf)),yf],np.einsum('n,nk,nkd->d',w,p,xf)-np.einsum('n,nd->d',w,xf[np.arange(len(yf)),yf])
            opt=minimize(objective,[1.]+[0.]*(x.shape[-1]-1),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=120,ftol=1e-11,gtol=1e-7))
            assert np.isfinite(opt.fun);params[str(int(g))]=dict(theta=opt.x.tolist(),converged=bool(opt.success))
            z=np.where(mask[b],x[b]@opt.x,-np.inf);out[b]=np.exp(z-logsumexp(z,axis=1,keepdims=True))
        return out,params
    menus=[]
    for name,_,_ in VARIANTS:
        menus.append((name+'_direct',[logits[name]],[ (.4,1.6)],0 if name=='base' else 1))
        if name=='base':continue
        # Logit differences are invariant to action-independent log normalizers.
        menus.append((name+'_guidance',[base,logits[name]-base],[(.4,1.6),(-2.,2.)],1))
        menus.append((name+'_stack',[base,q,logits[name]-base],[(.4,1.6),(0.,40.),(-2.,2.)],1))
    menus.append(('soft1000',[base,q],[(.4,1.6),(0.,40.)],0))
    records={};losses={}
    for name,features,bounds,extra in menus:
        x=np.stack(features,axis=-1);nodes=cost if name.endswith('_stack') or name=='soft1000' else np.zeros(n)
        for grouping,group in [('global',np.zeros(n,int)),('elo',cells%4)]:
            label=f'{name}_{grouping}';p,params=fit_predict(x,bounds,group,fm,np.ones(n,bool));loss=-np.log(p[ar,target]);oof=np.full(n,np.nan)
            for fold in range(3):
                val=fm&(cv==fold);pred,_=fit_predict(x,bounds,group,fm&(cv!=fold),val)
                oof[val]=-np.log(pred[ar[val],target[val]])
            c=means(oof[fm],cells[fm]);d=means(loss[~fm],cells[~fm])
            records[label]=dict(parameters=params,group=grouping,variant=name,extra_full_prefix_queries=extra,
                mean_search_nodes=float(means(nodes,cells).mean()),training_equivalent_cm=None,
                fit_game_cv=dict(macro_ce=float(c.mean()),expert_ce=float(c[3::4].mean())),
                confirmation=dict(macro_ce=float(d.mean()),expert_ce=float(d[3::4].mean()),cells=d.tolist()))
            losses[label]=loss
        print('Conditioning calibrated',name,flush=True)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(91383).multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    for name in set(selected.values()):
        diff=losses[name]-losses['soft1000_elo'];sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),diff[~fm]);draws=(w@sums)/den
        records[name]['confirmation_delta_vs_soft1000_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August training-seen development only. Shifted ratings are hypothetical inference queries, not the user rating supplied to the baseline. Golden CM pending.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fit_cv_selected=selected,results=records,analysis_seconds=time.monotonic()-start)
    atomic(folder/'results.json',report)
    with (folder/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,'seconds',report['analysis_seconds'],flush=True)
    for name in sorted(set(selected.values())|{'soft1000_elo','base_direct_elo'}):print(name,records[name],flush=True)


if __name__=='__main__':main()
