"""Does early value convergence improve per-position budget allocation? CPU cached study."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .engine.service import ROOT,atomic
from .engine.balanced_eval import digest
from .engine.sample_data import read
from .engine.deep_frontier_live import features
from .engine.deep_frontier import BUDGETS,atomic_npz
from .engine.value_of_compute import choose,solve,design
from .engine.analyze_august import means


def main():
    begin=time.monotonic();out=ROOT/'aug-convergence-router-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-deep-frontier-v1';trees=ROOT/'aug-deep-scale-v1';d=read(trees.name)
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'));n=len(rows);ar=np.arange(n)
    plan=dict(source_sha256=digest(Path(__file__)),sample_sha256=digest(trees/'sample.json'),parent_plan_sha256=digest(source/'plan.json'),
        calibration={p.name:digest(p) for p in sorted(source.glob('calibration-*.npz'))},
        feature_spec='Append4 causal convergence features at no extra NN cost: log(.001+prior-weighted RMS(Q128-Q64)); log(.001+max absolute value change over legal moves); Jensen-Shannon divergence between softmax(log prior+8Q64) and softmax(log prior+8Q128); whether their top move differs. Coefficient8 fixed, never fit from targets.',
        selection='Nested game-cross-fitted targets unchanged from parent. Compare state vs state+convergence at caps512/1000/2000, fixed ridge.1, no extra hyperparameter sweep. No targets or future-search outputs in features. Same reused August checking games; no golden CM.',
        cost='Both methods pay128 upfront, then select from128/256/512/1000/4000/16000. Cached replay only; live rerun required before promotion.',
        distinction='Earlier instability test reallocated work BETWEEN ROOT ACTIONS. This tests allocation BETWEEN POSITIONS. A null on one does not reject the other.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    path=out/'features.npz'
    if not path.exists():
        sys.path.insert(0,str(ROOT/'runtime/cpu-audit'));import _allie_scaled_count as native
        root=np.zeros((n,2432));q128=np.zeros(mask.shape);q64=np.zeros(mask.shape);nodes=np.zeros((6,n))
        for p in sorted(source.glob('[0-9]*.npz')):
            with np.load(p) as f:
                lo=int(p.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi]);root[lo:hi]=f['root'];q128[lo:hi]=f['q'][0];nodes[:,lo:hi]=f['nodes']
            with np.load(trees/p.name) as f:
                t={key:f[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
                q64[lo:hi]=native.Backup(t,64,16.).reduce(np.log(.2),-.5,ids[lo:hi].astype(np.int32))[0]
            print('convergence-prefix',hi,n,flush=True)
        inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text());assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
        with np.load(inv/'feats.npz') as f:side=f['feats']
        seconds=np.array([side[r['row'],r['column']-1,0] for r in rows]);base=features(rows,root,q128,ids,mask,seconds)
        logits=np.where(mask,root[:,378:2346][ar[:,None],ids],-np.inf);prior=softmax(logits,axis=1)
        p64,p128=softmax(logits+8*q64,axis=1),softmax(logits+8*q128,axis=1);avg=(p64+p128)/2
        log=lambda x:np.log(np.maximum(x,1e-300))
        js=.5*np.sum(p64*(log(p64)-log(avg))+p128*(log(p128)-log(avg)),axis=1)
        delta=q128-q64
        extra=np.c_[np.log(.001+np.sqrt((prior*delta**2).sum(1))),np.log(.001+np.max(np.where(mask,abs(delta),0.),axis=1)),js,p64.argmax(1)!=p128.argmax(1)]
        atomic_npz(path,base=base,extra=extra,nodes=nodes,q64=q64,game=games)
    with np.load(path) as f:
        np.testing.assert_array_equal(f['game'],games);base,extra,nodes=f['base'],f['extra'],f['nodes']
    served=np.zeros((4,6,n));targets=np.zeros((4,6,n))
    for oi in range(4):
        for bi,b in enumerate(BUDGETS):
            with np.load(source/f'calibration-{oi}-{b}.npz') as f:served[oi,bi]=f['served'];targets[oi,bi]=f['targets']
    with np.load(source/'scores.npz') as f:oldloss=dict(zip(f['names'],f['loss']));oldchoices=dict(zip(f['names'],f['choices']))
    records={};losses={};choices={}
    for cap in (512,1000,2000):
        for kind,feat in [('state',base),('convergence',np.c_[base,extra])]:
            name=f'{kind}{cap}';params=[];oof=np.full(n,np.nan)
            for oi,outer in enumerate([None,0,1,2]):
                train=fm if outer is None else fm&(cv!=outer)
                x,mu,sd=design(cells,feat,train,'state');ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
                coef,penalty=solve(x[train],targets[oi,:,train],w,.1,BUDGETS,cap)
                chosen=choose(x,coef,penalty,BUDGETS);ll=served[oi,chosen,ar]
                params.append(dict(outer=outer,kind=kind,cap=cap,coef=coef.tolist(),penalty=penalty,mean=mu.tolist(),scale=sd.tolist()))
                if outer is None:losses[name]=ll;choices[name]=chosen
                else:take=fm&(cv==outer);oof[take]=ll[take]
            if kind=='state':
                np.testing.assert_array_equal(choices[name],oldchoices[name]);np.testing.assert_array_equal(losses[name],oldloss[name])
            a,b=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm]);cost=means(nodes[choices[name],ar][~fm],cells[~fm])
            records[name]=dict(parameters=params,training_equivalent_cm=None,mean_nodes=float(cost.mean()),expert_mean_nodes=float(cost[3::4].mean()),
                confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(512482).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        cap=rec['parameters'][0]['cap'];delta=losses[name]-losses[f'state{cap}'];pt=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_same_cap']=dict(macro=float(pt.mean()),expert=float(pt[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
        print(name,rec['confirmation'],rec['fit_game_cv'],rec['mean_nodes'],flush=True)
    selected={str(cap):{m:min([f'state{cap}',f'convergence{cap}'],key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')} for cap in (512,1000,2000)}
    atomic(out/'results.json',dict(results=records,fit_cv_selected_by_cap=selected,seconds=time.monotonic()-begin,plan_sha256=digest(pp)))
    atomic_npz(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),choices=np.stack(list(choices.values())),games=games,cells=cells,fit=fm)
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
