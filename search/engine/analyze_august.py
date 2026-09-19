"""Equal-cell August calibration, game CV on fit only, untouched confirmation.

August may overlap model training. Neither its losses nor its fitted gains are
converted with July's scaling law. A selected policy still requires golden scoring.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from .service import ROOT,atomic
from .fit_policy import fit,loss_gradient

OUT=ROOT/'aug-search-v1'


def means(x,cells):
    n=np.bincount(cells,minlength=16);assert (n>0).all()
    return np.bincount(cells,weights=x,minlength=16)/n


def main():
    start=time.monotonic();rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions']
    plan=json.loads((OUT/'plan.json').read_text());n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);fitmask=np.array([r['fold']==0 for r in rows]);expert=cells%4==3
    games=np.array([r['game'] for r in rows]);cv=np.array([int(hashlib.sha256(('cv:'+r['game']).encode()).hexdigest(),16)%3 for r in rows])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    root=np.zeros((n,2432),float);treeq=np.zeros((4,n,k),float);treecost=np.zeros((4,n),float)
    mctsq=np.zeros((4,6,n,k),float);mctscost=np.zeros((4,n),float);visits=np.zeros((4,n,k),float);stats=[]
    for kind,bs in [('tree',32),('mcts',128)]:
        for lo in range(0,n,bs):
            with np.load(OUT/f'{kind}-{lo:06d}.npz') as z:
                hi=lo+len(z['game']);kk=z['ids'].shape[1]
                assert list(z['game'])==list(games[lo:hi])
                np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk]);np.testing.assert_array_equal(z['mask'],mask[lo:hi,:kk])
                stats.append(dict(kind=kind,**json.loads(str(z['stats']))))
                if kind=='tree':root[lo:hi]=z['root'];treeq[:,lo:hi,:kk]=z['q'];treecost[:,lo:hi]=z['evaluated_nodes']
                else:mctsq[:,:,lo:hi,:kk]=z['q'];mctscost[:,lo:hi]=z['evaluated_nodes'];visits[:,lo:hi,:kk]=z['visits']
    logits=root[:,378:2346][ar[:,None],ids]
    raw=logsumexp(root[:,378:2346],axis=1)-root[ar,np.array([r['target'] for r in rows])]
    probabilities={};losses={};records={};cv_losses={}
    def record(name,p,nll,cost,params=None,cvloss=None):
        probabilities[name]=p;losses[name]=nll
        if cvloss is not None:cv_losses[name]=cvloss
        rec=dict(parameters=params,training_equivalent_cm=None,
            mean_nodes=float(means(cost,cells).mean()),expert_mean_nodes=float(means(cost,cells)[3::4].mean()))
        for label,m in [('fit',fitmask),('confirmation',~fitmask)]:
            ce=means(nll[m],cells[m]);accuracy=means((p.argmax(1)==target)[m],cells[m])
            rec[label]=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist(),
                macro_accuracy=float(accuracy.mean()),expert_accuracy=float(accuracy[3::4].mean()))
        if cvloss is not None:
            ce=means(cvloss[fitmask],cells[fitmask]);rec['fit_game_cv']=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist())
        records[name]=rec
    zero=np.zeros_like(logits)
    p,nll=loss_gradient([1.,0.],logits,zero,mask,target,'forward',return_policy=True)
    record('legal',p,nll,np.zeros(n))
    # Fitted family menu is declared in source before scoring. Every arm reported.
    variants=[('temperature',zero,np.zeros(n),'forward')]
    for depth in (2,4):
        for direction in ('forward','reverse'):variants.append((f'depth{depth}_{direction}',treeq[depth-1],treecost[depth-1],direction))
    for b,budget in enumerate(plan['budgets']):
        for j,label in [(0,'mcts'),(1,'expectation'),(3,'soft0.1')]:
            for direction in ('forward','reverse'):variants.append((f'{label}{budget}_{direction}',mctsq[b,j],mctscost[b],direction))
    groups={'global':np.zeros(n,int),'elo':cells%4,'format':cells//4}
    def train_predict(q,direction,group,training,predicting):
        params={};p=np.zeros_like(q)
        for g in np.unique(group):
            train=training&(group==g);pred=predicting&(group==g)
            count=np.bincount(cells[train],minlength=16);weight=1/count[cells[train]]
            fitted=fit(logits[train],q[train],mask[train],target[train],direction,weight)
            params[str(int(g))]=fitted
            p[pred],_=loss_gradient([fitted['alpha'],fitted['beta']],logits[pred],q[pred],mask[pred],target[pred],direction,return_policy=True)
        return p,params
    for base,q,cost,direction in variants:
        for grouping,group in groups.items():
            name=base+'_'+grouping;p,params=train_predict(q,direction,group,fitmask,np.ones(n,bool))
            nll=-np.log(p[ar,target]);oof=np.full(n,np.nan)
            for fold in range(3):
                validation=fitmask&(cv==fold)
                cp,_=train_predict(q,direction,group,fitmask&(cv!=fold),validation)
                oof[validation]=-np.log(cp[ar[validation],target[validation]])
            assert np.isfinite(oof[fitmask]).all()
            record(name,p,nll,cost,params,oof)
        print('August calibrated',base,flush=True)
    # Select separately for macro and expert using only OUT-OF-FOLD fit scores.
    selected={metric:min(cv_losses,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    # Whole-game paired bootstrap on confirmation only, for preregistered simple
    # controls and fit-CV selections. No best-confirmation selection is performed.
    _,ix=np.unique(games[~fitmask],return_inverse=True);g=ix.max()+1
    count=np.zeros((g,16));np.add.at(count,(ix,cells[~fitmask]),1)
    rng=np.random.default_rng(613812);w=rng.multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    keys=set(selected.values())|{'depth4_forward_global','mcts1000_reverse_global','temperature_global','legal'}
    reference=losses['legal']
    for name in sorted(keys):
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fitmask]),(losses[name]-reference)[~fitmask])
        draws=(w@sums)/den
        point=means((losses[name]-reference)[~fitmask],cells[~fitmask])
        records[name]['confirmation_delta_vs_legal']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    rawce=means(raw[~fitmask],cells[~fitmask])
    report=dict(stage='August tuning/confirmation. Both may be training-seen; not golden and no CM conversion.',
        sample_sha256=hashlib.sha256((ROOT/'aug-tune-v1/sample.json').read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('fit_policy.py')]},
        fit_cv_selected=selected,raw_confirmation=dict(macro_ce=float(rawce.mean()),expert_ce=float(rawce[3::4].mean())),
        results=records,timing=dict(worker_seconds=sum(s['seconds'] for s in stats),analysis_seconds=time.monotonic()-start),
        nodes_semantics='Actual newly model-evaluated non-root nodes per position; equal-cell means. Tree horizons share computation; standalone wall times not inferred from snapshots.')
    atomic(OUT/'results.json',report)
    with (OUT/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,fit=fitmask,games=games)
    for name in sorted(keys):print(name,records[name]['confirmation'],flush=True)
    print('FIT-CV selections',selected,'seconds',report['timing'],flush=True)


if __name__=='__main__':main()
