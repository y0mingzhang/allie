"""Fit-CV choice of FPU/exploration variants; confirmation remains game-disjoint."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,atomic
from .compact_native import load
from .fit_policy import fit,loss_gradient
from .analyze_august import means


def main(folder='aug-selection-v1'):
    start=time.monotonic();out=ROOT/folder;assert out.resolve().parent==ROOT.resolve()
    variant_spec=json.loads((out/'plan.json').read_text())['variants']
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions']
    n=len(rows);ar=np.arange(n);cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    module=load();variants={};budgets=[64,256,1000]
    for name in ['zero',*[v[0] for v in variant_spec]]:
        q=np.zeros((3,n,k));cost=np.zeros((3,n));root=np.zeros((n,2432))
        for lo in range(0,n,512):
            file=(ROOT/'aug-compact-v1' if name=='zero' else out/name)/f'{lo:06d}.npz'
            with np.load(file) as z:
                hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);root[lo:hi]=z['z']
                if name=='zero':
                    data={key:z[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
                    for b,budget in enumerate(budgets):
                        q[b,lo:hi]=module.reduce(data,budget,.1,.1)[np.arange(hi-lo)[:,None],ids[lo:hi]]
                        cost[b,lo:hi]=z['evaluated_nodes'][list(z['budgets']).index(budget)]
                else:
                    kk=z['ids'].shape[1];np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk]);np.testing.assert_array_equal(z['mask'],mask[lo:hi,:kk])
                    q[:,lo:hi,:kk]=z['q'];cost[:,lo:hi]=z['evaluated_nodes']
        if name=='zero':baseline=root.copy()
        else:np.testing.assert_array_equal(root,baseline)
        variants[name]=(q,cost)
    logits=baseline[:,378:2346][ar[:,None],ids];records={};losses={}
    def predict(q,group,training,predicting):
        p=np.zeros_like(q);params={}
        for g in np.unique(group):
            train=training&(group==g);pred=predicting&(group==g);count=np.bincount(cells[train],minlength=16)
            f=fit(logits[train],q[train],mask[train],target[train],'forward',1/count[cells[train]]);params[str(int(g))]=f
            p[pred],_=loss_gradient([f['alpha'],f['beta']],logits[pred],q[pred],mask[pred],target[pred],'forward',return_policy=True)
        return p,params
    for name,(qs,costs) in variants.items():
        for b,budget in enumerate(budgets):
            q=qs[b];cost=costs[b]
            for grouping,group in [('global',np.zeros(n,int)),('elo',cells%4)]:
                label=f'{name}_{budget}_{grouping}';p,params=predict(q,group,fm,np.ones(n,bool));loss=-np.log(p[ar,target]);oof=np.full(n,np.nan)
                for f in range(3):
                    val=fm&(cv==f);pred,_=predict(q,group,fm&(cv!=f),val);oof[val]=-np.log(pred[ar[val],target[val]])
                d=means(loss[~fm],cells[~fm]);c=means(oof[fm],cells[fm]);accuracy=means((p.argmax(1)==target)[~fm],cells[~fm])
                records[label]=dict(variant=name,budget=budget,group=grouping,parameters=params,
                    mean_nodes=float(means(cost,cells).mean()),expert_mean_nodes=float(means(cost,cells)[3::4].mean()),training_equivalent_cm=None,
                    fit_game_cv=dict(macro_ce=float(c.mean()),expert_ce=float(c[3::4].mean())),
                    confirmation=dict(macro_ce=float(d.mean()),expert_ce=float(d[3::4].mean()),cells=d.tolist(),
                                      macro_accuracy=float(accuracy.mean()),expert_accuracy=float(accuracy[3::4].mean())))
                losses[label]=loss
            print('Selection calibrated',name,budget,flush=True)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    for name in set(selected.values())|{v+'_1000_elo' for v in variants}:
        diff=losses[name]-losses['zero_1000_elo'];sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),diff[~fm]);draws=(w@sums)/den
        records[name]['confirmation_delta_vs_zero1000_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August potentially model-training-seen development; CM pending unchanged July golden.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fit_cv_selected=selected,results=records,analysis_seconds=time.monotonic()-start)
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,'seconds',report['analysis_seconds'],flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else 'aug-selection-v1')
