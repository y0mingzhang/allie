"""Own/opponent soft-backup temperatures, using only cached model trees.

Select on 3-way game CV within August fit fold; report every arm on the
game-disjoint August confirmation fold. Both folds may be model-training-seen.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,atomic
from .compact_native import load
from .fit_policy import fit,loss_gradient
from .analyze_august import means

OUT=ROOT/'aug-compact-v1'
TAUS=[0.,.025,.1,.5,float('inf')]


def main():
    start=time.monotonic();module=load()
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows])
    fitmask=np.array([r['fold']==0 for r in rows]);cv=np.array([int(hashlib.sha256(('cv:'+r['game']).encode()).hexdigest(),16)%3 for r in rows])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):
        ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    root=np.zeros((n,2432));cost=np.zeros((2,n));qs=np.zeros((2,5,5,n,k))
    plan=json.loads((OUT/'plan.json').read_text());bs=plan['spec']['roots_per_batch']
    for lo in range(0,n,bs):
        with np.load(OUT/f'{lo:06d}.npz') as z:
            data={key:z[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            hi=lo+len(z['game']);assert list(z['game'])==list(games[lo:hi]);root[lo:hi]=z['z']
            for b,budget in enumerate((256,1000)):
                cost[b,lo:hi]=z['evaluated_nodes'][list(z['budgets']).index(budget)]
                for ti,tau in enumerate(TAUS):
                    for tj,opp in enumerate(TAUS):
                        q=module.reduce(data,budget,tau,opp)
                        qs[b,ti,tj,lo:hi]=q[np.arange(hi-lo)[:,None],ids[lo:hi]]
    reduction_seconds=time.monotonic()-start
    logits=root[:,378:2346][ar[:,None],ids];records={};losses={};cv_losses={}
    groups={'global':np.zeros(n,int),'elo':cells%4}
    def predict(q,group,training,predicting):
        params={};p=np.zeros_like(q)
        for g in np.unique(group):
            train=training&(group==g);pred=predicting&(group==g)
            count=np.bincount(cells[train],minlength=16)
            f=fit(logits[train],q[train],mask[train],target[train],'forward',1/count[cells[train]])
            params[str(int(g))]=f
            p[pred],_=loss_gradient([f['alpha'],f['beta']],logits[pred],q[pred],mask[pred],target[pred],'forward',return_policy=True)
        return p,params
    variants=[(f'soft{budget}_own{tau:g}_opp{opp:g}',qs[b,ti,tj],cost[b],budget,tau,opp)
              for b,budget in enumerate((256,1000)) for ti,tau in enumerate(TAUS) for tj,opp in enumerate(TAUS)]
    variants.append(('temperature',np.zeros_like(logits),np.zeros(n),0,None,None))
    for label,q,nodes,budget,tau,opp in variants:
        for grouping,group in groups.items():
            name=f'{label}_{grouping}';p,params=predict(q,group,fitmask,np.ones(n,bool));loss=-np.log(p[ar,target])
            oof=np.full(n,np.nan)
            for f in range(3):
                validation=fitmask&(cv==f)
                pred,_=predict(q,group,fitmask&(cv!=f),validation)
                oof[validation]=-np.log(pred[ar[validation],target[validation]])
            assert np.isfinite(oof[fitmask]).all()
            rec=dict(parameters=params,budget=budget,own_tau=tau if tau is None or np.isfinite(tau) else 'inf',
                     opponent_tau=opp if opp is None or np.isfinite(opp) else 'inf',group=grouping,
                     mean_nodes=float(means(nodes,cells).mean()),expert_mean_nodes=float(means(nodes,cells)[3::4].mean()),training_equivalent_cm=None)
            for split,values,m in [('fit',loss,fitmask),('confirmation',loss,~fitmask),('fit_game_cv',oof,fitmask)]:
                ce=means(values[m],cells[m]);rec[split]=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist())
            records[name]=rec;losses[name]=loss;cv_losses[name]=oof
        print('Asymmetric calibrated',label,flush=True)
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    # Paired whole-game uncertainty for selections versus identical-cost symmetric
    # backups. Sampling uses no model scores and keeps all 16 equal-weight cells.
    _,ix=np.unique(games[~fitmask],return_inverse=True);g=ix.max()+1
    count=np.zeros((g,16));np.add.at(count,(ix,cells[~fitmask]),1)
    rng=np.random.default_rng(611839);w=rng.multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    for name in set(selected.values()):
        ref='soft1000_own0.1_opp0.1_elo';diff=losses[name]-losses[ref]
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fitmask]),diff[~fitmask]);draws=(w@sums)/den
        rec=records[name];point=means(diff[~fitmask],cells[~fitmask])
        rec['confirmation_delta_vs_soft1000']=dict(reference=ref,macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August training-seen development only; CM pending unchanged July golden confirmation.',
        plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fit_cv_selected=selected,results=records,
        timing=dict(tree_reduction_seconds=reduction_seconds,total_analysis_seconds=time.monotonic()-start))
    atomic(OUT/'asymmetric-results.json',report)
    with (OUT/'asymmetric-scores.npz').open('wb') as f:
        np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cv_loss=np.stack(list(cv_losses.values())),cells=cells,games=games,fit=fitmask)
    print('SELECTED',selected,report['timing'],flush=True)
    for name in sorted(set(selected.values())|{'soft256_own0.1_opp0.1_elo','soft1000_own0.1_opp0.1_elo'}):
        print(name,records[name],flush=True)


if __name__=='__main__':main()
