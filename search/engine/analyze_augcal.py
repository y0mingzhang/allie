"""Frozen August-selected policies: paired golden CE/CM and search-node frontier."""
import json
import time
from pathlib import Path
import numpy as np
from .analyze_balanced import bootstrap_deltas,cellmean
from .balanced_eval import ROOT,atomic,digest
from ..training_cm import multiplier

OUT=ROOT/'golden-augcal-v1'


def main():
    start=time.monotonic();plan=json.loads((OUT/'plan.json').read_text())
    assert digest(ROOT/'golden-balanced-v1/sample.json')==plan['sample_sha256']
    assert digest(ROOT/'training-cm-laws.json')==plan['laws_sha256']
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];n=len(rows)
    names=['port_raw','legal',*plan['methods']];scores={k:[] for k in names};stats=[];cost=[];elapsed=[]
    for lo in range(0,n,plan['roots_per_block']):
        with np.load(OUT/f'{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            assert list(z['ply'])==[r['ply'] for r in rows[lo:lo+len(z['game'])]]
            for k in names:scores[k].append(z[k])
            stats.append(json.loads(str(z['stats'])));cost.append(z['evaluated_nodes']);elapsed.append(z['elapsed'])
    scores={k:np.concatenate(v) for k,v in scores.items()};cost=np.concatenate(cost,axis=1)
    old=[];old_nodes=0
    for lo in range(0,n,32):
        with np.load(ROOT/f'golden-balanced-v1/tree-{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            old.append(z['calibrated_four_ply'])
            st=json.loads(str(z['stats']));old_nodes+=sum(st['leaves_by_depth'])
    names.append('previous_four_ply');scores['previous_four_ply']=np.concatenate(old)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);expert=np.arange(3,16,4)
    base=json.loads((ROOT/'golden-baseline/results.json').read_text())['methods']
    full=np.array(list(base['canonical_raw']['cells'].values()));cellnames=list(base['canonical_raw']['cells'])
    ref=np.array([r['canonical_raw_nll'] for r in rows]);correct=np.array([r['canonical_raw_correct'] for r in rows])
    loss=np.stack([scores[k][:,0] for k in names],axis=1);acc=np.stack([scores[k][:,1] for k in names],axis=1)
    assert np.isfinite(loss).all();delta=loss-ref[:,None];ad=acc-correct[:,None]
    boot=bootstrap_deltas(np.concatenate([delta,ad],axis=1),cells,games)
    draws=full[None,:,None]+boot[:,:,:len(names)]
    points=full[:,None]+np.stack([cellmean(delta[:,i],cells) for i in range(len(names))],axis=1)
    laws=json.loads((ROOT/'training-cm-laws.json').read_text());ci=lambda v:np.quantile(v,[.025,.975]).tolist()
    results={};legal=names.index('legal')
    for i,name in enumerate(names):
        ece=[]
        for c in range(16):
            x=scores[name][cells==c];bins=np.minimum((x[:,2]*15).astype(int),14)
            ece.append(sum(len(a)*abs(float(a[:,1].mean()-a[:,2].mean())) for b in range(15) if len(a:=x[bins==b]))/len(x))
        item=dict(sample_macro_ece=float(np.mean(ece)),sample_expert_ece=float(np.array(ece)[expert].mean()),
            cells={c:dict(ce=float(points[j,i]),ci95=ci(draws[:,j,i]),sample_ece=ece[j],
                delta_vs_legal=float(points[j,i]-points[j,legal]),delta_vs_legal_ci95=ci(draws[:,j,i]-draws[:,j,legal])) for j,c in enumerate(cellnames)})
        budget=plan['methods'].get(name,{}).get('budget',0)
        if name=='previous_four_ply':item.update(mean_nodes=old_nodes/n,expert_mean_nodes=None)
        elif budget:
            b=plan['budgets'].index(budget);cc=cellmean(cost[b],cells)
            item.update(mean_nodes=float(cc.mean()),expert_mean_nodes=float(cc[expert].mean()),
                instrumented_snapshot_seconds=float(np.array(elapsed)[:,b].sum()))
        else:item.update(mean_nodes=0.,expert_mean_nodes=0.)
        for metric,idx in [('macro',np.arange(16)),('expert_macro',expert)]:
            point=float(points[idx,i].mean());samples=draws[:,idx,i].mean(1);lo,hi=ci(samples)
            law=laws['metrics'][metric]['law'];alternative=laws['metrics'][metric]['alternative_shared_floor'];raw=base['official_raw'][metric]
            cm=lambda x:multiplier(law,laws['budget_nd'],raw,float(x))
            item[metric]=point;item[metric+'_ci95']=[lo,hi]
            item[metric+'_training_eq_cm']=cm(point);item[metric+'_cm_ci95']=[cm(hi),cm(lo)]
            item[metric+'_cm_shared_floor_sensitivity']=multiplier(alternative,laws['budget_nd'],raw,point)
            item[metric+'_accuracy']=base['canonical_raw'][metric+'_accuracy']+float(cellmean(ad[:,i],cells)[idx].mean())
            item[metric+'_accuracy_ci95']=ci(base['canonical_raw'][metric+'_accuracy']+boot[:,idx,len(names)+i].mean(1))
            for reference in ('legal','aug_direct','previous_four_ply','old_mcts_reverse'):
                j=names.index(reference);item[metric+'_delta_vs_'+reference]=float((points[idx,i]-points[idx,j]).mean())
                item[metric+'_delta_vs_'+reference+'_ci95']=ci((draws[:,idx,i]-draws[:,idx,j]).mean(1))
                item[metric+'_cm_vs_'+reference]=cm(point)/cm(points[idx,j].mean())
        results[name]=item
    old_result=json.loads((ROOT/'golden-balanced-v1/results.json').read_text())['methods']['calibrated_four_ply']
    for key in ('macro','expert_macro','macro_ci95','expert_macro_ci95','macro_training_eq_cm','expert_macro_training_eq_cm'):
        np.testing.assert_allclose(results['previous_four_ply'][key],old_result[key],atol=1e-12,rtol=0)
    for name,a in results.items():
        a['point_dominated_by']=[other for other,b in results.items() if other!=name and b['mean_nodes']<=a['mean_nodes'] and
            b['macro']<=a['macro'] and b['expert_macro']<=a['expert_macro'] and
            (b['mean_nodes']<a['mean_nodes'] or b['macro']<a['macro'] or b['expert_macro']<a['expert_macro'])]
        a['supported_win_over_four_ply_at_no_more_nodes']=(a['mean_nodes']<=results['previous_four_ply']['mean_nodes'] and
            a['macro_delta_vs_previous_four_ply_ci95'][1]<0 and a['expert_macro_delta_vs_previous_four_ply_ci95'][1]<0)
    report=dict(stage='August-fit-CV-selected golden budget frontier; reused sample, every frozen method reported.',
        methods=results,positions=n,games=len(set(games)),plan_sha256=digest(OUT/'plan.json'),analysis_sha256=digest(Path(__file__)),
        estimator='Known full canonical-raw cell CE + paired sampled difference; equal16/4cell macros, whole-game bootstrap.',
        caveat='CM conditional on transferred training-law shape; intervals exclude law-fit and adaptive repeated-benchmark uncertainty. Point dominance is not statistical dominance.',
        cost_note='Per-root expanded nonterminal nodes requiring model predictions, excluding root prefill; actual prefill/new-token work and wall time separate. Snapshots share a tree; instrumented snapshot time includes earlier backup extraction and is not a standalone benchmark.',
        scoring_seconds=sum(s['seconds'] for s in stats),new_tokens=sum(s['new_tokens'] for s in stats),
        evaluated_leaves=sum(s['evaluated_leaves'] for s in stats),analysis_seconds=time.monotonic()-start)
    atomic(OUT/'results.json',report)
    for name,r in results.items():print(name,*(r[k] for k in ('mean_nodes','macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm')),flush=True)


if __name__=='__main__':main()
