"""Shared paired golden metrics for newly frozen inference experiments."""
import json
import numpy as np
from .balanced_eval import ROOT
from .analyze_balanced import bootstrap_deltas,cellmean
from ..training_cm import multiplier


def summarize(rows,scores,costs,references=('legal','four_ply','soft1000')):
    names=list(scores);n=len(rows);cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);expert=np.arange(3,16,4)
    assert all(x.shape==(n,3) and np.isfinite(x).all() for x in scores.values())
    base=json.loads((ROOT/'golden-baseline/results.json').read_text())['methods'];full=np.array(list(base['canonical_raw']['cells'].values()));cellnames=list(base['canonical_raw']['cells'])
    ref=np.array([r['canonical_raw_nll'] for r in rows]);correct=np.array([r['canonical_raw_correct'] for r in rows])
    delta=np.stack([scores[k][:,0] for k in names],1)-ref[:,None];ad=np.stack([scores[k][:,1] for k in names],1)-correct[:,None]
    boot=bootstrap_deltas(np.concatenate([delta,ad],1),cells,games);draws=full[None,:,None]+boot[:,:,:len(names)]
    points=full[:,None]+np.stack([cellmean(delta[:,i],cells) for i in range(len(names))],1)
    laws=json.loads((ROOT/'training-cm-laws.json').read_text());ci=lambda x:np.quantile(x,[.025,.975]).tolist();results={}
    for i,name in enumerate(names):
        cost=costs[name]
        item=dict(mean_nodes=float(cost) if np.ndim(cost)==0 else float(cellmean(cost,cells).mean()),
            expert_mean_nodes=None if np.ndim(cost)==0 else float(cellmean(cost,cells)[expert].mean()),
            cells={c:dict(ce=float(points[j,i]),ci95=ci(draws[:,j,i])) for j,c in enumerate(cellnames)})
        for metric,idx in [('macro',np.arange(16)),('expert_macro',expert)]:
            law=laws['metrics'][metric]['law'];raw=base['official_raw'][metric];cm=lambda x:multiplier(law,laws['budget_nd'],raw,float(x))
            point=float(points[idx,i].mean());lo,hi=ci(draws[:,idx,i].mean(1));item[metric]=point;item[metric+'_ci95']=[lo,hi]
            item[metric+'_training_eq_cm']=cm(point);item[metric+'_cm_ci95']=[cm(hi),cm(lo)]
            item[metric+'_cm_shared_floor_sensitivity']=multiplier(laws['metrics'][metric]['alternative_shared_floor'],laws['budget_nd'],raw,point)
            item[metric+'_accuracy']=base['canonical_raw'][metric+'_accuracy']+float(cellmean(ad[:,i],cells)[idx].mean())
            item[metric+'_accuracy_ci95']=ci(base['canonical_raw'][metric+'_accuracy']+boot[:,idx,len(names)+i].mean(1))
            for other in references:
                j=names.index(other);diff=(draws[:,idx,i]-draws[:,idx,j]).mean(1)
                item[metric+'_delta_vs_'+other]=float((points[idx,i]-points[idx,j]).mean())
                item[metric+'_delta_vs_'+other+'_ci95']=ci(diff);item[metric+'_cm_vs_'+other]=cm(point)/cm(points[idx,j].mean())
        ece=[]
        for c in range(16):
            x=scores[name][cells==c];bins=np.minimum((x[:,2]*15).astype(int),14)
            ece.append(sum(len(a)*abs(float(a[:,1].mean()-a[:,2].mean())) for b in range(15) if len(a:=x[bins==b]))/len(x))
        item['sample_macro_ece']=float(np.mean(ece));item['sample_expert_ece']=float(np.array(ece)[expert].mean());results[name]=item
    return results


def controls(rows):
    """Original four-ply and prior soft-MCTS controls with their actual scores."""
    n=len(rows);scores={k:[] for k in ('legal','four_ply','soft1000')};nodes=[];four_nodes=0
    for lo in range(0,n,32):
        with np.load(ROOT/f'golden-balanced-v1/tree-{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            scores['four_ply'].append(z['calibrated_four_ply']);four_nodes+=sum(json.loads(str(z['stats']))['leaves_by_depth'])
    for lo in range(0,n,128):
        with np.load(ROOT/f'golden-augcal-v1/{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            scores['legal'].append(z['legal']);scores['soft1000'].append(z['soft0.11000_forward_elo']);nodes.append(z['evaluated_nodes'][-1])
    return {k:np.concatenate(v) for k,v in scores.items()},dict(legal=np.zeros(n),four_ply=four_nodes/n,soft1000=np.concatenate(nodes))
