"""Frozen development budget routers, scored on cached golden tree snapshots."""
import json
from pathlib import Path
import time
import numpy as np
from .balanced_eval import ROOT,atomic,digest
from .analyze_balanced import bootstrap_deltas,cellmean
from .fit_policy import loss_gradient
from ..training_cm import multiplier

OUT=ROOT/'golden-router-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    dev=ROOT/'aug-search-v1/budget-router.json';data=json.loads(dev.read_text())
    plan=dict(methods={k:data['results'][k] for k in data['golden_preregistered']},
        dev_sha256=digest(dev),sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),
        parent_plan_sha256=digest(ROOT/'golden-augcal-v1/plan.json'),
        laws_sha256=digest(ROOT/'training-cm-laws.json'),sources={p.name:digest(p) for p in
            (Path(__file__),Path(__file__).with_name('budget_router.py'),Path(__file__).with_name('fit_policy.py'))},
        selection='All four predeclared development-fit-CV budget assignments, no golden selection. Output is recalibrated soft.1/forward/Elo at each budget.',
        cost_note='Cached counterfactual stopping cost on the same explored trees. Actual dynamic-batched runtime has not been benchmarked. Root prefill is common and excluded from expanded-node count.')
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:atomic(path,plan)
    return plan


def main():
    plan=freeze();start=time.monotonic()
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];n=len(rows)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);expert=np.arange(3,16,4)
    scores={k:[] for k in ['legal','old_soft1000','old_four_ply',*plan['methods']]};cost={k:[] for k in scores}
    for lo in range(0,n,128):
        part=rows[lo:lo+128];ar=np.arange(len(part));target=np.array([r['legal'].index(r['target']) for r in part])
        with np.load(ROOT/f'golden-augcal-v1/{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in part];np.testing.assert_array_equal(z['ply'],[r['ply'] for r in part])
            ids=z['ids'];mask=z['mask'];logits=z['root'][:,378:2346][ar[:,None],ids].astype(float)
            for key,field in [('legal','legal'),('old_soft1000','soft0.11000_forward_elo')]:
                scores[key].append(z[field]);cost[key].append(z['evaluated_nodes'][-1] if key=='old_soft1000' else np.zeros(len(part)))
            groups=[]
            for r in part:
                p=r['prefix'];offset=3 if (len(p)-11)%2==0 else 7
                elo=sum(p[offset+j]*10**(3-j) for j in range(4));groups.append(np.searchsorted([1400,2000,2400],elo,side='right'))
            groups=np.array(groups);np.testing.assert_array_equal(groups,cells[lo:lo+len(part)]%4)
            for name,method in plan['methods'].items():
                p=np.zeros_like(logits);nodes=np.zeros(len(part))
                for g in np.unique(groups):
                    selected=groups==g;budget=method['budgets_by_elo'][g];q=np.zeros_like(logits)
                    if budget:
                        b=[16,64,256,1000].index(budget);q=z['q'][b,3].astype(float);nodes[selected]=z['evaluated_nodes'][b,selected]
                    f=method['parameters'][str(int(g))]
                    p[selected],_=loss_gradient([f['alpha'],f['beta']],logits[selected],q[selected],mask[selected],target[selected],'forward',return_policy=True)
                prediction=np.where(mask&(p==p.max(1)[:,None]),ids,1968).min(1)
                scores[name].append(np.stack([-np.log(p[ar,target]),prediction==ids[ar,target],p.max(1)],axis=1));cost[name].append(nodes)
    old_total=0
    for lo in range(0,n,32):
        with np.load(ROOT/f'golden-balanced-v1/tree-{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            scores['old_four_ply'].append(z['calibrated_four_ply']);old_total+=sum(json.loads(str(z['stats']))['leaves_by_depth'])
    scores={k:np.concatenate(v) for k,v in scores.items()};cost={k:np.concatenate(v) for k,v in cost.items() if v}
    names=list(scores);base=json.loads((ROOT/'golden-baseline/results.json').read_text())['methods']
    full=np.array(list(base['canonical_raw']['cells'].values()));cellnames=list(base['canonical_raw']['cells'])
    ref=np.array([r['canonical_raw_nll'] for r in rows]);correct=np.array([r['canonical_raw_correct'] for r in rows])
    losses=np.stack([scores[k][:,0] for k in names],1);ad=np.stack([scores[k][:,1] for k in names],1)-correct[:,None]
    delta=losses-ref[:,None];boot=bootstrap_deltas(np.concatenate([delta,ad],1),cells,games)
    draws=full[None,:,None]+boot[:,:,:len(names)];points=full[:,None]+np.stack([cellmean(delta[:,i],cells) for i in range(len(names))],1)
    laws=json.loads((ROOT/'training-cm-laws.json').read_text());ci=lambda v:np.quantile(v,[.025,.975]).tolist();results={}
    for i,name in enumerate(names):
        item=dict(mean_nodes=old_total/n if name=='old_four_ply' else float(cellmean(cost[name],cells).mean()),
                  expert_mean_nodes=None if name=='old_four_ply' else float(cellmean(cost[name],cells)[expert].mean()),
                  cells={c:dict(ce=float(points[j,i]),ci95=ci(draws[:,j,i])) for j,c in enumerate(cellnames)})
        for metric,idx in [('macro',np.arange(16)),('expert_macro',expert)]:
            law=laws['metrics'][metric]['law'];raw=base['official_raw'][metric]
            cm=lambda x:multiplier(law,laws['budget_nd'],raw,float(x))
            point=float(points[idx,i].mean());lo,hi=ci(draws[:,idx,i].mean(1))
            item[metric]=point;item[metric+'_ci95']=[lo,hi];item[metric+'_training_eq_cm']=cm(point);item[metric+'_cm_ci95']=[cm(hi),cm(lo)]
            item[metric+'_cm_shared_floor_sensitivity']=multiplier(laws['metrics'][metric]['alternative_shared_floor'],laws['budget_nd'],raw,point)
            item[metric+'_accuracy']=base['canonical_raw'][metric+'_accuracy']+float(cellmean(ad[:,i],cells)[idx].mean())
            item[metric+'_accuracy_ci95']=ci(base['canonical_raw'][metric+'_accuracy']+boot[:,idx,len(names)+i].mean(1))
            for reference in ('legal','old_four_ply','old_soft1000'):
                j=names.index(reference);item[metric+'_delta_vs_'+reference]=float((points[idx,i]-points[idx,j]).mean())
                item[metric+'_delta_vs_'+reference+'_ci95']=ci((draws[:,idx,i]-draws[:,idx,j]).mean(1))
                item[metric+'_cm_vs_'+reference]=cm(point)/cm(points[idx,j].mean())
        ece=[]
        for c in range(16):
            x=scores[name][cells==c];bins=np.minimum((x[:,2]*15).astype(int),14)
            ece.append(sum(len(a)*abs(float(a[:,1].mean()-a[:,2].mean())) for b in range(15) if len(a:=x[bins==b]))/len(x))
        item['sample_macro_ece']=float(np.mean(ece));item['sample_expert_ece']=float(np.array(ece)[expert].mean());results[name]=item
    old=json.loads((ROOT/'golden-augcal-v1/results.json').read_text())['methods']
    for name,reference in [('old_four_ply','previous_four_ply'),('old_soft1000','soft0.11000_forward_elo'),('legal','legal')]:
        for key in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm'):
            np.testing.assert_allclose(results[name][key],old[reference][key],atol=1e-12,rtol=0)
    for name,r in results.items():
        r['supported_win_over_four_ply_at_no_more_nodes']=(r['mean_nodes']<=results['old_four_ply']['mean_nodes'] and
            r['macro_delta_vs_old_four_ply_ci95'][1]<0 and r['expert_macro_delta_vs_old_four_ply_ci95'][1]<0)
    report=dict(stage='Frozen development allocation scored on reused golden sample; all preregistered policies reported.',
        methods=results,plan_sha256=digest(OUT/'plan.json'),positions=n,analysis_seconds=time.monotonic()-start,
        caveat='CM conditional on shifted training-law shape. Bootstrap is by whole game; intervals exclude law-fit and adaptive repeated-benchmark uncertainty.',
        cost_note=plan['cost_note'])
    atomic(OUT/'results.json',report)
    for name,r in results.items():print(name,*(r[k] for k in ('mean_nodes','macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm')),flush=True)


if __name__=='__main__':main()
