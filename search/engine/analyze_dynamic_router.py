"""Paired golden analysis for actually executed mixed-budget search."""
import json
import time
import numpy as np
from .balanced_eval import ROOT,atomic,digest
from .analyze_balanced import bootstrap_deltas,cellmean
from ..training_cm import multiplier
from .fit_policy import loss_gradient

OUT=ROOT/'golden-dynamic-router-v1'


def main():
    start=time.monotonic();plan=json.loads((OUT/'plan.json').read_text());assert plan['sample_sha256']==digest(ROOT/'golden-balanced-v1/sample.json')
    assert plan['laws_sha256']==digest(ROOT/'training-cm-laws.json')
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];n=len(rows);scores=[];nodes=[];stats=[];live_policies=[]
    for lo in range(0,n,plan['roots_per_block']):
        with np.load(OUT/f'{lo:06d}.npz') as z:
            assert list(z['game'])==[r['game'] for r in rows[lo:lo+len(z['game'])]]
            np.testing.assert_array_equal(z['ply'],[r['ply'] for r in rows[lo:lo+len(z['game'])]])
            scores.append(z['score']);nodes.append(z['nodes']);stats.append(json.loads(str(z['stats'])))
            live_policies.extend([p[m] for p,m in zip(z['policy'],z['mask'])])
    score=np.concatenate(scores);nodes=np.concatenate(nodes);four=[];legal=[];fixed=[]
    for lo in range(0,n,32):
        with np.load(ROOT/f'golden-balanced-v1/tree-{lo:06d}.npz') as z:four.append(z['calibrated_four_ply'])
    policy_differences=[]
    for lo in range(0,n,128):
        with np.load(ROOT/f'golden-augcal-v1/{lo:06d}.npz') as z:
            legal.append(z['legal']);fixed.append(z['soft0.11000_forward_elo'])
            part=rows[lo:lo+len(z['game'])];ar=np.arange(len(part));groups=np.array([r['cell']%4 for r in part])
            target=np.array([r['legal'].index(r['target']) for r in part]);mask=z['mask'];ids=z['ids'];logits=z['root'][:,378:2346][ar[:,None],ids].astype(float)
            p=np.zeros_like(logits)
            for g in np.unique(groups):
                select=groups==g;budget=plan['method']['budgets_by_elo'][g];q=z['q'][[16,64,256,1000].index(budget),3].astype(float)
                f=plan['method']['parameters'][str(int(g))]
                p[select],_=loss_gradient([f['alpha'],f['beta']],logits[select],q[select],mask[select],target[select],'forward',return_policy=True)
            for i,(row,m) in enumerate(zip(p,mask)):
                cached=row[m];live=live_policies[lo+i];np.testing.assert_allclose(cached.sum(),1,atol=1e-12);np.testing.assert_allclose(live.sum(),1,atol=1e-12)
                policy_differences.append((float(np.abs(cached-live).max()),float(np.abs(cached-live).sum()),float(np.sum(cached*np.log(cached/live)))))
    scores={'dynamic':score,'four_ply':np.concatenate(four),'legal':np.concatenate(legal),'fixed1000':np.concatenate(fixed)};names=list(scores)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);expert=np.arange(3,16,4)
    base=json.loads((ROOT/'golden-baseline/results.json').read_text())['methods'];full=np.array(list(base['canonical_raw']['cells'].values()));cellnames=list(base['canonical_raw']['cells'])
    ref=np.array([r['canonical_raw_nll'] for r in rows]);correct=np.array([r['canonical_raw_correct'] for r in rows]);delta=np.stack([scores[k][:,0] for k in names],1)-ref[:,None]
    ad=np.stack([scores[k][:,1] for k in names],1)-correct[:,None]
    boot=bootstrap_deltas(np.concatenate([delta,ad],1),cells,games);draws=full[None,:,None]+boot[:,:,:len(names)];points=full[:,None]+np.stack([cellmean(delta[:,i],cells) for i in range(len(names))],1)
    laws=json.loads((ROOT/'training-cm-laws.json').read_text());ci=lambda x:np.quantile(x,[.025,.975]).tolist();results={}
    old=json.loads((ROOT/'golden-router-v1/results.json').read_text())['methods']
    for i,name in enumerate(names):
        item=dict(mean_nodes=float(cellmean(nodes,cells).mean()) if name=='dynamic' else old[{'four_ply':'old_four_ply','legal':'legal','fixed1000':'old_soft1000'}[name]]['mean_nodes'],
            expert_mean_nodes=float(cellmean(nodes,cells)[expert].mean()) if name=='dynamic' else old[{'four_ply':'old_four_ply','legal':'legal','fixed1000':'old_soft1000'}[name]]['expert_mean_nodes'],
            cells={c:dict(ce=float(points[j,i]),ci95=ci(draws[:,j,i])) for j,c in enumerate(cellnames)})
        for metric,idx in [('macro',np.arange(16)),('expert_macro',expert)]:
            law=laws['metrics'][metric]['law'];raw=base['official_raw'][metric];cm=lambda x:multiplier(law,laws['budget_nd'],raw,float(x))
            point=float(points[idx,i].mean());lo,hi=ci(draws[:,idx,i].mean(1));item[metric]=point;item[metric+'_ci95']=[lo,hi]
            item[metric+'_training_eq_cm']=cm(point);item[metric+'_cm_ci95']=[cm(hi),cm(lo)]
            item[metric+'_cm_shared_floor_sensitivity']=multiplier(laws['metrics'][metric]['alternative_shared_floor'],laws['budget_nd'],raw,point)
            item[metric+'_accuracy']=base['canonical_raw'][metric+'_accuracy']+float(cellmean(ad[:,i],cells)[idx].mean())
            item[metric+'_accuracy_ci95']=ci(base['canonical_raw'][metric+'_accuracy']+boot[:,idx,len(names)+i].mean(1))
            for other in ('legal','four_ply','fixed1000'):
                j=names.index(other);diff=(draws[:,idx,i]-draws[:,idx,j]).mean(1)
                item[metric+'_delta_vs_'+other]=float((points[idx,i]-points[idx,j]).mean())
                item[metric+'_delta_vs_'+other+'_ci95']=ci(diff)
                # Conservative simultaneous upper bound for four router arms x
                # two quality metrics; still not a correction for all prior research.
                item[metric+'_delta_vs_'+other+'_family8_upper']=float(np.quantile(diff,1-.05/8))
                item[metric+'_cm_vs_'+other]=cm(point)/cm(points[idx,j].mean())
        ece=[]
        for c in range(16):
            x=scores[name][cells==c];bins=np.minimum((x[:,2]*15).astype(int),14)
            ece.append(sum(len(a)*abs(float(a[:,1].mean()-a[:,2].mean())) for b in range(15) if len(a:=x[bins==b]))/len(x))
        item['sample_macro_ece']=float(np.mean(ece));item['sample_expert_ece']=float(np.array(ece)[expert].mean());results[name]=item
    for name,other in [('four_ply','old_four_ply'),('fixed1000','old_soft1000'),('legal','legal')]:
        for key in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm'):np.testing.assert_allclose(results[name][key],old[other][key],atol=1e-12,rtol=0)
    r=results['dynamic'];r['supported_frontier_win_vs_four_ply']=(r['mean_nodes']<=results['four_ply']['mean_nodes'] and r['macro_delta_vs_four_ply_family8_upper']<0 and r['expert_macro_delta_vs_four_ply_family8_upper']<0)
    report=dict(methods=results,positions=n,plan_sha256=digest(OUT/'plan.json'),
        scoring_seconds=sum(s['seconds'] for s in stats),forward_seconds=sum(s['forward_seconds'] for s in stats),new_tokens=sum(s['new_tokens'] for s in stats),
        unique_nonroot_nn_requests=sum(s['unique_nonroot_nn_requests'] for s in stats),
        logical_nonroot_nn_requests=int(nodes.sum()),
        cached_live_policy_difference=dict(max_per_position_max_abs=float(np.max(np.array(policy_differences)[:,0])),
            max_per_position_l1=float(np.max(np.array(policy_differences)[:,1])),
            mean_kl=float(np.mean(np.array(policy_differences)[:,2])),max_kl=float(np.max(np.array(policy_differences)[:,2]))),
        analysis_seconds=time.monotonic()-start,caveat='Reused golden sample, no retuning. CM is conditional training equivalence; bootstrap excludes law uncertainty and all prior adaptive benchmark use. Family8 bound covers four routers x two metrics only.')
    atomic(OUT/'results.json',report);print('Dynamic router result',r,'timing',report['scoring_seconds'],report['analysis_seconds'],flush=True)


if __name__=='__main__':main()
