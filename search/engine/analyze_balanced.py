"""Per-cell paired difference estimates and whole-game uncertainty; no selection."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import training_cm
ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
OUT=ROOT/'golden-balanced-v1'


def cellmean(values,cells,ncells=16):
    return np.bincount(cells,weights=values,minlength=ncells)/np.bincount(cells,minlength=ncells)


def bootstrap_deltas(deltas,cells,games,repeats=2000,seed=919381):
    """One shared game resampling for all methods/cells; never pool cell means."""
    _,ix=np.unique(games,return_inverse=True);g=ix.max()+1;m=deltas.shape[1]
    count=np.zeros((g,16));np.add.at(count,(ix,cells),1.)
    sums=np.zeros((g,16,m));np.add.at(sums,(ix,cells),deltas)
    rng=np.random.default_rng(seed);output=[]
    for lo in range(0,repeats,128):
        weights=rng.multinomial(g,np.full(g,1/g),size=min(128,repeats-lo)).astype(float)
        den=weights@count;assert (den>0).all()
        output.append((weights@sums.reshape(g,16*m)).reshape(-1,16,m)/den[:,:,None])
    return np.concatenate(output)


def main():
    sample=json.loads((OUT/'sample.json').read_text());rows=sample['positions'];n=len(rows)
    baseline=json.loads((ROOT/'golden-baseline/results.json').read_text())['methods']
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);expert=np.arange(3,16,4)
    names=['port_raw','legal','calibrated_direct','calibrated_two_ply','calibrated_four_ply','released_adaptive','repaired_fixed','mcts_batch_legal']
    scores={k:[] for k in names};cost=[]
    for prefix,bs,keys in [('tree',32,names[:5]),('mcts',256,names[5:])]:
        for lo in range(0,n,bs):
            with np.load(OUT/f'{prefix}-{lo:06d}.npz') as z:
                assert list(z['game'])==[r['game'] for r in rows[lo:lo+bs]]
                assert list(z['ply'])==[r['ply'] for r in rows[lo:lo+bs]]
                for k in keys:scores[k].append(z[k])
                cost.append(dict(kind=prefix,**json.loads(str(z['stats']))))
    scores={k:np.concatenate(v) for k,v in scores.items()}
    assert all(v.shape==(n,3) for v in scores.values())
    reference=np.array([r['canonical_raw_nll'] for r in rows])
    refcorrect=np.array([r['canonical_raw_correct'] for r in rows])
    loss=np.stack([scores[k][:,0] for k in names],axis=1)
    correct=np.stack([scores[k][:,1] for k in names],axis=1)
    assert np.isfinite(loss).all()
    canonical=baseline['canonical_raw'];cell_names=list(canonical['cells'])
    full=np.array(list(canonical['cells'].values()))
    loss_delta=loss-reference[:,None];acc_delta=correct-refcorrect[:,None]
    boot=bootstrap_deltas(np.concatenate([loss_delta,acc_delta],axis=1),cells,games)
    boot_loss=full[None,:,None]+boot[:,:,:len(names)]
    methods={};laws=json.loads((ROOT/'training-cm-laws.json').read_text())
    legal_i=names.index('legal');cheap_i=names.index('calibrated_direct')
    anchored=np.stack([full+cellmean(loss_delta[:,i],cells) for i in range(len(names))],axis=1)
    ci=lambda x:np.quantile(x,[.025,.975]).tolist()
    for i,name in enumerate(names):
        point=anchored[:,i];draws=boot_loss[:,:,i]
        ordinary=cellmean(loss[:,i],cells)
        ece=[]
        for c in range(16):
            a=scores[name][cells==c];bins=np.minimum((a[:,2]*15).astype(int),14)
            gap=0.
            for b in range(15):
                x=a[bins==b]
                if len(x):gap+=len(x)*abs(float(x[:,1].mean()-x[:,2].mean()))
            ece.append(gap/len(a))
        item=dict(macro=float(point.mean()),expert_macro=float(point[expert].mean()),
                  ordinary_sample_macro=float(ordinary.mean()),ordinary_sample_expert=float(ordinary[expert].mean()),
                  macro_ci95=ci(draws.mean(1)),expert_macro_ci95=ci(draws[:,expert].mean(1)),
                  macro_accuracy=canonical['macro_accuracy']+float(cellmean(acc_delta[:,i],cells).mean()),
                  expert_macro_accuracy=canonical['expert_macro_accuracy']+float(cellmean(acc_delta[:,i],cells)[expert].mean()),
                  sample_macro_ece=float(np.mean(ece)),sample_expert_ece=float(np.mean(np.array(ece)[expert])),
                  cells={c:dict(ce=float(point[j]),ci95=ci(draws[:,j]),sample_ce=float(ordinary[j]),sample_ece=float(ece[j]),
                                  delta_vs_legal=float(point[j]-anchored[j,legal_i]),
                                  delta_vs_legal_ci95=ci(draws[:,j]-boot_loss[:,j,legal_i])) for j,c in enumerate(cell_names)})
        for metric,idx in [('macro',np.arange(16)),('expert_macro',expert)]:
            law=laws['metrics'][metric]['law'];alternative=laws['metrics'][metric]['alternative_shared_floor'];c=laws['budget_nd'];raw=baseline['official_raw'][metric]
            point_ce=float(point[idx].mean());sample_ce=draws[:,idx].mean(1)
            cm=lambda l,x:training_cm.multiplier(l,c,raw,float(x))
            lo,hi=ci(sample_ce)
            total=cm(law,point_ce);baselegal=cm(law,float(anchored[idx,legal_i].mean()));basecheap=cm(law,float(anchored[idx,cheap_i].mean()))
            item[metric+'_training_eq_cm']=total
            item[metric+'_cm_ci95']=[cm(law,hi),cm(law,lo)]
            item[metric+'_cm_shared_floor_sensitivity']=cm(alternative,point_ce)
            item[metric+'_cm_vs_legal']=total/baselegal
            item[metric+'_cm_vs_calibrated_direct']=total/basecheap
            # Differences remain paired, rather than subtracting marginal CIs.
            item[metric+'_delta_vs_calibrated_direct']=float((point-anchored[:,cheap_i])[idx].mean())
            item[metric+'_delta_vs_calibrated_direct_ci95']=ci((draws-boot_loss[:,:,cheap_i])[:,idx].mean(1))
        methods[name]=item
    report=dict(stage='Preregistered balanced golden confirmation; all methods reported without selection',
                positions=n,games=len(set(games)),counts=np.bincount(cells,minlength=16).tolist(),
                estimator='Known full canonical-raw cell mean + sample mean paired difference; then equal-cell macro.',
                caveat='Training-equivalent CM is conditional on the transferred, vertically anchored law. Bootstrap covers sampled games, not law-fit uncertainty. Runtime/port drift is included and reported separately.',
                original_full_baselines=baseline,methods=methods,
                timing=dict(summed_four_ply_block_seconds=sum(c['seconds'] for c in cost if c['kind']=='tree'),
                            summed_two_MCTS_block_seconds=sum(c['seconds'] for c in cost if c['kind']=='mcts'),
                            note='Two-ply and cheap policies reuse four-ply/root predictions; standalone costs are measured separately on development data.'),
                runtime=json.loads((OUT/'runtime.json').read_text()),execution=json.loads((OUT/'execution.json').read_text()))
    tmp=OUT/'results.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'results.json')
    for k,v in methods.items():print(k,*(round(v[x],6) for x in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm')),flush=True)


if __name__=='__main__':main()
