"""Frozen August-selected utility normalization on cached golden search trees."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax, logsumexp
from .balanced_eval import ROOT, atomic, digest
from .golden_metrics import controls, summarize

OUT=ROOT/'golden-utilities-v1'


def freeze():
    OUT.mkdir(exist_ok=True);dev=ROOT/'aug-selection-v1/utilities.json';d=json.loads(dev.read_text())
    assert set(d['fit_cv_selected'].values())=={'standardized10','standardized20'}
    plan=dict(methods={k:dict(floor=f,parameters=d['results'][k]['parameters']) for k,f in [('standardized10',.1),('standardized20',.2)]},
        dev_sha256=digest(dev),parent_sha256=digest(ROOT/'golden-explore-v1/plan.json'),
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),laws_sha256=digest(ROOT/'training-cm-laws.json'),
        sources={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('golden_metrics.py'),Path(__file__).with_name('analyze_utilities.py'))},
        stage='Both August fit-gameCV-selected normalization floors reported. Identical cp2.5 1000sim trees, original priors and node count; CPU postprocessing only. No golden selection between arms; reused sample.')
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic(p,plan)
    return plan


def main():
    plan=freeze();start=time.monotonic();rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    scores,costs=controls(rows);records={k:[] for k in [*plan['methods'],'cp25_fixed']};nodes=[];transform_seconds=0.
    for lo in range(0,len(rows),512):
        with np.load(ROOT/f'golden-explore-v1/{lo:06d}.npz') as z:
            part=rows[lo:lo+len(z['game'])];assert list(z['game'])==[r['game'] for r in part]
            ar=np.arange(len(part));groups=np.array([r['cell']%4 for r in part]);target=np.array([r['legal'].index(r['target']) for r in part])
            ids=z['ids'];mask=z['mask'];q=z['q'][-1];logits=z['z'][:,378:2346][ar[:,None],ids].astype(float)
            records['cp25_fixed'].append(z['cp25_fixed']);nodes.append(z['cp25_fixed_nodes'])
        begin=time.monotonic();p=softmax(np.where(mask,logits,-np.inf),axis=1)
        mu=(p*q).sum(1,keepdims=True);sd=np.sqrt((p*(q-mu)**2).sum(1,keepdims=True))
        for name,spec in plan['methods'].items():
            theta=np.array(spec['parameters']['theta'])[groups]
            u=(q-mu)/np.maximum(sd,spec['floor'])
            a=np.where(mask,theta[:,0,None]*logits+theta[:,1,None]*u,-np.inf)
            lp=a-logsumexp(a,axis=1,keepdims=True);prob=np.exp(lp)
            pred=np.where(mask&(prob==prob.max(1)[:,None]),ids,1968).min(1)
            records[name].append(np.stack([-lp[ar,target],pred==ids[ar,target],prob.max(1)],1))
        transform_seconds+=time.monotonic()-begin
    for name,sc in records.items():scores[name]=np.concatenate(sc);costs[name]=np.concatenate(nodes)
    results=summarize(rows,scores,costs,references=('legal','four_ply','cp25_fixed'))
    previous=json.loads((ROOT/'golden-explore-v1/results.json').read_text())['methods']['cp25_fixed']
    for key in ('macro','expert_macro'):np.testing.assert_allclose(results['cp25_fixed'][key],previous[key],atol=1e-12,rtol=0)
    atomic(OUT/'results.json',dict(methods=results,plan_sha256=digest(OUT/'plan.json'),analysis_seconds=time.monotonic()-start,
        transform_seconds=transform_seconds,additional_gpu_queries=0,caveat=plan['stage']+' Training-equivalent CM is conditional on the frozen law; game CIs exclude law and benchmark-reuse uncertainty.'))
    for name in plan['methods']:
        r=results[name];print(name,{k:r[k] for k in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm','macro_delta_vs_cp25_fixed_ci95','expert_macro_delta_vs_cp25_fixed_ci95')},flush=True)


if __name__=='__main__':main()
