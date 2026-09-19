"""Frozen residual temperature after the selected conditional strength mixture."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT, atomic
from .balanced_eval import digest
from .temperature_stack import evaluate
from .golden_metrics import summarize


def main():
    out = ROOT/'golden-temperature-stack-v1'; out.mkdir(exist_ok=True)
    dev = ROOT/'aug-temperature-stack-v1/results.json'
    d = json.loads(dev.read_text()); assert set(d['fit_cv_selected'].values()) == {'state0.01'}
    parent = ROOT/'golden-conditional-mixture-v1'
    tree = ROOT/'golden-expanded-stack-v2'
    plan = dict(parameters=d['results']['state0.01']['parameters'][0], dev_sha256=digest(dev),
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'), laws_sha256=digest(ROOT/'training-cm-laws.json'),
        parent_scores_sha256=digest(parent/'scores.npz'), parent_results_sha256=digest(parent/'results.json'),
        input_blocks={p.name:digest(p) for p in sorted(tree.glob('[0-9]*.npz'))},
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('temperature_stack.py'),Path(__file__).with_name('golden_metrics.py')]},
        selection='Both expanded-August fit-gameCV metrics selected state0.01; fit coefficients frozen before golden. Parent is all0.001 conditional strength mixture, including pre-move clock. Temperature itself uses only prior entropy, Q spread, ply and predicted think time.',
        semantics='Output-only transform on exactly the cached parent policy and identical NN work; no extra neural evaluations. Reused golden, exploratory comparison. CM conditional on frozen anchored training law; final success requires fresh disjoint games.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    start=time.monotonic();rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    with np.load(parent/'scores.npz') as f:
        np.testing.assert_array_equal(f['game'],[r['game'] for r in rows]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows])
        scores={name:f['scores'][list(f['names']).index(name)] for name in ('port_raw','new_legal','all0.001')}
        nodes=f['nodes'];offsets=f['policy_offsets'];values=f['all0.001_policy']
    output=[];policies=[];eta_values=[]
    params=plan['parameters']
    for path in sorted(tree.glob('[0-9]*.npz')):
        assert digest(path)==plan['input_blocks'][path.name]
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);part=rows[lo:hi];n=len(part);ar=np.arange(n)
            np.testing.assert_array_equal(f['game'],[r['game'] for r in part]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in part])
            ids,mask,root,q=f['ids'],f['mask'],f['z'],f['subtree_sigma10_q']
            p=np.zeros(mask.shape);p[mask]=values[offsets[lo]:offsets[hi]]
            np.testing.assert_allclose(p.sum(1),1.,atol=1e-14)
            target=np.array([r['legal'].index(r['target']) for r in part])
            np.testing.assert_allclose(-np.log(p[ar,target]),scores['all0.001'][lo:hi,0],atol=1e-14)
            z=np.where(mask,root[:,378:2346][ar[:,None],ids].astype(float),0.)
            prior=softmax(np.where(mask,z,-np.inf),axis=1);qm=(prior*q).sum(1)
            entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
            qstd=np.sqrt((prior*(q-qm[:,None])**2).sum(1))
            tp=softmax(root[:,2350:2413].astype(float),axis=1);tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
            feat=np.column_stack([entropy,np.log(.01+qstd),[len(r['prefix'])-11 for r in part],np.log1p(tp@tc)])
            assert params['fields']==[0,1,2,3]
            x=np.c_[np.ones(n),np.clip((feat-params['mean'])/params['scale'],-3,3)]
            logp=np.where(mask,np.log(np.maximum(p,1e-300)),0.)
            new=evaluate(np.array(params['theta']),x,logp,mask,target)
            eta_values.extend(np.clip(1+x@np.array(params['theta']),.5,2.).tolist())
            chosen=np.where(new==new.max(1)[:,None],ids,1968).min(1)
            output.append(np.stack([-np.log(new[ar,target]),chosen==ids[ar,target],new.max(1)],axis=1));policies.append(new[mask])
    scores['temperature']=np.concatenate(output)
    costs={k:nodes if k in ('all0.001','temperature') else np.zeros(len(rows)) for k in scores}
    result=summarize(rows,scores,costs,references=('new_legal','all0.001'))
    old=json.loads((parent/'results.json').read_text())
    for key in ('macro','expert_macro'):
        np.testing.assert_allclose(result['all0.001'][key],old['methods']['all0.001'][key],atol=1e-12,rtol=0)
    atomic(out/'results.json',dict(methods=result,positions=len(rows),plan_sha256=digest(pp),analysis_seconds=time.monotonic()-start,
        inverse_temperature_quantiles=np.quantile(eta_values,[0,.1,.5,.9,1]).tolist(),caveat=plan['semantics']))
    with (out/'scores.npz').open('wb') as f:
        np.savez_compressed(f,names=list(scores),scores=np.stack(list(scores.values())),nodes=nodes,policy=np.concatenate(policies),policy_offsets=offsets,
            game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    print('temperature',{k:result['temperature'][k] for k in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm','mean_nodes')},flush=True)


if __name__=='__main__':main()
