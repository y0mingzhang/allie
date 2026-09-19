"""Frozen adaptive-router repeat/permutation audit. No method selection."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .growforest_native import load
from .diff_backup_native import load as backup_load
from .handles import HandleOracle
from .budget_policy import policy,route
from .value_of_compute import features
from .golden_value_of_compute import clocks,metrics
from .golden_metrics import summarize
from .analyze_august import means

OUT=ROOT/'golden-order-audit-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'golden-value-of-compute-v1';base=json.loads((source/'plan.json').read_text())
    for filename,sha in base['sources'].items():assert digest(Path(__file__).with_name(filename))==sha,(filename,'changed')
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];n=len(rows)
    assert digest(ROOT/'golden-balanced-v1/sample.json')==base['sample_sha256']
    plan=dict(base_plan_sha256=digest(source/'plan.json'),source_sha256=digest(Path(__file__)),base_sources=base['sources'],
        reference_scores_sha256=digest(source/'scores.npz'),reference_blocks={p.name:digest(p) for p in sorted((source/'state0.1').glob('[0-9]*.npz'))},
        router=base['methods']['state0.1'],budget_policies=base['budget_policies'],budgets=base['budgets'],
        sidecars=base['sidecars'],sample_sha256=base['sample_sha256'],laws_sha256=base['laws_sha256'],
        roots_per_block=1024,threads=4,initial_simulations=128,skip_forced=True,
        orders=dict(repeat=list(range(n)),reverse=list(range(n-1,-1,-1)),shuffle=np.random.default_rng(983192).permutation(n).tolist()),
        purpose='Audit same frozen state0.1 router: same-order repeat and two predeclared fixed permutations. No calibration refitting, no selection. Compare actual per-position policy, budget decisions and macro/expert CE to the live original. Different query batch composition can change BF16 numerics and discrete expansion; do not assume equivalence.',
        threshold='If absolute macro/expert CE drift >.001 for any permutation, final claim requires a batch-invariant final evaluator. Individual policy divergence also reported. This is an audit threshold, not an acceptance guarantee.')
    pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return plan

def run(oracle,spec):
    plan=freeze();start=time.monotonic();module,backup=load(),backup_load()
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    seconds=clocks(rows,plan)
    for name,order in plan['orders'].items():
        config=plan['router'];ordered=[rows[i] for i in order]
        folder=OUT/name;folder.mkdir(exist_ok=True)
        for lo in range(0,len(rows),plan['roots_per_block']):
            path=folder/f'{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            position_index=np.array(order[lo:lo+plan['roots_per_block']]);part=ordered[lo:lo+plan['roots_per_block']];n=len(part);ar=np.arange(n);cells=np.array([r['cell'] for r in part])
            forced=np.array([len(r['legal'])==1 for r in part]);sec=seconds[position_index]
            initial=np.where(forced,0,128).tolist()
            assert 1000*n+sum(len(r['prefix']) for r in part)<oracle.runner.max_total_num_tokens
            begin=time.monotonic();oracle.reset()
            bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits.astype(float)
            tree=module.Tree([r['prefix'] for r in part],z,initial,[2.5]*n,plan['threads'])
            def advance():
                while not tree.done:
                    h=tree.select()
                    if len(h):tree.update(bridge(h))
            advance()
            k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),np.int32);mask=np.zeros((n,k),bool)
            for i,r in enumerate(part):
                ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
            target=np.array([r['legal'].index(r['target']) for r in part])
            if name=='fixed512':index=np.full(n,2)
            elif config['kind']=='elo':index=route(cells,None,config,plan['budgets'])
            else:
                compact=tree.compact()
                q128=backup.Backup(compact,128).reduce(np.log(.2),-.5,ids)[0]
                del compact
                p128=policy(part,z,q128,ids,mask,sec,plan['budget_policies']['128'])
                f=features(part,z,q128,p128,mask,ids,sec)
                index=route(cells,f,config,plan['budgets'])
            budgets=np.asarray(plan['budgets'])[index];budgets[forced]=0
            tree.grow(budgets.tolist());advance()
            compact=tree.compact()
            q=backup.Backup(compact,1000).reduce(np.log(.2),-.5,ids)[0]
            nodes=np.array(tree.evals);stats=tree.stats()
            assert int(nodes.sum())==bridge.queries
            del tree,compact
            p=np.zeros(mask.shape)
            for bi,b in enumerate(plan['budgets']):
                take=index==bi
                if not take.any():continue
                p[take]=policy([part[i] for i in np.flatnonzero(take)],z[take],q[take],ids[take],mask[take],sec[take],plan['budget_policies'][str(b)])
            logits=z[:,378:2346][ar[:,None],ids]
            raw=metrics(softmax(z[:,378:2346],axis=1),np.broadcast_to(np.arange(1968),(n,1968)),np.array([r['target']-378 for r in part]))
            legal=metrics(softmax(np.where(mask,logits,-np.inf),axis=1),ids,target)
            stats.update(seconds=time.monotonic()-begin,forward_seconds=oracle.forward_seconds,new_tokens=oracle.new_tokens,
                unique_nonroot_nn_requests=bridge.queries,nominal_mean=float(budgets.mean()))
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez_compressed(f,scores=metrics(p,ids,target),raw=raw,legal=legal,policy=p,ids=ids,mask=mask,z=z,q=q,
                    position_index=position_index,nodes=nodes,budgets=budgets,stats=json.dumps(stats),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
            tmp.replace(path);print(name,lo+n,'/',len(rows),stats['seconds'],float(nodes.mean()),flush=True)
    return analyze(time.monotonic()-start)


def analyze(elapsed=None):
    start=time.monotonic();plan=freeze()
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions'];n=len(rows);cells=np.array([r['cell'] for r in rows])
    k=max(len(r['legal']) for r in rows);mask=np.arange(k)[None,:]<np.array([len(r['legal']) for r in rows])[:,None]
    folders={'original':ROOT/'golden-value-of-compute-v1/state0.1',**{key:OUT/key for key in plan['orders']}}
    scores,costs,policies,budgets,times={},{},{},{},{}
    for name,folder in folders.items():
        sc=np.zeros((n,3));nodes=np.zeros(n);pp=np.zeros((n,k));bb=np.zeros(n,int);seen=np.zeros(n,bool);timing=0.
        for path in sorted(folder.glob('[0-9]*.npz')):
            if name=='original':assert digest(path)==plan['reference_blocks'][path.name]
            with np.load(path) as f:
                ix=f['position_index'] if name!='original' else np.arange(int(path.stem),int(path.stem)+len(f['game']))
                assert not seen[ix].any();seen[ix]=True
                np.testing.assert_array_equal(f['game'],[rows[i]['game'] for i in ix]);np.testing.assert_array_equal(f['ply'],[rows[i]['ply'] for i in ix])
                for j,i in enumerate(ix):np.testing.assert_array_equal(f['ids'][j][f['mask'][j]],np.array(rows[i]['legal'])-378)
                sc[ix]=f['scores'];nodes[ix]=f['nodes'];pp[ix,:f['policy'].shape[1]]=f['policy'];bb[ix]=f['budgets']
                timing+=json.loads(str(f['stats']))['seconds']
        assert seen.all();assert np.isfinite(pp).all() and (pp[mask]>0).all()
        np.testing.assert_allclose(pp.sum(1),1.,atol=1e-13)
        scores[name]=sc;costs[name]=nodes;policies[name]=pp;budgets[name]=bb;times[name]=timing
    result=summarize(rows,scores,costs,references=('original',))
    checks={};base=policies['original']
    for name in plan['orders']:
        p=policies[name];diff=np.abs(p-base)
        kl=(base*(np.log(np.maximum(base,1e-300))-np.log(np.maximum(p,1e-300)))).sum(1)
        delta=scores[name][:,0]-scores['original'][:,0];dc=means(delta,cells)
        checks[name]=dict(macro_ce_delta=float(dc.mean()),expert_ce_delta=float(dc[3::4].mean()),
            max_policy_abs=float(diff.max()),mean_tv=float(diff.sum(1).mean()/2),max_kl=float(kl.max()),
            kl_quantiles=np.quantile(kl,[0,.5,.9,.99,1]).tolist(),policy_abs_per_position_quantiles=np.quantile(diff.max(1),[0,.5,.9,.99,1]).tolist(),
            changed_budget_fraction=float(np.mean(budgets[name]!=budgets['original'])),
            exceeds_001=bool(max(abs(dc.mean()),abs(dc[3::4].mean()))>.001))
        print('AUDIT',name,checks[name],flush=True)
    atomic(OUT/'results.json',dict(methods=result,audit=checks,method_seconds=times,scoring_seconds=elapsed,
        analysis_seconds=time.monotonic()-start,plan_sha256=digest(OUT/'plan.json'),purpose=plan['purpose']))
    np.savez_compressed(OUT/'scores.npz',names=list(scores),scores=np.stack(list(scores.values())),nodes=np.stack(list(costs.values())),game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    return dict(output=str(OUT),scoring_seconds=elapsed,audit=checks)

if __name__=='__main__':
    import sys
    if '--freeze' in sys.argv:freeze()
    else:analyze()
