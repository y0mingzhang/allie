"""Frozen lambda3 critic projection, paired live golden control on identical trees."""
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,softmax
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .growforest_native import load
from .scaled_count_native import load as backup_load
from .bellman_projection_native import load as projection_load
from .handles import HandleOracle
from .budget_policy import policy
from .golden_metrics import summarize

OUT=ROOT/'golden-bellman-projection-v1'
GOLD=Path('/data/group_data/dei-group/yimingz3/allie/strat-eval-v1')


def freeze():
    OUT.mkdir(exist_ok=True)
    dev=ROOT/'aug-bellman-projection-expanded-v1/results.json';report=json.loads(dev.read_text())
    assert set(report['fit_cv_selected'].values())=={'lambda3'}
    manifest=json.loads((GOLD/'manifest.json').read_text())
    plan=dict(strength=3.,budget=1000,roots_per_block=256,threads=4,
        parameters={name:report['results'][name]['parameters'][0] for name in ('unchanged','lambda3')},
        dev_sha256=digest(dev),sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),
        laws_sha256=digest(ROOT/'training-cm-laws.json'),export_sha256=digest(ROOT/'serving-export/provenance.json'),
        canonical_baseline_sha256=digest(ROOT/'golden-baseline/results.json'),
        old_control_scores_sha256=digest(ROOT/'golden-temperature-stack-v1/scores.npz'),
        sidecars=dict(strat=manifest['sha256'],feats=manifest['feats_sha256']),
        sources={name:digest(Path(__file__).with_name(name)) for name in (
            'golden_bellman.py','bellman_projection.cpp','bellman_projection_native.py','diff_backup.cpp',
            'growforest.cpp','threadforest.cpp','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp',
            'mcts_native.hpp','board.cpp','handles.py','direct.py','scaled_count.cpp','budget_policy.py','golden_metrics.py')},
        selection='lambda3 selected by BOTH expanded August gameCV metrics, all calibration frozen. Golden reports old cached parent, fresh unchanged parent and projected variant. No selection among them on golden.',
        execution='Fresh current-hardware search, same tree and model outputs for parent/projection. Raw critic still drives expansion; projection is output-only. Every block verifies zero-strength Q identity before scoring lambda3. One root prefill and actual NN counts charged identically. CPU projection timing reported separately.',
        caveat='Reused8192 golden sample is exploratory; every method goes in registry. Original full raw anchors and CM laws fixed. No training, external memory, observed future time/outcome, or target-dependent search. Final success needs fresh disjoint games; target remains10x on BOTH metrics.')
    pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return plan


def run(oracle,spec):
    plan=freeze();start=time.monotonic()
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    for key,file in [('strat','strat.npz'),('feats','feats.npz')]:assert digest(GOLD/file)==plan['sidecars'][key]
    with np.load(GOLD/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(GOLD/'feats.npz') as f:side=f['feats']
    seconds=[]
    for row in rows:
        rr,cc=row['row'],row['column'];assert tokens[rr,cc]==row['target'] and labels[rr,cc]==row['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(row['prefix']):cc],row['prefix'])
        seconds.append(side[rr,cc-1,0])
    seconds=np.array(seconds);del tokens,labels,side
    native,reducer,projection=load(),backup_load(),projection_load()
    for lo in range(0,len(rows),plan['roots_per_block']):
        dest=OUT/f'{lo:06d}.npz'
        if dest.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+plan['roots_per_block']];n=len(part);ar=np.arange(n)
        width=max(len(r['legal']) for r in part);ids=np.zeros((n,width),np.int32);mask=np.zeros((n,width),bool)
        target=np.array([r['legal'].index(r['target']) for r in part]);forced=np.array([len(r['legal'])==1 for r in part])
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
        tick=time.monotonic();oracle.reset();bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits.astype(float)
        prefill=oracle.new_tokens;tree=native.Tree([r['prefix'] for r in part],z,np.where(forced,0,128).tolist(),[2.5]*n,plan['threads'])
        def advance():
            while not tree.done:
                h=tree.select()
                if len(h):tree.update(bridge(h))
        advance();tree.grow(np.where(forced,0,1000).tolist());advance();compact=tree.compact();nodes=np.array(tree.evals)
        assert int(nodes.sum())==bridge.queries
        q=reducer.Backup(compact,1000,16.).reduce(np.log(.2),-.5,ids)[0]
        worker=projection.Projection(compact,1000);worker.project(0.)
        np.testing.assert_array_equal(worker.reduce(np.log(.2),-.5,ids)[0],q)
        pt=time.monotonic();info=worker.project(plan['strength']);newq=worker.reduce(np.log(.2),-.5,ids)[0];process_seconds=time.monotonic()-pt
        assert np.isfinite(newq).all() and abs(newq).max()<=1+1e-12
        scores={};policies={}
        raw=z[:,378:2346]-logsumexp(z[:,378:2346],axis=1,keepdims=True)
        actual=np.array([r['target']-378 for r in part])
        scores['port_raw']=np.stack([-raw[ar,actual],raw.argmax(1)==actual,np.exp(raw.max(1))],1)
        p0=softmax(np.where(mask,z[:,378:2346][ar[:,None],ids],-np.inf),axis=1)
        for name,prob in [('new_legal',p0),*[(name,policy(part,z,value,ids,mask,seconds[lo:lo+n],plan['parameters'][name]))
                                          for name,value in [('unchanged',q),('lambda3',newq)]]]:
            chosen=np.where(prob==prob.max(1)[:,None],ids,1968).min(1)
            scores[name]=np.stack([-np.log(prob[ar,target]),chosen==ids[ar,target],prob.max(1)],1)
            policies[name]=prob
        stat=tree.stats();del tree,worker
        stat.update(seconds=time.monotonic()-tick,projection_seconds=process_seconds,prefill_tokens=prefill,
                    forward_seconds=oracle.forward_seconds,root_queries=n,clipped=int(info['clipped']),active=int(info['active']),job=os.environ.get('SLURM_JOB_ID'))
        tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,**compact,root=z,q=q,projected_q=newq,nodes=nodes,ids=ids,mask=mask,
            score_names=list(scores),scores=np.stack(list(scores.values())),policy_names=list(policies),policy=np.stack(list(policies.values())),
            stats=json.dumps(stat),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
        tmp.replace(dest);print('golden Bellman',lo+n,len(rows),stat['seconds'],flush=True)
    scores={name:[] for name in ('port_raw','new_legal','unchanged','lambda3')};counts=[];stats=[]
    for path in sorted(OUT.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],[r['game'] for r in rows[lo:hi]])
            for name in scores:scores[name].append(f['scores'][list(f['score_names']).index(name)])
            counts.append(f['nodes']);stats.append(json.loads(str(f['stats'])))
    scores={k:np.concatenate(v) for k,v in scores.items()};nodes=np.concatenate(counts)
    with np.load(ROOT/'golden-temperature-stack-v1/scores.npz') as f:
        np.testing.assert_array_equal(f['game'],[r['game'] for r in rows]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows])
        scores['old_cached']=f['scores'][list(f['names']).index('temperature')];oldnodes=f['nodes']
    costs={k:nodes if k in ('unchanged','lambda3') else oldnodes if k=='old_cached' else np.zeros(len(rows)) for k in scores}
    result=summarize(rows,scores,costs,references=('new_legal','unchanged','old_cached'))
    atomic(OUT/'results.json',dict(methods=result,positions=len(rows),plan_sha256=digest(OUT/'plan.json'),
        elapsed_seconds=time.monotonic()-start,search_block_seconds=sum(s['seconds'] for s in stats),
        projection_seconds=sum(s['projection_seconds'] for s in stats),prefill_tokens=sum(s['prefill_tokens'] for s in stats),
        clipping_fraction=sum(s['clipped'] for s in stats)/sum(s['active'] for s in stats),caveat=plan['caveat']))
    np.savez_compressed(OUT/'scores.npz',names=list(scores),scores=np.stack(list(scores.values())),nodes=nodes,
                        game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    print('GOLDEN BELLMAN',{k:{m:v[m] for m in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm')} for k,v in result.items()},flush=True)
    return dict(study=OUT.name,positions=len(rows),seconds=time.monotonic()-start)


if __name__=='__main__':freeze()
