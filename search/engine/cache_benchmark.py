"""Test larger root batches at equal MCTS budgets on the existing allocation."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax,logsumexp
from .service import ROOT,GLOBAL_STOP,atomic
from .backup_native import load
from .cache_diagnose import compare


def run(oracle,spec):
    module=load();rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'][:1024]
    runs=[];baseline_root=None;baseline_q=None;baseline_visits=None
    stage=spec.get('stage','large')
    for bs in ([128] if stage=='small' else [128,512]):
        start=time.monotonic();roots=[];qs=[];visits=[];cost=[]
        for lo in range(0,len(rows),bs):
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            part=rows[lo:lo+bs];oracle.reset();begin=time.monotonic()
            z=oracle([r['prefix'] for r in part]);tree=module.Tree([r['prefix'] for r in part],z,[1000]*len(part),[1.25]*len(part))
            for _ in range(1000):
                prefixes=tree.select()
                if prefixes:tree.update(oracle(prefixes))
            q=np.zeros((len(part),1968),np.float32);v=np.zeros_like(q,np.int32)
            for i,(ids,n,values,prior) in enumerate(tree.snapshot()):q[i,ids]=values;v[i,ids]=n
            stats=tree.stats();del tree
            stats.update(seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
            roots.append(z);qs.append(q);visits.append(v);cost.append(stats)
        z=np.concatenate(roots);q=np.concatenate(qs);v=np.concatenate(visits)
        if bs==128:
            old_roots=[];old_q=np.zeros_like(q);old_v=np.zeros_like(v)
            for lo in range(0,len(rows),128):
                with np.load(ROOT/f'aug-search-v1/mcts-{lo:06d}.npz') as old:
                    old_roots.append(old['root']);ids=old['ids'];mask=old['mask']
                    for i in range(len(ids)):
                        old_q[lo+i,ids[i,mask[i]]]=old['q'][-1,0,i,mask[i]]
                        old_v[lo+i,ids[i,mask[i]]]=old['visits'][-1,i,mask[i]]
            old_z=np.concatenate(old_roots)
            # A fresh OLD-capacity process reproduces the same one-root rounding
            # difference as the new-capacity process (027/028 diagnosis). Keep
            # historical differences visible; compare capacity with fresh runs.
            drift=dict(historical_root=compare(old_z,z,rows),
                historical_q_max_gap=float(np.max(np.abs(q-old_q))),
                historical_changed_visits=int((v!=old_v).sum()))
            assert drift['historical_root']['raw_kl']<1e-6,drift
            if stage=='small':
                with (ROOT/'cache-small-reference.npz').open('wb') as f:np.savez_compressed(f,z=z,q=q,v=v)
            else:
                with np.load(ROOT/'cache-small-reference.npz') as ref:
                    np.testing.assert_array_equal(z,ref['z'])
                    np.testing.assert_array_equal(q,ref['q'])
                    np.testing.assert_array_equal(v,ref['v'])
                drift['exact_fresh_small_capacity_identity']=True
            baseline_root=z;baseline_q=q;baseline_visits=v
        else:
            a=baseline_root[:,378:2346].astype(float);b=z[:,378:2346].astype(float)
            a-=logsumexp(a,axis=1,keepdims=True);b-=logsumexp(b,axis=1,keepdims=True)
            y=np.array([r['target']-378 for r in rows]);ar=np.arange(len(rows))
            drift=dict(raw_ce_delta=float((a[ar,y]-b[ar,y]).mean()),raw_kl=float((np.exp(a)*(a-b)).sum(1).mean()),
                mean_q_abs_gap=float(np.abs(q-baseline_q).sum()/sum(len(r['legal']) for r in rows)),
                changed_visit_entries=int((v!=baseline_visits).sum()))
            assert abs(drift['raw_ce_delta'])<.005 and drift['raw_kl']<.005,drift
        report=dict(batch_size=bs,positions=len(rows),seconds=time.monotonic()-start,
            block_seconds=sum(c['seconds'] for c in cost),forward_seconds=sum(c['forward_seconds'] for c in cost),
            evaluated_leaves=sum(c['evaluated_leaves'] for c in cost),new_tokens=sum(c['new_tokens'] for c in cost),drift=drift)
        report['evaluated_leaves_per_second']=report['evaluated_leaves']/report['block_seconds'];runs.append(report)
        print('Cache benchmark',report,flush=True)
    result=dict(stage='Same checkpoint and1000sim/root, storage capacity and root batching only',
        capacity=oracle.capacity,startup_seconds=oracle.startup_seconds,runs=runs,
        speedup=runs[0]['block_seconds']/runs[-1]['block_seconds'],note='Speedup uses block time excluding reference file reads. BF16 batching may perturb branches; drift reported. No new quality or CM claim.')
    atomic(ROOT/f'cache-benchmark-{stage}.json',result);return result
