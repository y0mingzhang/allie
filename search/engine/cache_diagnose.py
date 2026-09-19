"""Isolate root numerics after a cache-capacity change; no search or calibration."""
import json
import time
import numpy as np
from scipy.special import logsumexp
from .service import ROOT, atomic


def compare(a,b,rows):
    x=a[:,378:2346].astype(float);y=b[:,378:2346].astype(float)
    x-=logsumexp(x,axis=1,keepdims=True);y-=logsumexp(y,axis=1,keepdims=True)
    ix=np.arange(len(rows));target=np.array([r['target']-378 for r in rows])
    changed=np.flatnonzero(np.any(a!=b,axis=1)).tolist()
    return dict(changed_rows=changed,changed_elements=int((a!=b).sum()),
        max_logit_gap=float(np.max(np.abs(a-b))),
        raw_ce_delta=float((x[ix,target]-y[ix,target]).mean()),
        raw_kl=float((np.exp(x)*(x-y)).sum(1).mean()))


def run(oracle,spec):
    start=time.monotonic()
    tag=spec.get('tag','large')
    assert tag in ('large','small','large-restart')
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'][:1024]
    old=np.concatenate([np.load(ROOT/f'aug-search-v1/mcts-{lo:06d}.npz')['root']
                        for lo in range(0,1024,128)])
    replays=[]
    for rep in range(3):
        out=[]
        for lo in range(0,1024,128):
            oracle.reset();out.append(oracle([r['prefix'] for r in rows[lo:lo+128]]))
        replays.append(np.concatenate(out))
    report=dict(capacity=oracle.capacity,seconds=time.monotonic()-start,
        old_vs_replay=[compare(old,z,rows) for z in replays],
        repeat=[compare(replays[0],z,rows) for z in replays[1:]])
    changed=sorted(set(i for r in report['old_vs_replay'] for i in r['changed_rows']))
    report['changed_positions']=[dict(index=i,game=rows[i]['game'],ply=rows[i]['ply'],
                                    length=len(rows[i]['prefix'])) for i in changed]
    with (ROOT/f'cache-root-replay-{tag}.npz').open('wb') as f:
        np.savez_compressed(f,old=old,replay=np.array(replays))
    atomic(ROOT/f'cache-root-diagnosis-{tag}.json',report)
    return report
