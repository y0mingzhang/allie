"""Development-only throughput and numerical drift checks for the cached port."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import log_softmax, softmax
from .tree import batch

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'


def compare(a,b,rows):
    a=a.astype(float);b=b.astype(float)
    legal=np.zeros((len(rows),1968),bool)
    for i,r in enumerate(rows):legal[i,np.array(r['legal'])-378]=True
    lp=lambda x:log_softmax(np.where(legal,x[:,378:2346],-np.inf),axis=1)
    la,lb=lp(a),lp(b);target=np.array([r['target']-378 for r in rows]);ix=np.arange(len(rows))
    delta=(la-lb)[ix,target]
    kl=np.sum(np.exp(la)*np.where(legal,np.subtract(la,lb,where=legal,out=np.zeros_like(la)),0),axis=1)
    va=softmax(a[:,2413:2416],axis=1)@np.array([1.,.5,0.])
    vb=softmax(b[:,2413:2416],axis=1)@np.array([1.,.5,0.])
    # Paired game bootstrap: several positions from a game are correlated.
    games,inv=np.unique([r['game'] for r in rows],return_inverse=True)
    sums=np.bincount(inv,weights=delta);counts=np.bincount(inv)
    w=np.random.default_rng(912).poisson(1,(1024,len(games)))
    draws=(w@sums)/(w@counts)
    return dict(positions=len(rows),games=len(games),mean_legal_policy_kl=float(kl.mean()),
        legal_ce_delta=float(delta.mean()),paired_ce_delta_95pct=np.quantile(draws,[.025,.975]).tolist(),
        top_move_agreement=float((la.argmax(1)==lb.argmax(1)).mean()),
        wdl_mean_abs_delta=float(np.abs(va-vb).mean()),wdl_max_abs_delta=float(np.abs(va-vb).max()),
        max_logit_delta=float(np.abs(a-b).max()))


def run(oracle):
    rows=json.loads((ROOT/'dev.json').read_text())['positions']
    ref=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'] for i in range(0,len(rows),128)])
    fresh=[];cached=[];timing=[]
    for lo in range(0,len(rows),128):
        block=rows[lo:lo+128];seq=[r['prefix'] for r in block]
        oracle.reset();fresh.append(oracle(seq))
        oracle.reset();oracle([s[:-1] for s in seq]);cached.append(oracle(seq))
    fresh=np.concatenate(fresh);cached=np.concatenate(cached)
    report=dict(stage='Numerical/throughput checks on development data; no method tuning',
        constructor_startup_seconds=oracle.startup_seconds,
        port_vs_reference=compare(ref,fresh,rows),cached_vs_fresh=compare(fresh,cached,rows))
    for n in (16,32,64):
        for bs in (1024,2048):
            for repeat in range(2):
                oracle.reset();q,logits,stats=batch(rows[:n],oracle,batch_size=bs)
                stats.update(roots=n,batch_size=bs,repeat=repeat,new_tokens=oracle.new_tokens,
                    forward_seconds=oracle.forward_seconds,positions_per_second=n/stats['seconds'])
                timing.append(stats);print('benchmark',stats,flush=True)
    report['four_ply']=timing
    np.savez_compressed(ROOT/'engine-dev-parity.npz',reference=ref,fresh=fresh,cached=cached)
    return report
