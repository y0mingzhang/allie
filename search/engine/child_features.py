"""Opponent difficulty predicted after each legal candidate; model-only features.

The time head refers to the next mover (the opponent after our candidate).
No actual future thinking time or human reply is ever supplied.
"""
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from .balanced_eval import ROOT,GLOBAL_STOP,atomic,digest
from .native_board import from_prefix


def run(oracle,spec):
    out=ROOT/'aug-child-features-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-tune-v1/sample.json';rows=json.loads(source.read_text())['positions'];bs=512
    plan=dict(sample_sha256=digest(source),roots_per_batch=bs,
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('native_board.py'),Path(__file__).with_name('direct.py')]},
        features=['log_expected_opponent_seconds','opponent_time_entropy','opponent_legal_policy_entropy','opponent_illegal_mass','root_mover_oneply_value','draw_probability'],
        stage='August development only. All legal candidate children, terminal values exact and other terminal features zero. Predicted opponent time, never observed future time. Extra model queries charged explicitly.')
    p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic(p,plan)
    seconds=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    start=time.monotonic();stats_all=[]
    for lo in range(0,len(rows),bs):
        path=out/f'{lo:06d}.npz'
        if path.exists():
            with np.load(path) as z:stats_all.append(json.loads(str(z['stats'])))
            continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+bs];n=len(part);k=max(len(r['legal']) for r in part)
        oracle.reset();begin=time.monotonic();root=oracle([r['prefix'] for r in part]);features=np.zeros((n,k,6));nodes=np.zeros(n,int)
        queries=[];owners=[];legal=[];terminals=np.zeros((n,k),bool)
        for i,r in enumerate(part):
            b=from_prefix(r['prefix']);assert sorted(b.legal())==sorted(r['legal'])
            for j,move in enumerate(r['legal']):
                child=b.child(move);outcome=child.outcome()
                if outcome>=0:
                    terminals[i,j]=True;features[i,j,4]=0. if outcome==.5 else 1.;features[i,j,5]=float(outcome==.5)
                else:
                    queries.append(r['prefix']+[move]);owners.append((i,j));legal.append(child.legal());nodes[i]+=1
        # Prefix KV from every root is already cached. Leaf requests can be large;
        # DirectOracle chunks their token work, no generation scheduler involved.
        predictions=oracle(queries) if queries else np.empty((0,2432))
        tp=softmax(predictions[:,2350:2413].astype(float),axis=1);wdl=softmax(predictions[:,2413:2416].astype(float),axis=1)
        move=softmax(predictions[:,378:2346].astype(float),axis=1)
        for t,((i,j),moves) in enumerate(zip(owners,legal)):
            raw=move[t,np.array(moves)-378];mass=raw.sum();p=raw/mass
            features[i,j]=[np.log1p(tp[t]@seconds),-np.sum(tp[t]*np.log(np.maximum(tp[t],1e-300))),
                -np.sum(p*np.log(np.maximum(p,1e-300))),max(0.,1-mass),wdl[t,2]-wdl[t,0],wdl[t,1]]
        assert np.isfinite(features).all()
        stats=dict(seconds=time.monotonic()-begin,logical_nonroot_nn_requests=int(nodes.sum()),
            unique_nonroot_nn_requests=oracle.next_row-len(set(tuple(r['prefix']) for r in part)),new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,features=features,root=root,nodes=nodes,terminal=terminals,game=np.array([r['game'] for r in part]),stats=json.dumps(stats))
        tmp.replace(path);stats_all.append(stats);print('Opponent features',lo+n,'/',len(rows),stats['seconds'],flush=True)
    report=dict(positions=len(rows),seconds=time.monotonic()-start,cost=stats_all)
    atomic(out/'worker.json',report);return report
