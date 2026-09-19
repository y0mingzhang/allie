"""Compile the isolated parallel-root implementation and prove oracle equivalence."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import numpy as np
from .backup_native import ROOT

def load():
    import pybind11
    sources=[Path(__file__).with_name(n) for n in ('rollout.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/rollout';folder.mkdir(exist_ok=True)
    target=folder/('_allie_rollout'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-fopenmp','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_rollout')
    from .native_board import MOVES
    module.initialize(MOVES);return module

def test(module):
    import chess
    from .native_board import MOVES,MOVE_ID
    rows=json.loads((ROOT/'results/search-v1/aug-tune-v1/sample.json').read_text())['positions']
    picks=[rows[i] for i in (0,1200,2400)]
    forced=next(r for r in rows if len(r['legal'])==1);picks.append(forced)
    header=rows[0]['prefix'][:11];picks.append(dict(prefix=header+[MOVE_ID[x] for x in ['f2f3','e7e5','g2g4']]))
    ps=[r['prefix'] for r in picks];gids=[31,18,953,48,791];seed=238619;r=4;depth=8
    def oracle(prefixes):
        return np.array([np.random.default_rng(int(hashlib.sha256(np.asarray(p,np.int16).tobytes()).hexdigest()[:8],16)).normal(size=2432).astype(np.float32) for p in prefixes])
    def mix(x):
        mask=(1<<64)-1;x=(x+0x9e3779b97f4a7c15)&mask;x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
        x=((x^(x>>27))*0x94d049bb133111eb)&mask;return x^(x>>31)
    def uniform(gid,action,rep,dep):
        return (mix(seed^mix(gid)^mix(action+98317)^mix(rep+1+81233)^mix(dep+99871))>>11)*2**-53
    def board(p):
        b=chess.Board()
        for t in p[11:]:b.push_uci(MOVES[t-378])
        return b
    t=module.Rollouts(ps,oracle(ps),gids,r,depth,seed);known={i:p for i,p in enumerate(ps)};evaluations=np.zeros(len(ps),int)
    old_trace=None;old_states={}
    for dep in range(1,depth+1):
        handles=t.select();queries=[]
        for node,parent,move,length in handles:
            p=known[parent]+[move];assert len(p)==length
            b=board(known[parent]);assert chess.Move.from_uci(MOVES[move-378]) in b.legal_moves
            known[node]=p;queries.append(p)
        if len(handles):t.update(oracle(queries))
        trace=list(t.trace());q=np.zeros((len(ps),1968));sq=q.copy();counts=np.zeros_like(q,int)
        for root,action,rep,node,length,d,val,ended,fen in trace:
            if dep>1 and d==dep:
                old=old_states[(root,action,rep)];b=chess.Board(old[-1])
                legal=[MOVE_ID[m.uci()] for m in b.legal_moves]
                z=oracle([known[old[3]]])[0];w=np.exp(z[legal].astype(float)-float(z[legal].max()));w/=w.sum()
                u=uniform(gids[root],action,rep,dep)
                expected=legal[min(int(np.searchsorted(np.cumsum(w),u,side='right')),len(legal)-1)]
                b.push_uci(MOVES[expected-378]);assert b.fen()==fen,(dep,root,action,rep)
            # Forced roots are the only zero-depth trajectories.
            if d==0:assert len(list(board(ps[root]).legal_moves))==1;assert val==0
            elif node in known and not ended:
                b=board(known[node]);assert b.fen()==fen
                z=oracle([known[node]])[0];w=np.exp(z[2413:2416].astype(float)-float(z[2413:2416].max()));w/=w.sum()
                expected=(w[0]-w[2])*(1 if b.turn==board(ps[root]).turn else -1)
                np.testing.assert_allclose(val,expected,atol=2e-15)
            elif ended and d>0:
                b=chess.Board(fen);out=b.outcome(claim_draw=False)
                if out is not None:
                    expected=0. if out.winner is None else (1. if out.winner==board(ps[root]).turn else -1.)
                    assert val==expected
            q[root,action-378]+=val;sq[root,action-378]+=val*val;counts[root,action-378]+=1
        good=counts>0;q[good]/=counts[good];sq[good]/=counts[good];sq=np.maximum(sq-q*q,0.)
        snap=t.snapshot();np.testing.assert_allclose(snap['q'],q,atol=1e-15);np.testing.assert_allclose(snap['variance'],sq,atol=1e-15)
        old_states={(row[0],row[1],row[2]):row for row in trace}
        # At depth1 all replicas share the first node. Sampling checks below use
        # per-replica state, so reconstruct the sampled move from each child trace
        # at dep>=2 rather than guessing replica from its shared parent.
        old_trace=trace
    # Verify root/action RNG and outputs are independent of other roots/batch IDs.
    for i,p in enumerate(ps):
        one=module.Rollouts([p],oracle([p]),[gids[i]],r,depth,seed);paths={0:p}
        for dep in range(depth):
            hs=one.select();qs=[]
            for node,parent,move,length in hs:paths[node]=paths[parent]+[move];qs.append(paths[node])
            if len(hs):one.update(oracle(qs))
        np.testing.assert_array_equal(one.snapshot()['q'][0],t.snapshot()['q'][i])
        np.testing.assert_array_equal(one.snapshot()['variance'][0],t.snapshot()['variance'][i])
        assert one.evals[0]==t.evals[i]
    # Fool's mate d8h4 is an exact root win without an NN leaf call.
    assert t.snapshot()['q'][-1,MOVE_ID['d8h4']-378]==1.
    assert t.evals[3]==0
    print('PASS rollout legality, WDL/terminal perspective, sample means/variance, per-root RNG batch independence, forced-move skip')

if __name__=='__main__':test(load())
