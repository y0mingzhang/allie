"""Compile and independently check bounded tactical expansion and backups."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from .service import ROOT

def load():
    import pybind11
    src=Path(__file__).with_name('tactical.cpp');files=[src,*[src.with_name(n) for n in ('board.cpp','mcts_native.hpp')]]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}|{'python':sys.version}
    folder=ROOT/'runtime/tactical';folder.mkdir(exist_ok=True)
    target=folder/('_allie_tactical'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new');subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(src),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-I'+str(Path(__file__).resolve().parents[2]/'vendor/chess-library/include'),'-o',str(tmp)],check=True);tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));m=importlib.import_module('_allie_tactical')
    from .native_board import MOVES
    m.initialize(MOVES);return m

def test(m):
    import chess
    from .native_board import MOVES,MOVE_ID
    from .sample_data import read
    rows=read('aug-tune-expanded-v1')['rows'][:3];ps=[r['prefix'] for r in rows]
    ps.append(ps[0][:11]+[MOVE_ID[u] for u in ['f2f3','e7e5','g2g4']])
    def oracle(prefixes):
        zs=[]
        for p in prefixes:
            seed=int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)
            z=np.random.default_rng(seed).normal(size=2432).astype(np.float32);zs.append(z)
        return np.array(zs)
    t=m.Tree(ps,oracle(ps),128,6,2);seen=set();cost=0
    while not t.done:
        q=t.select()
        for p in q:
            assert tuple(p) not in seen;seen.add(tuple(p))
            b=chess.Board()
            for token in p[11:]:b.push_uci(MOVES[token-378])
            assert b.outcome(claim_draw=False) is None
        if q:t.update(oracle(q));cost+=len(q)
    assert cost==sum(t.evals) and max(t.evals)<=128
    data=t.inspect()
    for i,(par,p,base,term,edges,depth) in enumerate(data):
        assert depth<=6
        if par>=0 and data[par][5]>0:
            b=chess.Board()
            for token in data[par][1][11:]:b.push_uci(MOVES[token-378])
            mv=chess.Move.from_uci(MOVES[p[-1]-378]);assert b.is_check() or b.is_capture(mv) or mv.promotion
    for tau in [np.inf,.2,.05,0.]:
        value=np.zeros(len(data))
        for i in range(len(data)-1,-1,-1):
            par,p,base,term,edges,depth=data[i]
            if term>=0:value[i]=0 if term==.5 else -1;continue
            if not edges:value[i]=base;continue
            probs=np.array([e[2] for e in edges]);v=np.array([base if e[1]<0 else -value[e[1]] for e in edges]);probs/=probs.sum()
            value[i]=probs@v if np.isinf(tau) else (v.max() if tau==0 else tau*logsumexp(v/tau,b=probs))
        got=t.q(tau)
        for r,root in enumerate(t.roots):
            for move,child,p in data[root][4]:
                expected=-value[child] if child>=0 else data[root][2]
                np.testing.assert_allclose(got[r,move-378],expected,rtol=1e-11,atol=1e-11)
        assert got[-1,MOVE_ID['d8h4']-378]==1.
    print('PASS no duplicate/terminal NN requests, strict128budget/depth6, rule-only tactical moves, independent negamax/soft/mean backups and mate sign',flush=True)

if __name__=='__main__':test(load())
