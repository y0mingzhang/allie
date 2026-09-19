"""Build and test read-only alternative backups without changing the existing engine."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load():
    import pybind11
    sources = [Path(__file__).with_name(n) for n in ('backups.cpp','board.cpp','mcts_native.hpp')]
    key = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    key['python'] = sys.version
    folder = ROOT/'results/search-v1/runtime/backups'
    folder.mkdir(exist_ok=True)
    target = folder/('_allie_backups'+sysconfig.get_config_var('EXT_SUFFIX'))
    stamp = folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text()) != key:
        temp = target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
                        '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
                        '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target)
        stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder))
    module = importlib.import_module('_allie_backups')
    from .native_board import MOVES
    module.initialize(MOVES)
    return module


def test(module):
    """Compare every backed-up root against an independent Python recursion.

    The read-only instrumentation must also leave native MCTS visit/Q arrays exact.
    A forcing Fool's Mate root exercises terminal backup signs.
    """
    from .native_board import MOVES
    import _allie_board_v2 as old
    old.initialize(MOVES)
    rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][:3]
    mapping={m:378+i for i,m in enumerate(MOVES)}
    prefixes=[r['prefix'] for r in rows]+[rows[0]['prefix'][:11]+[mapping[m] for m in ('f2f3','e7e5','g2g4')]]
    def oracle(ps):
        zs=[]
        for p in ps:
            seed=int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)
            z=np.random.default_rng(seed).normal(size=2432).astype(np.float32)
            if p==prefixes[-1]: z[mapping['d8h4']]=10.
            zs.append(z)
        return np.array(zs)
    z=oracle(prefixes);sims=[64]*len(prefixes);cp=[1.25]*len(prefixes)
    new=module.Tree(prefixes,z,sims,cp);ref=old.NativeMCTS(prefixes,z,sims,cp)
    ref.first_prior=ref.preserve_depth=True
    for iteration in range(64):
        a,b=new.select(),ref.select();assert a==b
        if a: preds=oracle(a);new.update(preds);ref.update(preds)
        if iteration in (0,15,63):
            for record in new.snapshot(): assert sum(record[1])==iteration+1
    for a,b in zip(new.snapshot(),ref.summaries()):
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    stats=new.stats();assert sum(new.evals)==stats['evaluated_leaves']
    assert stats['terminal_visits']>0
    nodes=new.exported();roots=[i for i,x in enumerate(nodes) if x[0]<0]
    taus=[float('inf'),.5,.1,.025,0.]
    actual=new.backups(taus)
    for tau,got in zip(taus,actual):
        def value(i):
            _,boot,terminal,edges=nodes[i]
            if terminal>=0:return 0. if terminal==.5 else -1.
            if not edges:return -boot
            q=np.array([-boot if child<0 else -value(child) for child,p,m in edges])
            p=np.array([p for child,p,m in edges]);p/=p.sum()
            if np.isinf(tau):return float(p@q)
            if tau==0:return float(q.max())
            high=q.max();return float(high+tau*np.log(p@np.exp((q-high)/tau)))
        for root,(ids,q) in zip(roots,got):
            _,boot,_,edges=nodes[root]
            expected=[-boot if child<0 else -value(child) for child,p,m in edges]
            np.testing.assert_allclose(q,expected,atol=2e-12,rtol=0)
            assert np.max(np.abs(q))<=1+1e-12
    print('PASS backup recursion, terminal signs, intermediate visits, per-root costs, unchanged MCTS',flush=True)


if __name__=='__main__':
    test(load())
