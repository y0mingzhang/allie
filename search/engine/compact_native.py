"""Independent native module; never overwrites the frozen backup binary."""
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
    sources=[Path(__file__).with_name(n) for n in ('compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/compact';folder.mkdir(exist_ok=True)
    target=folder/('_allie_compact'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_compact')
    from .native_board import MOVES
    module.initialize(MOVES);return module


def test(module):
    from .backup_native import load as backup_load
    from .native_board import MOVES
    old=backup_load();rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][:3]
    mapping={m:378+i for i,m in enumerate(MOVES)}
    prefixes=[r['prefix'] for r in rows]+[rows[0]['prefix'][:11]+[mapping[m] for m in ('f2f3','e7e5','g2g4')]]
    def oracle(ps):
        out=[]
        for p in ps:
            seed=int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)
            z=np.random.default_rng(seed).normal(size=2432).astype(np.float32)
            if p==prefixes[-1]:z[mapping['d8h4']]=10.
            out.append(z)
        return np.array(out)
    z=oracle(prefixes);sims=[256]*len(prefixes);cp=[1.25]*len(prefixes)
    tree=module.Tree(prefixes,z,sims,cp);ref=old.Tree(prefixes,z,sims,cp)
    snapshots={};taus=[float('inf'),.5,.1,.025,0.]
    for step in range(1,257):
        a,b=tree.select(),ref.select();assert a==b
        if a:p=oracle(a);tree.update(p);ref.update(p)
        if step in (1,16,64,256):snapshots[step]=ref.backups(taus)
    for a,b in zip(tree.snapshot(),ref.snapshot()):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    compact=tree.compact();assert tree.stats()==ref.stats()
    for step,snapshot in snapshots.items():
        for tau,records in zip(taus,snapshot):
            q=module.reduce(compact,step,tau,tau)
            for i,(ids,values) in enumerate(records):
                np.testing.assert_allclose(q[i,ids],values,atol=3e-12,rtol=0)
    # Independently recurse over the full edge list for asymmetric temperatures.
    nodes=ref.exported();roots=[i for i,x in enumerate(nodes) if x[0]<0]
    for own,opp in [(.025,.5),(.5,.025),(float('inf'),0.),(0.,float('inf'))]:
        def value(i,depth):
            _,boot,terminal,edges=nodes[i]
            if terminal>=0:return 0. if terminal==.5 else -1.
            if not edges:return -boot
            q=np.array([-boot if c<0 else -value(c,depth+1) for c,p,m in edges])
            p=np.array([p for c,p,m in edges]);p/=p.sum();tau=opp if depth%2 else own
            if np.isinf(tau):return p@q
            if tau==0:return q.max()
            return q.max()+tau*np.log(p@np.exp((q-q.max())/tau))
        q=module.reduce(compact,256,own,opp)
        for ri,r in enumerate(roots):
            _,boot,_,edges=nodes[r]
            for c,p,m in edges:np.testing.assert_allclose(q[ri,m-378],-boot if c<0 else -value(c,1),atol=3e-12,rtol=0)
    assert tree.stats()['terminal_visits']>0
    print('PASS compact snapshots, asymmetric independent recursion, terminal signs, identical tree/visits/cost',flush=True)


if __name__=='__main__':test(load())
