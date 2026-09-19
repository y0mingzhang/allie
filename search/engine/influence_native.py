"""Value-influence allocation compiler, control identity and independent choice checks."""
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
    sources=[Path(__file__).with_name(n) for n in ('influence.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/influence';folder.mkdir(exist_ok=True)
    target=folder/('_allie_influence'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_influence')
    from .native_board import MOVES
    module.initialize(MOVES);return module


def test(module):
    from .coverage_native import load as old_load
    from .native_board import MOVES,from_prefix
    old=old_load();rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][:3]
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
    z=oracle(prefixes);sims=[64]*len(prefixes);cp=[2.5]*len(prefixes)
    a=module.Tree(prefixes,z,sims,cp,0,0.,-1.,.1);b=old.Tree(prefixes,z,sims,cp,0,0.,2.)
    for _ in range(64):
        x,y=a.select(),b.select();assert x==y
        if x:p=oracle(x);a.update(p);b.update(p)
    for key,value in a.compact().items():np.testing.assert_array_equal(value,b.compact()[key])
    assert a.stats()==b.stats()
    for mixture in (0.,.25,1.):
        tree=module.Tree(prefixes,z,sims,cp,0,0.,mixture,.1)
        for _ in range(64):
            records=tree.exported();visits=tree.visits;roots=[i for i,r in enumerate(records) if r[0]<0]
            values={}
            def value(i):
                if i in values:return values[i]
                _,boot,terminal,edges=records[i]
                if terminal>=0:v=0. if terminal==.5 else -1.
                elif not edges:v=-boot
                else:
                    q=np.array([-boot if c<0 else -value(c) for c,p,m in edges]);p=np.array([p for c,p,m in edges])
                    v=q.max()+.1*np.log(p@np.exp((q-q.max())/.1)/p.sum())
                values[i]=v;return v
            if mixture<1:
                for i in range(len(records)):np.testing.assert_allclose(tree.values[i],value(i),atol=3e-12,rtol=0)
            expected=[]
            for ri,root in enumerate(roots):
                path=list(prefixes[ri]);i=root
                while records[i][3]:
                    _,boot,_,edges=records[i]
                    p=np.array([p for c,p,m in edges]);n=np.array([0 if c<0 else visits[c] for c,p,m in edges])
                    if i==root:w=np.sqrt(p*(1-p))
                    elif mixture==1:w=p
                    else:
                        q=np.array([-boot if c<0 else -value(c) for c,p,m in edges]);sp=p*np.exp((q-q.max())/.1);sp/=sp.sum()
                        w=(1-mixture)*sp+mixture*p/p.sum()
                    c,_,move=edges[int(np.argmax(w/(1+n)))];path.append(move)
                    if c<0:break
                    i=c
                if from_prefix(path).outcome()<0:expected.append(path)
            got=tree.select();assert got==expected,(mixture,got,expected)
            if got:tree.update(oracle(got))
        assert tree.stats()['terminal_visits']>0
    print('PASS exact coverage control; independent recursive soft values and full selected paths for every step; terminal signs and prior/soft/mixture allocations',flush=True)


if __name__=='__main__':test(load())
