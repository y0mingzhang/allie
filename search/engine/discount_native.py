"""Critic-regularized backup; identity and independent-recursion checks."""
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
    sources=[Path(__file__).with_name(n) for n in ('discount.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/discount';folder.mkdir(exist_ok=True)
    target=folder/('_allie_discount'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_discount')


def test(module):
    from .compact_native import load as old_load
    old=old_load()
    # Includes unexpanded mass, unequal priors, delayed birth, checkmate and draw.
    data={k:np.array(v) for k,v in dict(
        parent=[-1,0,0,1,1,2,2],move=[-1,0,1,2,3,4,5],depth=[0,1,1,2,2,2,2],
        born=[0,1,2,3,4,5,6],roots=[0],degree=[3,2,3,0,0,0,0],
        prior=[1.,.3,.4,.2,.8,.2,.3],boot=[.2,-.3,.4,.5,-.1,.7,-.9],
        mass=[1.,1.,1.,0.,0.,0.,0.],terminal=[-1.,-1.,-1.,0.,.5,-1.,-1.]).items()}
    for budget in (1,3,6):
        for lam in (0.,.25,.5,.75,1.):
            def ref(i):
                if data['terminal'][i]>=0:return 0. if data['terminal'][i]==.5 else -1.
                base=-data['boot'][i]
                if data['mass'][i]==0 or lam==0:return base
                children=np.flatnonzero((data['parent']==i)&(data['born']<=budget))
                q=np.array([-ref(c) for c in children]+[base])
                p=np.array([data['prior'][c] for c in children]+[max(0.,data['mass'][i]-data['prior'][children].sum())])
                q=q[p>0];p=p[p>0];v=q.max()+.1*np.log(p@np.exp((q-q.max())/.1)/p.sum())
                return (1-lam)*base+lam*v
            result=module.reduce(data,budget,.1,lam)
            for c in (1,2):
                expected=-ref(c) if data['born'][c]<=budget else -data['boot'][0]
                np.testing.assert_allclose(result[0,data['move'][c]],expected,atol=1e-12,rtol=0)
            if lam==1:np.testing.assert_array_equal(result,old.reduce(data,budget,.1,.1))
    print('PASS lambda1 exact baseline; independent recursion including partial mass, budget filtering, terminal signs and lambda0 critic limit',flush=True)


if __name__=='__main__':test(load())
