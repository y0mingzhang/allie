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
    sources=[Path(__file__).with_name(n) for n in ('behavior.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/behavior';folder.mkdir(exist_ok=True)
    target=folder/('_allie_behavior'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_behavior')
    return module

def test(module):
    # Tree with a partial policy, a terminal loss and an unvisited mass. The
    # scalar Python recursion computes expectation under the normalized tilted
    # policy independently, including the depth-parity and value signs.
    data=dict(parent=np.array([-1,0,0,1,1]),move=np.array([-379,3,7,5,8]),
        depth=np.array([0,1,1,2,2]),born=np.array([0,1,2,3,4]),
        degree=np.array([3,3,0,2,0]),prior=np.array([0,.4,.3,.2,.5]),
        boot=np.array([-.2,.1,0,.3,0]),mass=np.array([1,1,0,1,0.]),
        terminal=np.array([-1,-1,0,-1,.5]),roots=np.array([0]))
    def ref(budget,own,opp):
        def val(i):
            if data['terminal'][i]>=0:return 0. if data['terminal'][i]==.5 else -1.
            base=-data['boot'][i];total=data['mass'][i]
            if total==0:return base
            children=[j for j in range(5) if data['parent'][j]==i and data['born'][j]<=budget]
            w=[data['prior'][j] for j in children];q=[-val(j) for j in children]
            w.append(max(0.,total-sum(w)));q.append(base)
            w=np.array(w);q=np.array(q);active=w>0;w=w[active];q=q[active]
            tau=opp if data['depth'][i]%2 else own
            if np.isinf(tau):return (w*q).sum()/w.sum()
            if tau==0:return q.max()
            weights=w*np.exp((q-q.max())/tau);return (weights*q).sum()/weights.sum()
        out=np.full(1968,-data['boot'][0])
        for j in (1,2):
            if data['born'][j]<=budget:out[data['move'][j]]=-val(j)
        return out
    for budget in range(5):
        for a,b in [(0.,0.),(.1,.1),(.2,.1),(float('inf'),float('inf'))]:
            np.testing.assert_allclose(module.reduce(data,budget,a,b)[0],ref(budget,a,b),rtol=0,atol=1e-14)
    print('PASS independent recursive tilted-policy expected outcomes, unseen mass, terminals, mixed depths and prefix budgets',flush=True)
if __name__=='__main__':test(load())
