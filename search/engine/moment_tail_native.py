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
    sources=[Path(__file__).with_name(n) for n in ('moment_tail.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/moment_tail';folder.mkdir(exist_ok=True)
    target=folder/('_allie_moment_tail'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-fopenmp','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_moment_tail')
    from .native_board import MOVES
    return module

def test(module):
    from .compact_native import load as reference_load
    ref=reference_load()
    data=dict(parent=np.array([-1,0,0,0,1,1,2,4],np.int32),move=np.array([-1,11,12,13,21,22,31,41],np.int32),
        depth=np.array([0,1,1,1,2,2,2,3],np.int32),born=np.arange(8,dtype=np.int32),
        degree=np.array([4,3,2,0,1,0,0,0],np.int32),prior=np.array([0,.6,.2,.1,.7,.2,.9,1.]),
        boot=np.array([-.2,.1,-.1,.3,-.4,.2,.5,.3]),mass=np.array([1.,1.,1.,0.,1.,0.,0.,0.]),
        terminal=np.array([-1.,-1.,-1.,.5,-1.,-1.,-1.,1.]),roots=np.array([0],np.int32))
    def solve(i,budget,strength,regularizer):
        if data['terminal'][i]>=0:return 0. if data['terminal'][i]==.5 else -1.
        base=-data['boot'][i];mass=data['mass'][i]
        if not mass:return base
        children=[c for c in range(8) if data['parent'][c]==i and data['born'][c]<=budget]
        vals=[];weights=[];raw=[]
        for c in children:
            vals.append(-solve(c,budget,strength,regularizer));weights.append(data['prior'][c])
            raw.append((0. if data['terminal'][c]==.5 else 1.) if data['terminal'][c]>=0 else data['boot'][c])
        seen=sum(weights);rest=max(0,mass-seen)
        if rest>1e-15 and strength>0:
            base=np.clip(base+strength*(seen*base-np.dot(weights,raw))/(regularizer*mass+rest),-1.,1.)
        vals.append(base);weights.append(rest);v=np.array(vals);w=np.array(weights);hi=v[w>0].max()
        return hi+.1*np.log(np.sum(w*np.exp((v-hi)/.1))/mass)
    for budget in (0,1,4,8):
        for strength,reg in [(0,.1),(1,0),(1,.01),(1,.1),(1,1.)]:
            got=module.reduce(data,budget,strength,reg);expected=np.full((1,1968),-data['boot'][0])
            for c in range(1,4):
                if data['born'][c]<=budget:expected[0,data['move'][c]]=-solve(c,budget,strength,reg)
            np.testing.assert_allclose(got,expected,atol=3e-15,rtol=3e-15)
            if strength==0:np.testing.assert_array_equal(got,ref.reduce(data,budget,.1,.1))
    print('PASS moment-tail independent recursion, zero-strength identity, terminal raw values, unvisited mass and clipping')

if __name__=='__main__':test(load())
