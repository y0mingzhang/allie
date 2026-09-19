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
    sources=[Path(__file__).with_name(n) for n in ('adaptive_temperature.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/adaptive_temperature';folder.mkdir(exist_ok=True)
    target=folder/('_allie_adaptive_temperature'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-fopenmp','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_adaptive_temperature')
    from .native_board import MOVES
    return module

def test(module):
    from .compact_native import load as reference_load
    reference=reference_load()
    data=dict(parent=np.array([-1,0,0,0,1,1,2,4],np.int32),
        move=np.array([-1,11,12,13,21,22,31,41],np.int32),
        depth=np.array([0,1,1,1,2,2,2,3],np.int32),
        born=np.arange(8,dtype=np.int32),degree=np.array([4,3,2,0,1,0,0,0],np.int32),
        prior=np.array([0,.6,.2,.1,.7,.2,.9,1.]),
        boot=np.array([-.2,.1,-.1,.3,-.4,.2,.5,.3]),
        mass=np.array([1.,1.,1.,0.,1.,0.,0.,0.]),
        terminal=np.array([-1.,-1.,-1.,.5,-1.,-1.,-1.,1.]),
        roots=np.array([0],np.int32))
    def solve(i,budget,mode,tau,scale):
        if data['terminal'][i]>=0:return (0. if data['terminal'][i]==.5 else -1.),1
        base=-data['boot'][i];mass=data['mass'][i]
        if mass==0:return base,1
        children=[c for c in range(len(data['parent'])) if data['parent'][c]==i and data['born'][c]<=budget]
        vals=[];weights=[];desc=1
        for c in children:
            v,n=solve(c,budget,mode,tau,scale);vals.append(-v);weights.append(data['prior'][c]);desc+=n
        rest=max(0,mass-sum(weights));vals.append(base);weights.append(rest)
        v=np.array(vals);w=np.array(weights);t=tau;dep=max(data['depth'][i],1)
        if mode==1:t*=np.sqrt(dep)
        if mode==2:t/=np.sqrt(dep)
        if mode==3:t*=np.sqrt(scale/(scale+desc-1))
        if mode==4:
            avg=w@v/mass;variance=max(0,w@(v*v)/mass-avg*avg);t=np.clip(tau*np.sqrt(variance),.01,.25)
        hi=v[w>0].max();value=hi+t*np.log(np.sum(w*np.exp((v-hi)/t))/mass)
        return value,desc
    for budget in (0,1,4,8):
        for mode,tau,scale in [(0,.05,16),(0,.1,16),(0,.2,16),(1,.05,16),(2,.2,16),(3,.2,16),(4,.5,16)]:
            got=module.reduce(data,budget,mode,tau,scale)
            expected=np.full((1,1968),-data['boot'][0])
            for c in range(1,4):
                if data['born'][c]<=budget:expected[0,data['move'][c]]=-solve(c,budget,mode,tau,scale)[0]
            np.testing.assert_allclose(got,expected,atol=2e-15,rtol=2e-15)
            if mode==0:np.testing.assert_array_equal(got,reference.reduce(data,budget,tau,tau))
    print('PASS adaptive-temperature independent recursion, constant identity, terminal/unvisited/budget handling')

if __name__=='__main__':test(load())
