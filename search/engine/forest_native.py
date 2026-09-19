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
    sources=[Path(__file__).with_name(n) for n in ('forest.cpp','coverage.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/forest';folder.mkdir(exist_ok=True)
    target=folder/('_allie_forest'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_forest')
    from .native_board import MOVES
    module.initialize(MOVES);return module

def canonical(data):
    keys=[]
    for i,p in enumerate(data['parent']):
        keys.append((('root',int(i)),) if p<0 else keys[int(p)]+(int(data['move'][i]),))
    return {key:tuple(data[k][i].item() for k in ('move','depth','born','degree','prior','boot','mass','terminal')) for i,key in enumerate(keys)}

def test(module):
    from .native_board import MOVES
    rows=json.loads((ROOT/'results/search-v1/aug-tune-v1/sample.json').read_text())['positions']
    forced=next(r for r in rows if len(r['legal'])==1)
    mapping={m:378+i for i,m in enumerate(MOVES)}
    prefixes=[r['prefix'] for r in rows[:4]]+[forced['prefix'],rows[0]['prefix'][:11]+[mapping[m] for m in ('f2f3','e7e5','g2g4')]]
    def oracle(ps):
        out=[]
        for p in ps:
            seed=int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)
            z=np.random.default_rng(seed).normal(size=2432).astype(np.float32)
            if p==prefixes[-1]:z[mapping['d8h4']]=10.
            out.append(z)
        return np.array(out)
    report=[]
    for depth in (100,2):
        sims=[0,64,256,1000,80,1000];cp=[2.5]*len(prefixes);z=oracle(prefixes)
        seq=module.Reference(prefixes,z,sims,cp,0,0.,2.)
        par=module.Tree(prefixes,z,sims,cp)
        seq.max_search_depth=par.max_search_depth=depth
        snapshots={}
        for i in range(1,max(sims)+1):
            p=seq.select()
            if p:seq.update(oracle(p))
            if i in (64,256,1000):
                snapshots[i]=(module.reduce(seq.compact(),i,.1,.1),seq.evals)
        rounds=0
        while not par.done:
            p=par.select()
            if p:par.update(oracle(p))
            rounds+=1
        a=canonical(seq.compact());b=canonical(par.compact())
        assert a.keys()==b.keys()
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        for sa,sb in zip(seq.snapshot(),par.snapshot()):
            for x,y in zip(sa,sb):np.testing.assert_allclose(x,y,rtol=0,atol=1e-14)
        for budget,(q,cost) in snapshots.items():
            np.testing.assert_allclose(q,module.reduce(par.compact(),budget,.1,.1),rtol=0,atol=1e-14)
            np.testing.assert_array_equal(cost,par.prefix_evals(budget))
        for key,value in seq.stats().items():
            if key!='requests':assert par.stats()[key]==value,(key,par.stats(),seq.stats())
        report.append(dict(depth=depth,rounds=rounds,sequential=seq.stats(),parallel=par.stats()))
    empty=module.Tree(prefixes[:1],oracle(prefixes[:1]),[0],[2.5]);assert empty.done
    print(json.dumps(dict(passed=True,cases=report,checks='All nodes/births/values; terminal and depth-limit visits; mixed and zero budgets; 64/256/1000 reductions and NN counts; forced root'),indent=2),flush=True)

if __name__=='__main__':test(load())
