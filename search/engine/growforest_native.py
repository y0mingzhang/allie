"""Compile an isolated resumable root-coverage forest."""
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
    sources=[Path(__file__).with_name(n) for n in ('growforest.cpp','threadforest.cpp','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/growforest';folder.mkdir(exist_ok=True)
    target=folder/('_allie_growforest'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-fopenmp','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_growforest')
    from .native_board import MOVES
    module.initialize(MOVES);return module

def test(module):
    from .threadforest_native import load as reference_load
    from .diff_backup_native import load as backup_load
    ref=reference_load();backup=backup_load()
    rows=json.loads((ROOT/'results/search-v1/aug-tune-v1/sample.json').read_text())['positions'][:24]
    prefixes=[r['prefix'] for r in rows];n=len(rows)
    def oracle(ps):
        return np.array([np.random.default_rng(int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)).normal(size=2432).astype(np.float32) for p in ps])
    def advance(t,known):
        while not t.done:
            h=t.select();ps=[]
            for node,parent,move,length in h:
                pp=known[int(parent)]+[int(move)];assert len(pp)==length;known[int(node)]=pp;ps.append(pp)
            if ps:t.update(oracle(ps))
    z=oracle(prefixes);budgets=[0,128,256,512,1000,256]*4;cp=[2.5]*n
    a=ref.Tree(prefixes,z,budgets,cp,4);ka={i:p for i,p in enumerate(prefixes)};advance(a,ka)
    b=module.Tree(prefixes,z,[min(v,128) for v in budgets],cp,4);kb={i:p for i,p in enumerate(prefixes)};advance(b,kb)
    before=np.array(b.evals);b.grow(budgets);advance(b,kb)
    np.testing.assert_array_equal(a.evals,b.evals)
    da,db=a.compact(),b.compact()
    def semantic(data):
        keys=[]
        for i,par in enumerate(data['parent']):
            keys.append(('root',int(i)) if par<0 else keys[par]+(int(data['move'][i]),))
        return {key:tuple(data[s][i] for s in ('move','depth','born','degree','prior','boot','mass','terminal')) for i,key in enumerate(keys)}
    assert semantic(da)==semantic(db)
    ids=np.zeros((n,max(len(r['legal']) for r in rows)),np.int32)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378
    np.testing.assert_array_equal(backup.Backup(da,1000).reduce(np.log(.2),-.5,ids),backup.Backup(db,1000).reduce(np.log(.2),-.5,ids))
    for name in ('evaluated_leaves','simulations','terminal_visits'):
        assert a.stats()[name]==b.stats()[name]
    b.grow(budgets);assert b.done
    try:b.grow([0]*n)
    except ValueError:pass
    else:raise AssertionError('shrinking accepted')
    print('PASS mixed zero/128/256/512/1000 staged vs one-shot: semantic tree, birth records, node counts, exact backup+gradients; no-op and shrinking guards',flush=True)

if __name__=='__main__':test(load())
