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
    sources=[Path(__file__).with_name(n) for n in ('handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/handleforest';folder.mkdir(exist_ok=True)
    target=folder/('_allie_handleforest'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_handleforest')
    from .native_board import MOVES
    module.initialize(MOVES);return module

def test(module):
    from .forest_native import load as old_load
    old=old_load();rows=json.loads((ROOT/'results/search-v1/aug-tune-v1/sample.json').read_text())['positions'][:12]
    prefixes=[r['prefix'] for r in rows]
    def oracle(ps):
        return np.array([np.random.default_rng(int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)).normal(size=2432).astype(np.float32) for p in ps])
    z=oracle(prefixes);sims=[256]*len(rows);cp=[2.5]*len(rows)
    a=old.Tree(prefixes,z,sims,cp);b=module.Tree(prefixes,z,sims,cp)
    known={i:p for i,p in enumerate(prefixes)};rounds=0
    while not a.done:
        ps=a.select();hs=b.select();reconstructed=[]
        for node,parent,move,length in hs:
            assert int(node) not in known
            p=known[int(parent)]+[int(move)];assert len(p)==length
            known[int(node)]=p;reconstructed.append(p)
        assert ps==reconstructed
        if ps:
            z=oracle(ps);a.update(z);b.update(z)
        rounds+=1
    assert b.done
    for key,value in a.compact().items():np.testing.assert_array_equal(value,b.compact()[key])
    assert a.stats()==b.stats()
    print('PASS handle queries reconstruct every exact prefix; node arrays and statistics identical; rounds',rounds,flush=True)

if __name__=='__main__':test(load())
