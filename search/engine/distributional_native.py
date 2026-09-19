"""Isolated native extension; does not replace any existing tree binary."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def load():
    import pybind11
    sources=[Path(__file__).with_name(n) for n in ('distributional.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};key['python']=sys.version
    folder=ROOT/'results/search-v1/runtime/distributional';folder.mkdir(exist_ok=True)
    target=folder/('_allie_distributional'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),'-I'+pybind11.get_include(),
            '-I'+sysconfig.get_path('include'),'-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_distributional')
    from .native_board import MOVES
    module.initialize(MOVES);return module


def test(module):
    from .backup_native import load as reference
    from .native_board import MOVE_ID
    old=reference();rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][:3]
    prefixes=[r['prefix'] for r in rows]+[rows[0]['prefix'][:11]+[MOVE_ID[m] for m in ('f2f3','e7e5','g2g4')]]
    def oracle(ps):
        out=[]
        for p in ps:
            seed=int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)
            z=np.random.default_rng(seed).normal(size=2432).astype(np.float32)
            if p==prefixes[-1]:z[MOVE_ID['d8h4']]=10
            out.append(z)
        return np.array(out)
    z=oracle(prefixes);a=module.Tree(prefixes,z,[64]*len(prefixes),[1.25]*len(prefixes))
    b=old.Tree(prefixes,z,[64]*len(prefixes),[1.25]*len(prefixes))
    for step in range(64):
        x,y=a.select(),b.select();assert x==y
        if x:pred=oracle(x);a.update(pred);b.update(pred)
        if step in (0,15,63):
            for distribution,actual,ref in zip(a.distribution(),a.snapshot(),b.snapshot()):
                ids,wdl=distribution;wdl=np.array(wdl);np.testing.assert_array_equal(ids,actual[0])
                np.testing.assert_allclose(wdl.sum(1),1.,atol=1e-12)
                np.testing.assert_allclose(wdl[:,0]-wdl[:,2],actual[2],atol=1e-12)
                for u,v in zip(actual,ref):np.testing.assert_array_equal(u,v)
            np.testing.assert_array_equal(a.evals,b.evals)
    assert a.stats()['terminal_visits']>0
    print('PASS full WDL backup conservation, original scalar Q and tree exact, terminal and node costs')


if __name__=='__main__':test(load())
