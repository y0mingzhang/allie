"""Selection ablation compiler and default-path regression test."""
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
    sources=[Path(__file__).with_name(n) for n in ('selection.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/selection';folder.mkdir(exist_ok=True)
    target=folder/('_allie_selection'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_selection')
    from .native_board import MOVES
    module.initialize(MOVES);return module


def test(module):
    from .compact_native import load as load_old
    from .native_board import MOVES
    old=load_old();rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][:3]
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
    z=oracle(prefixes);sims=[64]*len(prefixes);cp=[1.25]*len(prefixes)
    a=module.Tree(prefixes,z,sims,cp,0,0.);b=old.Tree(prefixes,z,sims,cp)
    for _ in range(64):
        x,y=a.select(),b.select();assert x==y
        if x:p=oracle(x);a.update(p);b.update(p)
    for key,value in a.compact().items():np.testing.assert_array_equal(value,b.compact()[key])
    assert a.stats()==b.stats()
    # On the second simulation compare the selected root action against an
    # independent one-step PUCT calculation, including the mover-perspective sign.
    for mode,penalty in [(1,0.),(2,0.),(1,.2)]:
        tree=module.Tree(prefixes[:1],z[:1],[2],[1.25],mode,penalty)
        first=tree.select();tree.update(oracle(first));snap=tree.snapshot()[0]
        ids,visits,q,prior=map(np.asarray,snap);data=tree.compact();root=int(data['roots'][0])
        root_mean=-float(q[visits>0][0]);unseen=-float(data['boot'][root]) if mode==1 else -root_mean
        unseen=max(-1.,unseen-penalty*np.sqrt(prior[visits>0].sum()))
        factor=(np.log((1+19652.+1)/19652.)+1.25)
        score=np.where(visits>0,q,unseen)+factor*prior/(1+visits)
        expected=int(ids[score.argmax()])+378;next_prefix=tree.select()[0]
        assert next_prefix[len(prefixes[0])]==expected,(mode,penalty,expected,next_prefix)
    print('PASS default-path exact identity and independent FPU sign/selection checks',flush=True)


if __name__=='__main__':test(load())
