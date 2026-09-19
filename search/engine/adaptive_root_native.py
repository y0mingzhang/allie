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
    sources=[Path(__file__).with_name(n) for n in ('adaptive_root.cpp','threadforest.cpp','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/adaptive_root';folder.mkdir(exist_ok=True)
    target=folder/('_allie_adaptive_root'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-fopenmp','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_adaptive_root')
    from .native_board import MOVES
    module.initialize(MOVES);return module

def test(module):
    from .handleforest_native import load as old_load
    from .forest_native import canonical
    old=old_load();rows=json.loads((ROOT/'results/search-v1/aug-tune-v1/sample.json').read_text())['positions'][:4]
    prefixes=[r['prefix'] for r in rows];n=len(rows);sims=[64,128,256,0];cp=[2.5]*n;beta=[2.,4.,6.,8.];block=16
    def oracle(ps):
        return np.array([np.random.default_rng(int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16)).normal(size=2432).astype(np.float32) for p in ps])
    z=oracle(prefixes)
    def apply(t,h,known):
        ps=[]
        for node,parent,move,length in h:
            p=known[int(parent)]+[int(move)];assert len(p)==length;known[int(node)]=p;ps.append(p)
        if ps:t.update(oracle(ps))
    reference=old.Tree(prefixes,z,sims,cp);known={i:p for i,p in enumerate(prefixes)}
    while not reference.done:apply(reference,reference.select(),known)
    for mix in (0.,.5,1.):
        tree=module.Tree(prefixes,z,sims,cp,2,mix,beta,block);known={i:p for i,p in enumerate(prefixes)};checks=0
        def expected():
            out=[]
            for owner,(snap,back) in enumerate(zip(tree.snapshot(),tree.backups([.1])[0])):
                moves,counts,q,prior=map(np.asarray,snap);lookup=dict(zip(*back));q=np.array([lookup[int(m)] for m in moves]);counts=counts.copy()
                pi=np.exp(np.log(prior)+beta[owner]*q-(np.log(prior)+beta[owner]*q).max());pi/=pi.sum()
                w=np.sqrt(np.maximum(0.,(1-mix)*prior*(1-prior)+mix*pi*(1-pi)));pulls=[[] for _ in prior]
                current=int(counts.sum())
                for pull in range(current+1,min(sims[owner],current+block)+1):
                    j=int(np.argmax(w/(1+counts)));pulls[j].append(pull);counts[j]+=1
                out.extend((owner,j,p) for j,p in enumerate(pulls))
            return out
        assert list(tree.planned)==expected();checks+=1
        while not tree.done:
            ex=expected() if tree.stage_done else None
            h=tree.select()
            if ex is not None:assert list(tree.planned)==ex;checks+=1
            apply(tree,h,known)
            # Independently reduced root Q must agree with the maintained values.
            if tree.stage_done:
                vals=np.array(tree.values);data=tree.compact();reduced=module.reduce(data,1000,.1,.1)
                for node,parent in enumerate(data['parent']):
                    if parent in data['roots']:
                        owner=list(data['roots']).index(parent);move=data['move'][node]
                        np.testing.assert_allclose(-vals[node],reduced[owner,move],rtol=0,atol=1e-12)
        if mix==0.:
            a=canonical(reference.compact());b=canonical(tree.compact());assert a.keys()==b.keys()
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            for a,b in zip(reference.snapshot(),tree.snapshot()):
                for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(reference.evals,tree.evals)
        print('PASS adaptive mixture',mix,'independent block allocations',checks,'maintained values; zero mix exact static control',flush=True)
if __name__=='__main__':test(load())
