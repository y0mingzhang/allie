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
    sources=[Path(__file__).with_name(n) for n in ('soft_selection.cpp','compact.cpp','backups.cpp','board.cpp','mcts_native.hpp')]
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    folder=ROOT/'results/search-v1/runtime/soft-selection';folder.mkdir(exist_ok=True)
    target=folder/('_allie_softselection'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            '-I'+str(ROOT/'vendor/chess-library/include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key,indent=2)+'\n')
    sys.path.insert(0,str(folder));module=importlib.import_module('_allie_softselection')
    from .native_board import MOVES
    module.initialize(MOVES);return module



def test(module):
    from .selection_native import load as load_reference
    from .native_board import MOVES
    from .compact_native import load as compact_module
    ref=load_reference();reducer=compact_module()
    rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][:3]
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
    z=oracle(prefixes);sims=[96]*len(prefixes);cp=[2.5]*len(prefixes)
    a=module.Tree(prefixes,z,sims,cp,1,0.,False,.1);b=ref.Tree(prefixes,z,sims,cp,1,0.)
    for _ in range(96):
        x,y=a.select(),b.select();assert x==y
        if x:p=oracle(x);a.update(p);b.update(p)
    for key,v in a.compact().items():np.testing.assert_array_equal(v,b.compact()[key])
    assert a.stats()==b.stats()
    # Recompute every node's Bellman value independently from the exported tree.
    t=module.Tree(prefixes,z,sims,cp,1,0.,True,.1)
    for step in range(96):
        x=t.select()
        if x:t.update(oracle(x))
        d=t.compact();v=np.zeros(len(d['parent'])); children=[[] for _ in v]
        for j,parent in enumerate(d['parent']):
            if parent>=0:children[parent].append(j)
        for j in range(len(v)-1,-1,-1):
            if d['terminal'][j]>=0:
                v[j]=0 if d['terminal'][j]==.5 else -1.;continue
            qs=[-v[k] for k in children[j]];weights=[d['prior'][k] for k in children[j]]
            rest=max(0.,d['mass'][j]-sum(weights))
            if rest>0:qs.append(-d['boot'][j]);weights.append(rest)
            if not qs:v[j]=-d['boot'][j];continue
            hi=max(qs);v[j]=hi+.1*np.log(np.dot(weights,np.exp((np.array(qs)-hi)/.1))/sum(weights))
        np.testing.assert_allclose(t.values,v,atol=3e-12,rtol=0)
        q=reducer.reduce(d,step+1,.1,.1)
        for i,(moves,scores) in enumerate(t.backups([.1])[0]):np.testing.assert_allclose(q[i,moves],scores,atol=3e-12,rtol=0)
    # Independent PUCT action choice immediately after one expansion.
    t=module.Tree(prefixes[:1],z[:1],[2],[2.5],1,0.,True,.1)
    x=t.select();t.update(oracle(x));d=t.compact();root=int(d['roots'][0])
    ids,visits,_,prior=map(np.asarray,t.snapshot()[0]);q=np.full(len(ids),-d['boot'][root])
    for child in np.flatnonzero(d['parent']==root):q[np.where(ids==d['move'][child])[0][0]]=-t.values[child]
    expected=int(ids[np.argmax(q+(np.log((1+19652.+1)/19652.)+2.5)*prior/(1+visits))])+378
    assert t.select()[0][len(prefixes[0])]==expected
    print('PASS disabled identity, every-node incremental Bellman values including terminals, root output equivalence and independent PUCT action',flush=True)


if __name__=='__main__':test(load())
