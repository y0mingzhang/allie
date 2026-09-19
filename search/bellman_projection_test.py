"""Independent dense Gaussian solve and partial-tree guards for critic projection."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import numpy as np


def load():
    root=Path(__file__).resolve().parents[1]
    sources=[root/'search/engine'/s for s in ('bellman_projection.cpp','diff_backup.cpp')]
    out=root/'results/search-v1/runtime/cpu-audit/bellman'
    out.mkdir(exist_ok=True)
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    target=out/('_allie_bellman_projection'+sysconfig.get_config_var('EXT_SUFFIX'))
    stamp=out/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        import torch
        headers=Path(torch.__file__).parent/'include'
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
                        '-I'+str(headers),'-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(out))
    return importlib.import_module('_allie_bellman_projection')


def test(module):
    rng=np.random.default_rng(417857)
    for n in (9,40,90):
        par=np.array([-1]+[rng.integers(i) for i in range(1,n)],np.int32)
        prior=np.zeros(n)
        degree=np.zeros(n,np.int32)
        for i in range(n):
            ch=np.flatnonzero(par==i);degree[i]=len(ch)+int(i%3==0)
            if len(ch):prior[ch]=rng.dirichlet(np.ones(degree[i]))[:len(ch)]
        terminal=np.full(n,-1.);terminal[-2:]=[.5,1.]
        data=dict(parent=par,move=np.arange(n,dtype=np.int32),born=np.arange(n,dtype=np.int32),
                  degree=degree,prior=prior,boot=rng.uniform(-1,1,n),mass=np.ones(n),terminal=terminal,
                  roots=np.array([0],np.int32))
        for budget in (n//2,n):
            active=np.arange(n)<=budget;y=-data['boot'].copy();y[terminal>=0]=np.where(terminal[terminal>=0]==.5,0.,-1.)
            free=np.flatnonzero(active&(terminal<0));fixed=np.flatnonzero(active&(terminal>=0))
            for strength in (0.,.3,1.,3.,100.):
                mat=np.eye(n);rhs=y.copy()
                if strength:
                    for i in np.flatnonzero(active&(terminal<0)):
                        ch=np.flatnonzero((par==i)&active)
                        if not len(ch):continue
                        r=0. if len(ch)==degree[i] else max(0.,1-prior[ch].sum())
                        a=np.zeros(n);a[i]=1.;a[ch]=prior[ch]
                        precision=1/(1/strength+r*r)
                        mat+=precision*np.outer(a,a);rhs+=precision*a*r*y[i]
                want=y.copy();want[free]=np.linalg.solve(mat[np.ix_(free,free)],rhs[free]-mat[np.ix_(free,fixed)]@y[fixed])
                got=module.Projection(data,budget).project(strength)['mean']
                np.testing.assert_allclose(got[active],want[active],rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(got[fixed],y[fixed])
                if strength==0:np.testing.assert_array_equal(got,y)
    # Future nodes cannot affect a partial tree; lambda0 exactly recovers backup.
    base=module.Projection(data,n//2);base.project(0.)
    ids=np.arange(n,dtype=np.int32)[None,:]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'results/search-v1/runtime/cpu-audit'))
    import _allie_scaled_count
    np.testing.assert_array_equal(base.reduce(np.log(.2),-.5,ids),
        _allie_scaled_count.Backup(data,n//2,16.).reduce(np.log(.2),-.5,ids))
    altered=data.copy();altered['boot']=data['boot'].copy();altered['boot'][n//2+1:]=rng.normal(size=n-n//2-1)*100
    a,b=module.Projection(data,n//2),module.Projection(altered,n//2)
    a.project(3.);b.project(3.)
    np.testing.assert_array_equal(a.reduce(np.log(.2),-.5,ids),b.reduce(np.log(.2),-.5,ids))
    print('PASS dense normal equations at3 sizes/2budgets/5strengths, exact terminal values, zero identity, future-node exclusion',flush=True)


if __name__=='__main__':test(load())
