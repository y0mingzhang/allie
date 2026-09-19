"""Independent dense reference for unequal critic measurement variances."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from .bellman_projection_test import load as original_load


def load():
    root=Path(__file__).resolve().parents[1]
    sources=[root/'search/engine'/s for s in ('bellman_variance.cpp','bellman_projection.cpp','diff_backup.cpp')]
    out=root/'results/search-v1/runtime/cpu-audit/bellman-variance';out.mkdir(exist_ok=True)
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    target=out/('_allie_bellman_variance'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=out/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        import torch
        tmp=target.with_suffix('.new');headers=Path(torch.__file__).parent/'include'
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),'-I'+str(headers),
                        '-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(out));return importlib.import_module('_allie_bellman_variance')


def test(native):
    rng=np.random.default_rng(553940);original=original_load()
    for n in (12,55):
        parent=np.array([-1,-1]+[rng.integers(i) for i in range(2,n)],np.int32)
        prior=np.zeros(n);degree=np.zeros(n,np.int32)
        for i in range(n):
            ch=np.flatnonzero(parent==i);degree[i]=len(ch)+int(i%2==0)
            if len(ch):prior[ch]=rng.dirichlet(np.ones(degree[i]))[:len(ch)]
        data=dict(parent=parent,roots=np.array([0,1],np.int32),move=np.arange(n,dtype=np.int32),
                  born=np.r_[0,0,np.arange(2,n)].astype(np.int32),degree=degree,prior=prior,
                  boot=rng.uniform(-1,1,n),mass=np.ones(n),terminal=np.r_[np.full(n-2,-1.),.5,1.])
        for budget in (n//2,n):
            active=data['born']<=budget;terminal=data['terminal']>=0
            y=-data['boot'].copy();y[terminal]=np.where(data['terminal'][terminal]==.5,0.,-1.)
            free=np.flatnonzero(active&~terminal);fixed=np.flatnonzero(active&terminal)
            for strength in (0.,.3,3.,10.):
                a=native.Weighted(data,budget);b=original.Projection(data,budget)
                np.testing.assert_array_equal(a.project(strength,np.ones(n))['mean'],b.project(strength)['mean'])
                s=rng.uniform(.03,2,n);matrix=np.diag(1/s);rhs=y/s
                if strength:
                    for i in free:
                        ch=np.flatnonzero((parent==i)&active)
                        if not len(ch):continue
                        p=prior[ch];r=0. if len(ch)==degree[i] else max(0.,1-p.sum())
                        row=np.zeros(n);row[i]=1.;row[ch]=p;w=1/(1/strength+r*r*s[i])
                        matrix+=w*np.outer(row,row);rhs+=w*row*r*y[i]
                want=y.copy();want[free]=np.linalg.solve(matrix[np.ix_(free,free)],rhs[free]-matrix[np.ix_(free,fixed)]@y[fixed])
                got=a.project(strength,s)['mean']
                np.testing.assert_allclose(got[active],want[active],atol=1e-12,rtol=1e-12)
                np.testing.assert_array_equal(got[fixed],y[fixed])
    print('PASS unequal-variance dense solves, uniform bit identity, terminal and budget guards',flush=True)


if __name__=='__main__':test(load())
