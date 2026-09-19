"""Independent factor-subset and subtree solves for projection ablations."""
import hashlib, importlib, json, subprocess, sys, sysconfig
from pathlib import Path
import numpy as np
from .bellman_projection_test import load as full_load


def load():
    root=Path(__file__).resolve().parents[1]
    sources=[root/'search/engine'/s for s in ('bellman_modes.cpp','bellman_projection.cpp','diff_backup.cpp')]
    out=root/'results/search-v1/runtime/cpu-audit/bellman-modes';out.mkdir(exist_ok=True)
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    target=out/('_allie_bellman_modes'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=out/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        import torch
        tmp=target.with_suffix('.new');headers=Path(torch.__file__).parent/'include'
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),'-I'+str(headers),
                        '-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(out));return importlib.import_module('_allie_bellman_modes')


def dense(data,budget,strength,mode,subroot=None):
    par=data['parent'];n=len(par);active=data['born']<=budget
    if subroot is not None:
        descendants=np.zeros(n,bool);descendants[subroot]=True
        for i in range(subroot+1,n):descendants[i]=par[i]>=0 and descendants[par[i]]
        active &= descendants
    y=-data['boot'].copy();terminal=data['terminal']>=0;y[terminal]=np.where(data['terminal'][terminal]==.5,0.,-1.)
    free=np.flatnonzero(active&~terminal);fixed=np.flatnonzero(active&terminal)
    matrix=np.eye(n);rhs=y.copy()
    if strength:
        for i in np.flatnonzero(active&~terminal):
            if mode==1 and par[i]>=0:continue
            if mode==2 and par[i]<0:continue
            ch=np.flatnonzero((par==i)&active)
            if not len(ch):continue
            p=data['prior'][ch]/data['mass'][i]
            r=0. if len(ch)==data['degree'][i] else max(0.,1-p.sum())
            a=np.zeros(n);a[i]=1.;a[ch]=p;w=1/(1/strength+r*r)
            matrix+=w*np.outer(a,a);rhs+=w*a*r*y[i]
    result=y.copy();result[free]=np.linalg.solve(matrix[np.ix_(free,free)],rhs[free]-matrix[np.ix_(free,fixed)]@y[fixed])
    return result,active


def test(native):
    rng=np.random.default_rng(48102);reference=full_load()
    for n in (11,40):
        parent=np.array([-1,-1]+[rng.integers(i) for i in range(2,n)],np.int32)
        prior=np.zeros(n);degree=np.zeros(n,np.int32)
        for i in range(n):
            ch=np.flatnonzero(parent==i);degree[i]=len(ch)+int(i%2==0)
            if len(ch):prior[ch]=rng.dirichlet(np.ones(degree[i]))[:len(ch)]
        data=dict(parent=parent,roots=np.array([0,1],np.int32),move=np.arange(n,dtype=np.int32),
                  born=np.r_[0,0,np.arange(2,n)].astype(np.int32),degree=degree,prior=prior,
                  boot=rng.uniform(-1,1,n),mass=np.ones(n),terminal=np.r_[np.full(n-2,-1.),.5,1.])
        for budget in (n//2,n):
            for strength in (0.,.3,3.,10.):
                a=reference.Projection(data,budget);want=a.project(strength)['mean']
                b=native.Modes(data,budget);got=b.project(strength,0)['mean']
                np.testing.assert_array_equal(got,want)
                for mode in (1,2):
                    got=b.project(strength,mode)['mean'];want,active=dense(data,budget,strength,mode)
                    np.testing.assert_allclose(got[active],want[active],atol=1e-12,rtol=1e-12)
                got=b.project(strength,3)['mean']
                for i in np.flatnonzero(data['born']<=budget):
                    want,_=dense(data,budget,strength,0,subroot=i)
                    np.testing.assert_allclose(got[i],want[i],atol=1e-12,rtol=1e-12)
    print('PASS full bit identity, root-only/no-root factor dense solves, upward-only per-subtree dense solves,2 forests/2budget cuts/4strengths',flush=True)


if __name__=='__main__':test(load())
