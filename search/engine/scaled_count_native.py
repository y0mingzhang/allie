"""Count normalization: independent recursive check before long inference."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from .backup_native import ROOT


def load():
    import pybind11
    source=Path(__file__).with_name('scaled_count.cpp')
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,source.with_name('diff_backup.cpp'))}|dict(python=sys.version)
    folder=ROOT/'results/search-v1/runtime/scaled_count';folder.mkdir(exist_ok=True)
    target=folder/('_allie_scaled_count'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_scaled_count')


def test(module):
    from .diff_backup_native import load as original
    from scipy.special import logsumexp
    rng=np.random.default_rng(28577);n=70
    parent=np.array([-1]+[int(rng.integers(i)) for i in range(1,n)],np.int32);parent[1:4]=0
    prior=np.zeros(n);degree=np.array([sum(parent==i)+1 for i in range(n)],np.int32)
    for i in range(n):
        child=np.flatnonzero(parent==i)
        if len(child):prior[child]=rng.dirichlet(np.ones(len(child)+1))[:-1]
    d=dict(parent=parent,move=np.arange(n,dtype=np.int32),born=np.arange(n,dtype=np.int32),
        roots=np.array([0],np.int32),degree=degree,prior=prior,boot=rng.uniform(-1,1,n),mass=np.ones(n),terminal=np.full(n,-1.))
    d['terminal'][-2:]=[.5,1.];ids=np.arange(n,dtype=np.int32)[None,:]
    for budget in (0,13,70):
        for scale in (4.,16.,64.,256.):
            def recur(i):
                if d['terminal'][i]>=0:return (0. if d['terminal'][i]==.5 else -1.),1
                child=[j for j in range(n) if parent[j]==i and j<=budget]
                if not child:return -d['boot'][i],1
                vv=[recur(j) for j in child];count=1+sum(x[1] for x in vv)
                ps=np.r_[prior[child],max(0.,1-prior[child].sum())]
                values=np.array([-x[0] for x in vv]+[-d['boot'][i]])
                tau=.2*(1+(count-1)/scale)**-.5
                return tau*logsumexp(values/tau,b=ps),count
            expected=[-recur(int(j))[0] if j>0 and parent[j]==0 and j<=budget else -d['boot'][0] for j in ids[0]]
            got=module.Backup(d,budget,scale).reduce(np.log(.2),-.5,ids)[0]
            np.testing.assert_allclose(got[0],expected,atol=2e-14)
            if scale==16:np.testing.assert_array_equal(got,original().Backup(d,budget).reduce(np.log(.2),-.5,ids)[0])
    print('PASS independent count-scale recursion, terminal/budget semantics, exact scale16 identity',flush=True)


if __name__=='__main__':test(load())
