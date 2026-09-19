"""Player/opponent backup with a separate recursive reference and parity tests."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from .backup_native import ROOT


def load():
    import pybind11
    source=Path(__file__).with_name('perspective_backup.cpp')
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,source.with_name('diff_backup.cpp'))}|dict(python=sys.version)
    folder=ROOT/'results/search-v1/runtime/perspective_backup';folder.mkdir(exist_ok=True)
    target=folder/('_allie_perspective_backup'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_perspective_backup')


def test(module):
    from scipy.special import logsumexp
    from .diff_backup_native import load as original
    rng=np.random.default_rng(19676);n=80
    parent=np.array([-1]+[int(rng.integers(i)) for i in range(1,n)],np.int32);parent[1:4]=0
    degree=np.array([sum(parent==i)+1 for i in range(n)],np.int32);prior=np.zeros(n)
    for i in range(n):
        child=np.flatnonzero(parent==i)
        if len(child):prior[child]=rng.dirichlet(np.ones(len(child)+1))[:-1]
    data=dict(parent=parent,move=np.arange(n,dtype=np.int32),born=np.arange(n,dtype=np.int32),
        roots=np.array([0],np.int32),degree=degree,prior=prior,boot=rng.uniform(-1,1,n),mass=np.ones(n),terminal=np.full(n,-1.))
    data['terminal'][-2:]=[.5,1.];ids=np.arange(n,dtype=np.int32)[None,:]
    for budget in (0,13,80):
        for own,opp in ((1,1),(.5,1),(2,1),(1,.5),(1,2),(1,np.inf),(np.inf,1)):
            def recur(i,turn):
                if data['terminal'][i]>=0:return (0. if data['terminal'][i]==.5 else -1.),1
                child=[j for j in range(n) if parent[j]==i and data['born'][j]<=budget]
                if not child:return -data['boot'][i],1
                vv=[recur(j,1-turn) for j in child];count=1+sum(x[1] for x in vv)
                ps=np.r_[prior[child],max(0.,1-prior[child].sum())]
                values=np.array([-x[0] for x in vv]+[-data['boot'][i]])
                tau=.2*(1+(count-1)/16)**-.5*(opp if turn else own)
                return (float(ps@values) if np.isinf(tau) else tau*logsumexp(values/tau,b=ps)),count
            expected=[-recur(int(j),1)[0] if j>0 and parent[j]==0 and j<=budget else -data['boot'][0] for j in ids[0]]
            got=module.Backup(data,budget).reduce(own,opp,ids)
            np.testing.assert_allclose(got[0],expected,atol=2e-14)
            if own==opp==1:np.testing.assert_allclose(got,original().Backup(data,budget).reduce(np.log(.2),-.5,ids)[0],atol=2e-14)
    # Changing only opponent selectivity affects a two-ply root tree;
    # changing own selectivity cannot, since own internal nodes are leaves.
    edge=dict(parent=np.array([-1,0,1,1],np.int32),born=np.arange(4,dtype=np.int32),roots=np.array([0],np.int32),
        move=np.arange(4,dtype=np.int32),degree=np.array([1,2,0,0],np.int32),prior=np.array([0.,1.,.5,.5]),
        boot=np.array([0.,0.,-.8,.8]),mass=np.array([1.,1.,0.,0.]),terminal=np.full(4,-1.))
    b=module.Backup(edge,4);idx=np.array([[1]],np.int32)
    assert b.reduce(1,.5,idx)[0,0] < b.reduce(1,2,idx)[0,0]
    np.testing.assert_array_equal(b.reduce(.5,1,idx),b.reduce(2,1,idx))
    np.testing.assert_allclose(b.reduce(1,np.inf,idx),0,atol=1e-14)
    edge['terminal'][2:]=1.;np.testing.assert_allclose(module.Backup(edge,4).reduce(1,1,idx),-1.,atol=1e-14)
    print('PASS perspective recursion, unchanged baseline, correct own/opponent parity, full coverage, expectation limit and terminal signs',flush=True)


if __name__=='__main__':test(load())
