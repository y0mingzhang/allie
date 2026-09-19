"""Odd utility and critic-dependent-temperature backups with independent reference."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from .backup_native import ROOT

def load():
    import pybind11
    source=Path(__file__).with_name('geometry_backup.cpp');key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,source.with_name('diff_backup.cpp')]}|dict(python=sys.version)
    folder=ROOT/'results/search-v1/runtime/geometry_backup';folder.mkdir(exist_ok=True)
    target=folder/('_allie_geometry_backup'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_geometry_backup')

def test(module):
    from scipy.special import logsumexp
    from .diff_backup_native import load as original
    rng=np.random.default_rng(743);n=80
    parent=np.array([-1]+[int(rng.integers(i)) for i in range(1,n)],np.int32);parent[1:4]=0
    degree=np.array([sum(parent==i)+1 for i in range(n)],np.int32);prior=np.zeros(n)
    for i in range(n):
        cc=np.flatnonzero(parent==i)
        if len(cc):prior[cc]=rng.dirichlet(np.ones(len(cc)+1))[:-1]
    data=dict(parent=parent,move=np.arange(n,dtype=np.int32),born=np.arange(n,dtype=np.int32),
        roots=np.array([0],np.int32),degree=degree,prior=prior,boot=rng.uniform(-1,1,n),mass=np.ones(n),terminal=np.full(n,-1.))
    data['terminal'][-2:]=[.5,1.];ids=np.arange(n,dtype=np.int32)[None,:]
    for budget in (0,13,80):
        for kind,strength in ((0,0.),(1,.95),(2,3.),(3,.5),(3,-.5)):
            g=lambda x:np.arctanh(strength*x)/np.arctanh(strength) if kind==1 else np.arcsinh(strength*x)/np.arcsinh(strength) if kind==2 else x
            inv=lambda x:np.tanh(x*np.arctanh(strength))/strength if kind==1 else np.sinh(x*np.arcsinh(strength))/strength if kind==2 else x
            def recur(i):
                if data['terminal'][i]>=0:return (0. if data['terminal'][i]==.5 else -1.),1
                child=[j for j in range(n) if parent[j]==i and data['born'][j]<=budget]
                if not child:return g(-data['boot'][i]),1
                vv=[recur(j) for j in child];count=1+sum(x[1] for x in vv)
                ps=np.r_[prior[child],1-prior[child].sum()]
                values=np.array([-x[0] for x in vv]+[g(-data['boot'][i])])
                tau=.2*(1+(count-1)/16)**-.5
                if kind==3:tau*=(.05+1-data['boot'][i]**2)**strength
                return tau*logsumexp(values/tau,b=ps),count
            expected=[-inv(recur(int(j))[0]) if j>0 and parent[j]==0 and j<=budget else -data['boot'][0] for j in ids[0]]
            got=module.Backup(data,budget).reduce(kind,strength,ids)
            np.testing.assert_allclose(got[0],expected,atol=2e-14)
            assert np.max(np.abs(got))<=1+1e-12
            if kind==0:np.testing.assert_allclose(got,original().Backup(data,budget).reduce(np.log(.2),-.5,ids)[0],atol=2e-14)
            v=np.linspace(-1,1,501);np.testing.assert_allclose(inv(g(v)),v,atol=1e-14)
            np.testing.assert_allclose(g(-v),-g(v),atol=1e-14);assert (np.diff(g(v))>0).all()
    # Fully explored single path must preserve exact mate/draw values, not a clipped utility.
    edge=dict(parent=np.array([-1,0,1],np.int32),born=np.arange(3,dtype=np.int32),roots=np.array([0],np.int32),
        move=np.array([0,1,2],np.int32),degree=np.array([1,1,0],np.int32),prior=np.array([0.,1.,1.]),
        boot=np.array([-.2,-.8,-1.]),mass=np.array([1.,1.,0.]),terminal=np.array([-1.,-1.,1.]))
    for kind,strength in ((0,0.),(1,.95),(2,3.),(3,.5),(3,-.5)):
        np.testing.assert_allclose(module.Backup(edge,3).reduce(kind,strength,np.array([[1]],np.int32)),[[-1.]],atol=2e-14)
    print('PASS independent recursion, odd utility inverse, bounds, baseline identity, terminal and full-coverage semantics',flush=True)

if __name__=='__main__':test(load())
