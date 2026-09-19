"""Reverse-KL Bellman backup with independent constrained-solver reference."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from .backup_native import ROOT

def load():
    import pybind11
    source=Path(__file__).with_name('reverse_backup.cpp');key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,source.with_name('diff_backup.cpp')]}|dict(python=sys.version)
    folder=ROOT/'results/search-v1/runtime/reverse_backup';folder.mkdir(exist_ok=True)
    target=folder/('_allie_reverse_backup'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_reverse_backup')

def test(module):
    from scipy.optimize import minimize,brentq
    rng=np.random.default_rng(19684)
    def value(q,p,lam):
        p=np.asarray(p,float);p/=p.sum();q=np.asarray(q,float)
        maximum=q.max();gap=(maximum-q)/lam
        lower=np.max(p-gap)
        f=lambda logt:np.sum(p/(np.exp(logt)+gap))-1
        lt=brentq(f,np.log(lower),1e-10,xtol=1e-14)
        t=np.exp(lt);pi=p/(t+gap)
        np.testing.assert_allclose(pi.sum(),1.,atol=1e-12)
        return pi@q-lam*np.sum(p*np.log(p/pi))
    for k in (1,2,9):
        for lam in (.002,.1,1.,10.):
            q=rng.uniform(-1,1,k);p=rng.dirichlet(np.ones(k)*2)
            got=module.value(q,p,lam)
            np.testing.assert_allclose(got,value(q,p,lam),atol=1e-12)
            assert p@q-1e-12<=got<=q.max()+1e-12
            np.testing.assert_allclose(module.value(q,7*p,lam),got,atol=1e-13)
            if k>1 and lam>=.1:
                def objective(pi):return -(pi@q-lam*np.sum(p*np.log(p/pi)))
                def jac(pi):return -q-lam*p/pi
                opt=minimize(objective,p,jac=jac,method='SLSQP',bounds=[(1e-10,1)]*k,
                    constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(k))],options=dict(ftol=1e-12,maxiter=300))
                assert opt.success,opt
                np.testing.assert_allclose(got,-opt.fun,atol=1e-9)
    for prior in ([.2,.8],[1e-30,1.]):
        for lam in (.001,.1,1.):
            np.testing.assert_allclose(module.value([1.,-1.],prior,lam),value([1.,-1.],prior,lam),atol=1e-11)
    for lam in (.001,1.,1000.):
        np.testing.assert_allclose(module.value([.3,.3],[.25,.75],lam),.3,atol=1e-12)
    n=33
    parent=np.array([-1]+[int(rng.integers(i)) for i in range(1,n)],np.int32)
    degree=np.array([sum(parent==i)+1 for i in range(n)],np.int32)
    prior=np.zeros(n)
    for i in range(n):
        child=np.flatnonzero(parent==i)
        if len(child):prior[child]=rng.dirichlet(np.ones(len(child)+1))[:-1]
    data=dict(parent=parent,move=np.arange(n,dtype=np.int32),born=np.arange(n,dtype=np.int32),
        roots=np.array([0],np.int32),degree=degree,prior=prior,boot=rng.uniform(-.8,.8,n),
        mass=np.ones(n),terminal=np.full(n,-1.))
    data['terminal'][-2:]=[.5,1.]
    ids=np.arange(n,dtype=np.int32)[None,:]
    for budget in (0,7,33):
        for visited in (False,True):
            def recur(i):
                if data['terminal'][i]>=0:return (0. if data['terminal'][i]==.5 else -1.),1
                child=[j for j in range(n) if parent[j]==i and data['born'][j]<=budget]
                base=-data['boot'][i]
                if not child:return base,1
                result=[recur(j) for j in child];count=1+sum(t[1] for t in result)
                v=[-t[0] for t in result];p=[prior[j] for j in child]
                if not visited:v.append(base);p.append(1-sum(p))
                return value(v,p,.2*(1+(count-1)/16)**-.5),count
            expected=[]
            for j in ids[0]:
                expected.append(-recur(int(j))[0] if j>0 and parent[j]==0 and j<=budget else -data['boot'][0])
            got=module.Backup(data,budget).reduce(.2,-.5,ids,visited)
            np.testing.assert_allclose(got[0],expected,atol=1e-12)
    print('PASS constrained optimum, independent root solver, tiny priors, bounds, normalization, terminals, negamax, subtree counts, budget and unseen-mass variants',flush=True)

if __name__=='__main__':test(load())
