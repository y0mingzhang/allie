"""Native differentiated backup with independent torch/autograd reference."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from .backup_native import ROOT

def load():
    import pybind11
    source=Path(__file__).with_name('diff_backup.cpp');key=dict(source=hashlib.sha256(source.read_bytes()).hexdigest(),python=sys.version)
    folder=ROOT/'results/search-v1/runtime/diff_backup';folder.mkdir(exist_ok=True)
    target=folder/('_allie_diff_backup'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_diff_backup')

def test(module):
    import torch
    from .adaptive_temperature_native import load as reference_load
    reference=reference_load()
    rng=np.random.default_rng(718);n=80
    parent=np.array([-1]+[int(rng.integers(max(0,i-15),i)) for i in range(1,n)],np.int32)
    parent[1:4]=0
    depth=np.zeros(n,np.int32)
    for i in range(1,n):depth[i]=depth[parent[i]]+1
    degree=np.array([sum(parent==i)+1 for i in range(n)],np.int32)
    prior=np.zeros(n)
    for i in range(n):
        ch=np.flatnonzero(parent==i)
        if len(ch):prior[ch]=rng.dirichlet(np.ones(len(ch)+1))[:-1]
    data=dict(parent=parent,move=np.arange(n,dtype=np.int32),depth=depth,born=np.arange(n,dtype=np.int32),
        degree=degree,prior=prior,boot=rng.uniform(-.7,.7,n),mass=np.ones(n),terminal=np.full(n,-1.),roots=np.array([0],np.int32))
    data['terminal'][-2:]=[.5,1.];ids=np.arange(n,dtype=np.int32)[None,:]
    for budget in [0,12,80]:
        native=module.Backup(data,budget)
        def torch_backup(theta):
            def recur(i):
                if data['terminal'][i]>=0:return theta.sum()*0+(0. if data['terminal'][i]==.5 else -1.),1
                children=[c for c in range(n) if parent[c]==i and data['born'][c]<=budget]
                values=[];weights=[];desc=1
                for c in children:
                    v,k=recur(c);values.append(-v);weights.append(prior[c]);desc+=k
                base=theta.sum()*0-data['boot'][i]
                if not children:return base,1
                values.append(base);weights.append(max(0,1-sum(weights)))
                tau=torch.exp(theta[0]+theta[1]*np.log1p((desc-1)/16))
                value=tau*torch.logsumexp(torch.stack(values)/tau+torch.tensor(weights,dtype=torch.float64).log(),dim=0)
                return value,desc
            vals=[]
            for move in ids[0]:
                if move>0 and parent[move]==0 and data['born'][move]<=budget:vals.append(-recur(int(move))[0])
                else:vals.append(theta.sum()*0-data['boot'][0])
            return torch.stack(vals)
        for a,b in [(np.log(.1),0.),(np.log(.2),-.5),(np.log(.05),-.8)]:
            theta=torch.tensor([a,b],dtype=torch.float64,requires_grad=True)
            got=native.reduce(a,b,ids)
            value=torch_backup(theta);jac=torch.autograd.functional.jacobian(torch_backup,theta)
            np.testing.assert_allclose(got[0,0],value.detach().numpy(),atol=2e-14,rtol=2e-13)
            np.testing.assert_allclose(got[1:,0].T,jac.numpy(),atol=2e-14,rtol=2e-12)
            assert torch.autograd.gradcheck(torch_backup,(theta,),eps=1e-5,atol=1e-7,rtol=1e-5)
            for j in range(2):
                ap=np.array([a,b]);am=ap.copy();ap[j]+=1e-5;am[j]-=1e-5
                finite=(native.reduce(*ap,ids)[0]-native.reduce(*am,ids)[0])/2e-5
                np.testing.assert_allclose(finite,got[j+1],atol=1e-8,rtol=1e-6)
            if b in (0.,-.5):
                ref=reference.reduce(data,budget,0 if b==0 else 3,np.exp(a),16)[:,ids[0]]
                np.testing.assert_allclose(got[0],ref,atol=2e-14,rtol=2e-13)
    edge=dict(parent=np.array([-1,0,1],np.int32),born=np.arange(3,dtype=np.int32),
        roots=np.array([0],np.int32),move=np.array([0,1,2],np.int32),degree=np.array([1,1,0],np.int32),
        prior=np.array([0.,1.,1.]),boot=np.array([-.2,-.8,-1.]),mass=np.array([1.,1.,0.]),terminal=np.full(3,-1.))
    extreme=module.Backup(edge,3).reduce(np.log(.01),-1.,np.array([[1]],np.int32))
    np.testing.assert_allclose(extreme[:,0,0],[1.,0.,0.],atol=1e-14)
    assert np.isfinite(extreme).all()
    print('PASS independent float64 torch/autograd, gradcheck, finite differences, constant/subtree identity, budget, terminal and unseen branches')

if __name__=='__main__':test(load())
