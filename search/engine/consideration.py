"""Human consideration-set model: draw K candidates, pick best searched value."""
import itertools,json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .fit_policy import fit,loss_gradient
from .analyze_august import means

class Consideration:
    def __init__(self,z,q,mask):
        self.order=np.argsort(np.where(mask,q,np.inf),axis=1,kind='stable')
        self.inverse=np.argsort(self.order,axis=1)
        self.z=np.take_along_axis(z,self.order,axis=1)
        self.q=np.take_along_axis(np.where(mask,q,np.inf),self.order,axis=1)
        self.mask=np.take_along_axis(mask,self.order,axis=1)
        n,k=z.shape;idx=np.broadcast_to(np.arange(k),(n,k))
        ends=np.concatenate([self.q[:,1:]!=self.q[:,:-1],np.ones((n,1),bool)],axis=1)
        starts=np.concatenate([np.ones((n,1),bool),self.q[:,1:]!=self.q[:,:-1]],axis=1)
        self.end=np.minimum.accumulate(np.where(ends,idx,k)[:,::-1],axis=1)[:,::-1]
        self.start=np.maximum.accumulate(np.where(starts,idx,0),axis=1)
    def policy(self,theta):
        alpha,logk,lapse=theta;k=np.exp(logk)
        p=softmax(np.where(self.mask,alpha*self.z,-np.inf),axis=1)
        dp=p*(self.z-(p*self.z).sum(1,keepdims=True))
        cp=np.cumsum(p,axis=1);cd=np.cumsum(dp,axis=1)
        upper=np.take_along_axis(cp,self.end,axis=1);du=np.take_along_axis(cd,self.end,axis=1)
        lower=np.take_along_axis(np.pad(cp,((0,0),(1,0))),self.start,axis=1)
        dl=np.take_along_axis(np.pad(cd,((0,0),(1,0))),self.start,axis=1)
        upper=np.clip(upper,0,1);lower=np.minimum(np.clip(lower,0,1),upper)
        group=upper-lower;dg=du-dl
        powu=upper**k;powl=lower**k
        mass=powu-powl
        dm_alpha=k*(upper**(k-1)*du-lower**(k-1)*dl)
        dm_logk=k*(powu*np.log(np.maximum(upper,1e-300))-powl*np.log(np.maximum(lower,1e-300)))
        ratio=np.divide(p,group,out=np.zeros_like(p),where=group>0)
        dr=np.divide(dp-ratio*dg,group,out=np.zeros_like(p),where=group>0)
        considered=mass*ratio;dc=dm_alpha*ratio+mass*dr;dk=dm_logk*ratio
        result=(1-lapse)*considered+lapse*p
        grad=np.stack([(1-lapse)*dc+lapse*dp,(1-lapse)*dk,p-considered])
        # Remove only floating-point normalization drift, with matching derivative.
        total=result.sum(1,keepdims=True);dtotal=grad.sum(2,keepdims=True)
        grad=(grad*total[None,:,:]-result[None,:,:]*dtotal)/(total[None,:,:]**2);result/=total
        return result,grad
    def objective(self,theta,target,weight):
        p,g=self.policy(theta);idx=self.inverse[np.arange(len(target)),target]
        pt=p[np.arange(len(p)),idx]
        return float(weight@(-np.log(pt))),-(g[:,np.arange(len(p)),idx]*(weight/pt)[None,:]).sum(1)
    def original(self,theta):
        p,_=self.policy(theta)
        return np.take_along_axis(p,self.inverse,axis=1)

def test():
    p=np.array([.2,.3,.5]);z=np.log(p)[None,:]
    for q in [np.array([[-1.,0.,1.]]),np.array([[0.,0.,1.]]),np.zeros((1,3))]:
        c=Consideration(z,q,np.ones_like(q,bool))
        for k in (1,2,3):
            truth=np.zeros(3)
            for draw in itertools.product(range(3),repeat=k):
                weight=np.prod(p[list(draw)]);v=q[0,list(draw)];best=np.array(draw)[v==v.max()]
                for a in best:truth[a]+=weight/len(best)
            np.testing.assert_allclose(c.original([1,np.log(k),0])[0],truth,atol=2e-14)
    rng=np.random.default_rng(781);z=rng.normal(size=(23,7));q=rng.integers(0,4,z.shape).astype(float);mask=rng.random(z.shape)>.2;mask[:,0]=True
    c=Consideration(z,q,mask);theta=np.array([.9,np.log(3.2),.2]);p,g=c.policy(theta)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-14)
    for j in range(3):
        a=theta.copy();b=theta.copy();a[j]+=1e-5;b[j]-=1e-5
        finite=(c.policy(a)[0]-c.policy(b)[0])/2e-5
        np.testing.assert_allclose(finite,g[j],atol=2e-8,rtol=2e-5)
    assert (c.original(theta)[~mask]==0).all()
    print('PASS exact candidate-set enumeration, ties, K=1 prior identity, analytic gradients, masks and normalization')

def main():
    test();start=time.monotonic();out=ROOT/'aug-consideration-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-expanded-search-v1';sp=json.loads((source/'plan.json').read_text())
    plan=dict(sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),source_plan_sha256=digest(source/'plan.json'),
        formula='For candidates iid from calibrated prior p, the probability of selecting value-ranked action i from K draws is F(q_i)^K−F(q_i-)^K, split among tied actions proportional to prior. K is fit as an effective continuous count. Mix with prior via lapse to keep full support.',
        budget=1000,backup='constant',group='4 mover-Elo groups',
        bounds=dict(alpha=[.4,1.6],effective_K=[1,64],lapse=[.001,1.]),starts=[[2,.5],[8,.5]],
        selection='Fit alpha, logK and lapse within game CV, then report disjoint August confirmation. No labels or future moves enter per-query candidate-set integration. K is a behavioral parameter, not extra model-evaluated nodes.',
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('sample_data.py'),Path(__file__).with_name('fit_policy.py')]})
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    d=read('aug-tune-expanded-v1');cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];ids=d['ids'];mask=d['mask'];target=d['target'];n=len(target);ar=np.arange(n);group=cells%4
    logits=np.zeros_like(mask,float);q=np.zeros_like(logits);cost=np.zeros(n);bi=sp['budgets'].index(1000);mi=list(sp['methods']).index('constant')
    for lo in range(0,n,sp['roots_per_batch']):
        with np.load(source/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1];np.testing.assert_array_equal(z['game'],games[lo:hi]);np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk])
            logits[lo:hi]=z['z'][:,378:2346][np.arange(hi-lo)[:,None],ids[lo:hi]]
            q[lo:hi,:kk]=z['q'][bi,mi];cost[lo:hi]=z['evaluated_nodes'][bi]
    logits=np.where(mask,logits,0);losses={};oof={name:np.full(n,np.nan) for name in ['search','consideration']};paramsfull=None
    for fold,train in [(None,fm)]+[(f,fm&(cv!=f)) for f in range(3)]:
        policies={name:np.zeros_like(logits) for name in oof};params={}
        for g in range(4):
            tr=train&(group==g);pred=group==g;counts=np.bincount(cells[tr],minlength=16)
            weight=1/counts[cells[tr]];weight/=weight.sum()
            f=fit(logits[tr],q[tr],mask[tr],target[tr],'forward',weight);assert f['converged']
            policies['search'][pred],_=loss_gradient([f['alpha'],f['beta']],logits[pred],q[pred],mask[pred],target[pred],'forward',return_policy=True)
            c=Consideration(logits[tr],q[tr],mask[tr]);candidates=[]
            for k,lapse in plan['starts']:
                r=minimize(lambda theta:c.objective(theta,target[tr],weight),[f['alpha'],np.log(k),lapse],jac=True,
                    method='L-BFGS-B',bounds=[(.4,1.6),(0,np.log(64)),(.001,1)],options=dict(maxiter=100,ftol=1e-11,gtol=1e-7))
                candidates.append(r)
            r=min(candidates,key=lambda r:r.fun);assert r.success,r
            policies['consideration'][pred]=Consideration(logits[pred],q[pred],mask[pred]).original(r.x)
            params[str(g)]=dict(alpha=float(r.x[0]),K=float(np.exp(r.x[1])),lapse=float(r.x[2]),iterations=int(r.nit))
        if fold is None:paramsfull=params
        for name,p in policies.items():
            loss=-np.log(p[ar,target])
            if fold is None:losses[name]=loss
            else:oof[name][fm&(cv==fold)]=loss[fm&(cv==fold)]
    records={}
    for name in losses:
        ce=means(losses[name][~fm],cells[~fm]);cc=means(oof[name][fm],cells[fm])
        records[name]=dict(parameters=paramsfull if name=='consideration' else None,training_equivalent_cm=None,mean_nodes=float(cost.mean()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()))
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count
    delta=losses['consideration']-losses['search'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
    records['consideration']['delta_vs_search']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='Expanded August fit-game CV and confirmation, potentially training-seen. No golden CM.',results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,plan_sha256=digest(pp))
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,r in records.items():print(name,r['confirmation'],r.get('delta_vs_search'),flush=True)
    print('SELECTED',selected,'PARAMS',paramsfull,flush=True)
if __name__=='__main__':main()
