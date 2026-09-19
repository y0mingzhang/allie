"""Differentiate fixed-tree soft backups; fit two scalars or joint output calibration."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .service import ROOT,atomic
from .balanced_eval import digest
from .diff_backup_native import load,test
from .sample_data import read
from .fit_policy import fit,loss_gradient
from .analyze_august import means

def main(folder='aug-diff-backup-v2',sample='aug-tune-v1',source_folder='aug-coverage-v1/coverage_bernoulli',batch_size=512):
    start=time.monotonic();module=load();test(module)
    out=ROOT/folder;out.mkdir(exist_ok=True)
    source=ROOT/source_folder
    plan=dict(sample_sha256=digest(ROOT/sample/'sample.json'),budget=1000,
        inputs={p.name:digest(p) for p in sorted(source.glob('[0-9]*.npz'))},
        sample=sample,source_folder=source_folder,batch_size=batch_size,
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('sample_data.py'),Path(__file__).with_name('diff_backup.cpp'),Path(__file__).with_name('diff_backup_native.py')]},
        formula='tau=exp(a)*(1+(descendants-1)/16)^b; a=log tau0, b=count exponent.',
        variants=['constant_control','fixed_output_2scalar','joint_output_10scalar'],
        bounds=dict(tau0=[.01,.4],count_exponent=[-1.,0.],alpha=[.4,1.6],beta=[0.,40.]),
        starts=[[float(np.log(.1)),0.],[float(np.log(.2)),-.5]],
        selection='Both temperature starts fit on each training fold; choose by training loss. Three-way game CV selects method; all arms reported on disjoint August confirmation. Existing output alpha/beta refit inside each fold. Two-scalar arm holds output coefficients fixed; ten-scalar arm fits them jointly.',
        semantics='Output-only backup on unchanged root-coverage/cp2.5 PUCT trees. Selection does not read tau, so this changes no expansions. Frozen neural model. No future/target information enters per-query backup. No golden tuning. Native gradients checked against independent float64 torch/autograd and finite differences.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    d=read(sample);rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];ids=d['ids'];mask=d['mask'];target=d['target'];n=len(rows);ar=np.arange(n);group=cells%4
    blocks=[];root=np.zeros((n,2432));cost=np.zeros(n)
    for lo in range(0,n,batch_size):
        with np.load(source/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);np.testing.assert_array_equal(z['game'],games[lo:hi])
            payload={key:z[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            blocks.append((lo,hi,module.Backup(payload,1000)))
            root[lo:hi]=z['z'];cost[lo:hi]=z['evaluated_nodes'][-1]
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0)
    def reduce(ab):
        return np.concatenate([obj.reduce(*ab,ids[lo:hi].astype(np.int32)) for lo,hi,obj in blocks],axis=1)
    reference=reduce([np.log(.1),0])
    def base_fit(train):
        theta=np.zeros(10);theta[:2]=[np.log(.1),0]
        for g in range(4):
            tr=train&(group==g);count=np.bincount(cells[tr],minlength=16)
            fp=fit(logits[tr],reference[0,tr],mask[tr],target[tr],'forward',1/count[cells[tr]])
            assert fp['converged'];theta[2+g]=fp['alpha'];theta[6+g]=fp['beta']/10
        return theta
    def probabilities(theta,q):
        a=theta[2:6][group];b=10*theta[6:10][group]
        z=np.where(mask,a[:,None]*logits+b[:,None]*q,-np.inf);return np.exp(z-logsumexp(z,axis=1,keepdims=True))
    bounds=[(np.log(.01),np.log(.4)),(-1.,0.)]+[(.4,1.6)]*4+[(0.,4.)]*4
    def objective_factory(train,base,joint):
        count=np.bincount(cells[train],minlength=16);weights=np.zeros(n);weights[train]=1/count[cells[train]];weights/=weights.sum()
        cached={};evaluations=0
        def objective(x):
            nonlocal evaluations
            theta=x if joint else np.r_[x,base[2:]]
            key=tuple(theta[:2])
            if key not in cached:
                cached.clear();cached[key]=reduce(theta[:2]);evaluations+=1
            qq=cached[key];p=probabilities(theta,qq[0]);loss=float(weights@(-np.log(p[ar,target])))
            residual=p.copy();residual[ar,target]-=1;residual*=weights[:,None]
            beta=10*theta[6:10][group]
            grad=[np.sum(residual*beta[:,None]*qq[i]) for i in (1,2)]
            if joint:
                grad += [np.sum(residual[group==g]*logits[group==g]) for g in range(4)]
                grad += [10*np.sum(residual[group==g]*qq[0,group==g]) for g in range(4)]
            return loss,np.array(grad)
        return objective
    def optimize(train,base,joint):
        fn=objective_factory(train,base,joint);attempts=[]
        for ab in plan['starts']:
            x=np.r_[ab,base[2:]] if joint else np.array(ab)
            result=minimize(fn,x,jac=True,method='L-BFGS-B',bounds=bounds if joint else bounds[:2],
                options=dict(maxiter=100,ftol=1e-10,gtol=2e-6,maxls=30))
            assert np.isfinite(result.fun)
            attempts.append(dict(x=result.x.tolist(),loss=float(result.fun),success=bool(result.success),message=str(result.message),iterations=int(result.nit)))
        best=min(attempts,key=lambda z:z['loss'])
        # Do not silently promote a fit that terminated without convergence.
        assert best['success'],attempts
        x=np.array(best['x']);theta=x if joint else np.r_[x,base[2:]]
        return theta,dict(attempts=attempts,tau0=float(np.exp(theta[0])),count_exponent=float(theta[1]),alpha=theta[2:6].tolist(),beta=(10*theta[6:10]).tolist())
    fits=[];records={};losses={};oofs={name:np.full(n,np.nan) for name in plan['variants']}
    for fold,train in [(None,fm)]+[(f,fm&(cv!=f)) for f in range(3)]:
        base=base_fit(train);params={'constant_control':(base,dict(tau0=.1,count_exponent=0,alpha=base[2:6].tolist(),beta=(10*base[6:]).tolist()))}
        for joint,name in [(False,'fixed_output_2scalar'),(True,'joint_output_10scalar')]:
            params[name]=optimize(train,base,joint)
            print('Fitted',fold,name,params[name][1],flush=True)
        fits.append(params)
        for name,(theta,details) in params.items():
            p=probabilities(theta,reduce(theta[:2])[0]);loss=-np.log(p[ar,target])
            if fold is None:losses[name]=loss
            else:oofs[name][fm&(cv==fold)]=loss[fm&(cv==fold)]
    for name in plan['variants']:
        ce=means(losses[name][~fm],cells[~fm]);cvce=means(oofs[name][fm],cells[fm])
        records[name]=dict(parameters=fits[0][name][1],training_equivalent_cm=None,mean_nodes=float(cost.mean()),
            fit_game_cv=dict(macro_ce=float(cvce.mean()),expert_ce=float(cvce[3::4].mean())),
            confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()))
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    # Local conditioning in optimizer coordinates; active bounds excluded.
    theta=fits[0]['joint_output_10scalar'][0];fn=objective_factory(fm,fits[0]['constant_control'][0],True)
    eps=1e-4;h=[]
    for j in range(10):
        xp=theta.copy();xm=theta.copy();xp[j]+=eps;xm[j]-=eps;h.append((fn(xp)[1]-fn(xm)[1])/(2*eps))
    h=np.array(h).T;h=(h+h.T)/2
    free=[i for i,(lo,hi) in enumerate(bounds) if theta[i]>lo+1e-5 and theta[i]<hi-1e-5]
    eigen=np.linalg.eigvalsh(h[np.ix_(free,free)])
    conditioning=dict(free_coordinates=free,eigenvalues=eigen.tolist(),condition_abs=float(np.max(np.abs(eigen))/max(np.min(np.abs(eigen)),1e-30)),note='Local Hessian in [logtau,exponent,alpha,beta/10] coordinates; bound-active dimensions excluded. Not a causal interpretation of fitted scalars.')
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,rec in records.items():
        for ref in ['constant_control','fixed_output_2scalar']:
            delta=losses[name]-losses[ref];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
            rec['delta_vs_'+ref]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='Output-only backup calibration on August; game-disjoint confirmation, potentially training-seen. Golden CM pending.',
        fit_cv_selected=selected,results=records,joint_conditioning=conditioning,analysis_seconds=time.monotonic()-start,plan_sha256=digest(pp))
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,rec in records.items():print(name,rec['confirmation'],flush=True)
    print('SELECTED',selected,'condition',conditioning,flush=True)
if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        assert sys.argv[1]=='expanded'
        main('aug-expanded-backup-v1','aug-tune-expanded-v1','aug-expanded-search-v1',1024)
    else:main()
