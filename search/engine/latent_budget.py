"""Mixtures over latent deliberation budgets, with honest nested game CV.

Output policies use Elo-specific two-scalar calibration. Meta-gates are fit only
to cross-fitted component likelihoods. Every outer validation game's labels are
excluded from both levels. August confirmation is used for reporting only.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from .service import ROOT,atomic
from .fit_policy import fit,loss_gradient
from .analyze_august import means


def gate_fit(x,likelihood,cells,train,ridge):
    # Component0 logit is fixed at0; remaining four columns are identifiable.
    count=np.bincount(cells[train],minlength=16);w=1/count[cells[train]];w/=w.sum()
    a=x[train];l=likelihood[train]
    def objective(flat):
        theta=flat.reshape(x.shape[1],4);g=softmax(np.column_stack([np.zeros(len(a)),a@theta]),axis=1)
        mass=(g*l).sum(1);post=g*l/mass[:,None]
        loss=-w@np.log(mass)+.5*ridge*np.square(theta[4:]).sum()
        grad=a.T@(w[:,None]*(g[:,1:]-post[:,1:]));grad[4:]+=ridge*theta[4:]
        return loss,grad.ravel()
    opt=minimize(objective,np.zeros(x.shape[1]*4),jac=True,method='L-BFGS-B',bounds=[(-8,8)]*(x.shape[1]*4),options=dict(maxiter=250,ftol=1e-11,gtol=1e-7))
    assert np.isfinite(opt.fun)
    return opt.x.reshape(x.shape[1],4),dict(converged=bool(opt.success),iterations=int(opt.nit))


def main():
    start=time.monotonic();folder=ROOT/'aug-search-v1';rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);groups=cells%4;games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games]);inner=np.array([int(hashlib.sha256(('inner:'+g).encode()).hexdigest(),16)%2 for g in games])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    qs=np.zeros((5,n,k));cost=np.zeros((5,n));root=np.zeros((n,2432))
    for lo in range(0,n,128):
        with np.load(folder/f'mcts-{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1];assert list(z['game'])==list(games[lo:hi])
            root[lo:hi]=z['root'];qs[1:,lo:hi,:kk]=z['q'][:,3];cost[1:,lo:hi]=z['evaluated_nodes']
    logits=root[:,378:2346][ar[:,None],ids];base=softmax(np.where(mask,logits,-np.inf),axis=1)
    tp=softmax(root[:,2350:2413],axis=1);seconds=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    features=np.column_stack([np.log1p(tp@seconds),-(tp*np.log(np.maximum(tp,1e-300))).sum(1),
        -(base*np.log(np.maximum(base,1e-300))).sum(1),np.abs(softmax(root[:,2413:2416],axis=1)@np.array([1.,0.,-1.]))])
    # Shuffle independently within cell and fit/confirmation fold. No target or
    # future time is used; this is a diagnostic control, never a deployed method.
    shuffled=features[:,0].copy();rng=np.random.default_rng(88421)
    for fold in (False,True):
        for c in range(16):
            ix=np.flatnonzero((fm==fold)&(cells==c));shuffled[ix]=features[rng.permutation(ix),0]
    def components(train,pred):
        out=np.zeros((5,n,k));params=[]
        for j,q in enumerate(qs):
            par={}
            for g in range(4):
                a=train&(groups==g);b=pred&(groups==g);count=np.bincount(cells[a],minlength=16)
                f=fit(logits[a],q[a],mask[a],target[a],'forward',1/count[cells[a]]);par[str(g)]=f
                out[j,b],_=loss_gradient([f['alpha'],f['beta']],logits[b],q[b],mask[b],target[b],'forward',return_policy=True)
            params.append(par)
        return out,params
    # Cache honest outer predictions and inner cross-fits once, shared by gates.
    outer=[];oof=np.zeros((5,n,k))
    for f in range(3):
        train=fm&(cv!=f);val=fm&(cv==f);test_p,_=components(train,val);oof[:,val]=test_p[:,val]
        inner_p=np.zeros_like(oof)
        for h in range(2):
            held=train&(inner==h);p,_=components(train&(inner!=h),held);inner_p[:,held]=p[:,held]
        outer.append((train,val,inner_p[:,ar,target].T,test_p))
    full,parameters=components(fm,np.ones(n,bool));oof_like=oof[:,ar,target].T
    menus=[('elo_weights',[],0.),('predtime',[0],.003),('shuffled_time',[-1],.003),('state',[0,1,2,3],.003),('state_ridge',[0,1,2,3],.03)]
    def design(fields,train):
        feat=np.column_stack([shuffled if j<0 else features[:,j] for j in fields]) if fields else np.zeros((n,0))
        mu=feat[train].mean(0);sd=np.maximum(feat[train].std(0),1e-6);feat=(feat-mu)/sd
        hot=np.eye(4)[groups];x=np.column_stack([hot,*[hot*feat[:,j:j+1] for j in range(feat.shape[1])]])
        return x,dict(mean=mu.tolist(),scale=sd.tolist())
    def weights(x,theta):return softmax(np.column_stack([np.zeros(n),x@theta]),axis=1)
    records={};losses={}
    fixed_loss=-np.log(full[-1,ar,target]);d=means(fixed_loss[~fm],cells[~fm]);c=means(-np.log(oof[-1,ar[fm],target[fm]]),cells[fm])
    records['fixed1000']=dict(confirmation=dict(macro_ce=float(d.mean()),expert_ce=float(d[3::4].mean()),cells=d.tolist()),fit_game_cv=dict(macro_ce=float(c.mean()),expert_ce=float(c[3::4].mean())))
    losses['fixed1000']=fixed_loss
    for name,fields,ridge in menus:
        cv_loss=np.full(n,np.nan)
        for train,val,likelihood,p in outer:
            x,_=design(fields,train);theta,_=gate_fit(x,likelihood,cells,train,ridge);w=weights(x,theta)
            cv_loss[val]=-np.log((w[val]*p[:,ar[val],target[val]].T).sum(1))
        x,normalizer=design(fields,fm);theta,fit_info=gate_fit(x,oof_like,cells,fm,ridge);w=weights(x,theta)
        p=np.einsum('nb,bnk->nk',w,full);np.testing.assert_allclose(p.sum(1),1,atol=1e-12);loss=-np.log(p[ar,target])
        d=means(loss[~fm],cells[~fm]);c=means(cv_loss[fm],cells[fm]);losses[name]=loss
        records[name]=dict(fields=fields,ridge=ridge,theta=theta.tolist(),normalizer=normalizer,fit=fit_info,
            diagnostic_only=name=='shuffled_time',training_equivalent_cm=None,mean_nodes=float(means(cost[-1],cells).mean()),
            expert_mean_nodes=float(means(cost[-1],cells)[3::4].mean()),
            confirmation=dict(macro_ce=float(d.mean()),expert_ce=float(d[3::4].mean()),cells=d.tolist()),
            fit_game_cv=dict(macro_ce=float(c.mean()),expert_ce=float(c[3::4].mean())))
        print('Latent budget',name,records[name]['fit_game_cv'],records[name]['confirmation'],flush=True)
    selected={metric:min((k for k,r in records.items() if not r.get('diagnostic_only')),key=lambda k:records[k]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(11781).multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    for name in records:
        diff=losses[name]-fixed_loss;sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),diff[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_vs_fixed1000_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August training-seen development, nested game CV; no golden tuning or CM conversion.',
        component_budgets=[0,16,64,256,1000],component_parameters=parameters,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fit_cv_selected=selected,results=records,
        analysis_seconds=time.monotonic()-start,cost_note='All mixture components use the same1000-simulation tree, so all1000 are paid for; this is a quality experiment, not yet adaptive compute allocation.')
    atomic(folder/'latent-budget.json',report);print('SELECTED',selected,'seconds',report['analysis_seconds'],flush=True)


if __name__=='__main__':main()
