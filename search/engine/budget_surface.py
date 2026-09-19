"""Quality/cost ladder for the selected inference recipe; no new NN evaluations."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .diff_backup_native import load
from .fit_policy import fit
from .analyze_expanded import components
from .analyze_player_search import FACTORS
from .analyze_august import means
from .conditional_mixture import objective as gate_objective
from .temperature_stack import evaluate as temperature


def counts(data, budgets):
    parent=data['parent'].astype(np.int64)
    owner=np.where(parent<0,np.arange(len(parent)),parent)
    while True:
        jumped=owner[owner]
        if np.array_equal(jumped,owner):break
        owner=jumped
    mapping=np.full(len(parent),-1,np.int64);mapping[data['roots']]=np.arange(len(data['roots']))
    group=mapping[owner];assert (group>=0).all()
    return np.array([np.bincount(group[(parent>=0)&(data['born']<=b)&(data['terminal']<0)],minlength=len(data['roots'])) for b in budgets])


def main():
    start=time.monotonic();budgets=[128,256,512,1000]
    out=ROOT/'aug-budget-surface-v2';out.mkdir(exist_ok=True)
    source=ROOT/'aug-expanded-search-v1'
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n,ar=len(rows),np.arange(len(rows));group=cells%4
    plan=dict(budgets=budgets,sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        parent_plan_sha256=digest(source/'plan.json'),
        reference_sha256=digest(ROOT/'aug-temperature-stack-v1/results.json'),
        reducer_audit_sha256=digest(ROOT/'aug-budget-surface-v1/reducer-audit.json'),
        numerical_change='Corrected soft reducer: zero missing prior mass when all legal children are expanded. Legacy comparison audited across every block, largest Q drift 2.44735e-5 on one action. Refits use corrected values; report end-to-end CE drift.',
        sources={p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(s) for s in
            ('diff_backup.cpp','diff_backup_native.py','conditional_mixture.py','temperature_stack.py','fit_policy.py','analyze_expanded.py')]]},
        recipe='At each budget refit Elo alpha/beta, then all0.001 conditional strength mixture, then state0.01 residual temperature. Same selected formula and hyperparameters at each budget. Every stage refitted within3 game folds.',
        semantics='Logical prefix snapshots of fixed1000 root-coverage trees. Node count from nonterminal nonroot birth records, verified against stored256/1000 counters. No additional NN queries. No golden tuning; eventual live mixed-budget execution must verify batching drift.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    root=np.zeros((n,2432));q=np.zeros((len(budgets),*mask.shape));nodes=np.zeros((len(budgets),n))
    module=load()
    for lo in range(0,n,1024):
        with np.load(source/f'{lo:06d}.npz') as f:
            hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            data={k:f[k] for k in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            root[lo:hi]=f['z'];nodes[:,lo:hi]=counts(data,budgets)
            np.testing.assert_array_equal(nodes[[1,3],lo:hi],f['evaluated_nodes'])
            for bi,b in enumerate(budgets):q[bi,lo:hi]=module.Backup(data,b).reduce(np.log(.2),-.5,ids[lo:hi].astype(np.int32))[0]
            assert np.isfinite(q[:,lo:hi]).all()
    inventory=ROOT/'aug-tune-v1';manifest=json.loads((inventory/'manifest.json').read_text())
    for name in ('strat.npz','feats.npz'):assert digest(inventory/name)==manifest['files_sha256'][name]
    with np.load(inventory/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(inventory/'feats.npz') as f:feats=f['feats']
    seconds=[]
    for r in rows:
        rr,cc=r['row'],r['column'];assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(r['prefix']):cc],r['prefix']);seconds.append(feats[rr,cc-1,0])
    seconds=np.array(seconds);known=seconds>=0
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    prior=softmax(np.where(mask,logits,-np.inf),axis=1)
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)];predtime=np.log1p(softmax(root[:,2350:2413],axis=1)@tc)
    records,losses,all_params,policy_versions={}, {}, {}, []
    for bi,b in enumerate(budgets):
        qm=(prior*q[bi]).sum(1);qstd=np.sqrt((prior*(q[bi]-qm[:,None])**2).sum(1))
        features=np.column_stack([entropy,np.log(.01+qstd),[len(r['prefix'])-11 for r in rows],predtime,
            np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known])
        versions=[];oofs={k:np.full(n,np.nan) for k in ('fixed','gate','temperature')};all_params[str(b)]=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            params={}
            for g in range(4):
                tr=train&(group==g);ct=np.bincount(cells[tr],minlength=16)
                params[str(g)]=fit(logits[tr],q[bi,tr],mask[tr],target[tr],'forward',1/ct[cells[tr]])
                assert params[str(g)]['converged']
            comp=components(logits,q[bi],mask,params,group)
            fixed=np.einsum('a,ank->nk',softmax(-.5*np.log(FACTORS)**2),comp)
            mean=features[train].mean(0);scale=np.maximum(features[train].std(0),1e-6)
            x=np.c_[np.ones(n),np.clip((features-mean)/scale,-3,3)]
            ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
            pt=comp[:,ar,target].T
            opt=minimize(lambda t:gate_objective(t,x[train],pt[train],w,.001),np.zeros(8),jac=True,
                method='L-BFGS-B',bounds=[(-2,2)]*8,options=dict(ftol=1e-11,gtol=1e-7,maxiter=200));assert opt.success,opt
            mu=x@opt.x;mu[~known]=0.
            gw=softmax(-.5*np.log(FACTORS)[None,:]**2+mu[:,None]*np.log(FACTORS),axis=1)
            gp=np.einsum('na,ank->nk',gw,comp)
            logp=np.where(mask,np.log(np.maximum(gp,1e-300)),0.)
            tx=x[:,:5]
            tp=minimize(lambda t:temperature(t,tx[train],logp[train],mask[train],target[train],w,.01),np.zeros(5),jac=True,
                method='L-BFGS-B',bounds=[(-1,1)]*5,options=dict(ftol=1e-11,gtol=1e-7,maxiter=250));assert tp.success,tp
            final=temperature(tp.x,tx,logp,mask,target);versions.append(final)
            all_params[str(b)].append(dict(fold=fold,root=params,mean=mean.tolist(),scale=scale.tolist(),gate=opt.x.tolist(),temperature=tp.x.tolist()))
            for key,p in [('fixed',fixed),('gate',gp),('temperature',final)]:
                ll=-np.log(p[ar,target])
                if fold is None:losses[f'{b}_{key}']=ll
                else:oofs[key][fm&(cv==fold)]=ll[fm&(cv==fold)]
        policy_versions.append(np.array(versions,dtype=np.float32))
        for key in ('fixed','gate','temperature'):
            ce,cc=means(losses[f'{b}_{key}'][~fm],cells[~fm]),means(oofs[key][fm],cells[fm])
            records[f'{b}_{key}']=dict(budget=b,stage=key,parameters=all_params[str(b)][0],training_equivalent_cm=None,
                mean_nodes=float(nodes[bi].mean()),expert_mean_nodes=float(nodes[bi,cells%4==3].mean()),
                confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),
                fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())))
            print(b,key,ce.mean(),ce[3::4].mean(),nodes[bi].mean(),flush=True)
    old=json.loads((ROOT/'aug-temperature-stack-v1/results.json').read_text())
    drift={}
    for field in ('confirmation','fit_game_cv'):
        drift[field]={}
        for metric in ('macro_ce','expert_ce'):
            drift[field][metric]=records['1000_temperature'][field][metric]-old['results']['state0.01'][field][metric]
            assert abs(drift[field][metric])<1e-6,drift
    with (out/'policies.npz').open('wb') as f:
        np.savez_compressed(f,budgets=budgets,policy=np.array(policy_versions),q=q,root=root,nodes=nodes,ids=ids,mask=mask,target=target,cells=cells,games=games,fit=fm,cv=cv)
    atomic(out/'results.json',dict(stage=plan['semantics'],results=records,fold_parameters=all_params,analysis_seconds=time.monotonic()-start,
        plan_sha256=digest(pp),policies_sha256=digest(out/'policies.npz'),corrected_reducer_ce_delta=drift))


if __name__=='__main__':main()
