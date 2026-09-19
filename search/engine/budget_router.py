"""Fit-only allocation of a finite node budget across known rating groups.

The 625 assignments include direct/16/64/256/1000 simulations for each of four
Elo groups. Root policy calibration uses game CV, then allocation minimizes
CV macro CE under a development node cap. Confirmation games select nothing.
"""
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,atomic
from .fit_policy import fit,loss_gradient
from .analyze_august import means


def main():
    start=time.monotonic();rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);groups=cells%4;games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    qs=np.zeros((5,n,k));cost=np.zeros((5,n));root=np.zeros((n,2432));budgets=[0,16,64,256,1000]
    for lo in range(0,n,128):
        with np.load(ROOT/f'aug-search-v1/mcts-{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1];assert list(z['game'])==list(games[lo:hi])
            root[lo:hi]=z['root'];qs[1:,lo:hi,:kk]=z['q'][:,3];cost[1:,lo:hi]=z['evaluated_nodes']
    logits=root[:,378:2346][ar[:,None],ids];parameters=[];losses=[];oofs=[]
    def predict(q,train,pred):
        p=np.zeros_like(q);params={}
        for g in range(4):
            a=train&(groups==g);b=pred&(groups==g);count=np.bincount(cells[a],minlength=16)
            f=fit(logits[a],q[a],mask[a],target[a],'forward',1/count[cells[a]]);params[str(g)]=f
            p[b],_=loss_gradient([f['alpha'],f['beta']],logits[b],q[b],mask[b],target[b],'forward',return_policy=True)
        return p,params
    for q in qs:
        p,params=predict(q,fm,np.ones(n,bool));parameters.append(params);losses.append(-np.log(p[ar,target]));oof=np.full(n,np.nan)
        for fold in range(3):
            val=fm&(cv==fold);p,_=predict(q,fm&(cv!=fold),val);oof[val]=-np.log(p[ar[val],target[val]])
        oofs.append(oof)
    losses=np.array(losses);oofs=np.array(oofs)
    cv_group=np.array([means(x[fm],cells[fm]).reshape(4,4).mean(0) for x in oofs])
    cost_group=np.array([means(x[fm],cells[fm]).reshape(4,4).mean(0) for x in cost])
    possibilities=[]
    for assignment in itertools.product(range(5),repeat=4):
        a=np.array(assignment);possibilities.append((float(cv_group[a,np.arange(4)].mean()),float(cost_group[a,np.arange(4)].mean()),assignment))
    choices={f'cap{cap}':min((x for x in possibilities if x[1]<=cap),key=lambda x:(x[0],x[1]))[2] for cap in (64,128,256,512,768,844,1000)}
    choices.update({f'fixed{b}':(i,)*4 for i,b in enumerate(budgets)})
    records={};routed_losses={}
    for name,assignment in choices.items():
        a=np.array(assignment)[groups];loss=losses[a,ar];nodes=cost[a,ar];c=means(oofs[a,ar][fm],cells[fm]);d=means(loss[~fm],cells[~fm])
        records[name]=dict(budgets_by_elo=[budgets[i] for i in assignment],budget_indices=list(assignment),
            parameters={str(g):parameters[i][str(g)] for g,i in enumerate(assignment)},training_equivalent_cm=None,
            mean_nodes=float(means(nodes,cells).mean()),expert_mean_nodes=float(means(nodes,cells)[3::4].mean()),
            fit_game_cv=dict(macro_ce=float(c.mean()),expert_ce=float(c[3::4].mean())),
            confirmation=dict(macro_ce=float(d.mean()),expert_ce=float(d[3::4].mean()),cells=d.tolist()))
        routed_losses[name]=loss
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1;count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(71832).multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    for name in choices:
        diff=routed_losses[name]-routed_losses['fixed1000'];sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),diff[~fm]);draws=w@sums/den
        records[name]['confirmation_delta_vs_fixed1000_ci95']=dict(macro=np.quantile(draws.mean(1),[.025,.975]).tolist(),expert=np.quantile(draws[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August may be model-training-seen. Output parameters and budget assignments fit without confirmation labels. CM pending golden.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sample_sha256=hashlib.sha256((ROOT/'aug-tune-v1/sample.json').read_bytes()).hexdigest(),
        candidate_budgets=budgets,results=records,analysis_seconds=time.monotonic()-start,
        golden_preregistered=['cap256','cap512','cap768','fixed1000'],
        notes='Routing reads only the mover Elo bucket. All positions stay scored, including direct-policy fallbacks. Cached-counterfactual cost/quality require a separate dynamic-batching performance validation.')
    atomic(ROOT/'aug-search-v1/budget-router.json',report)
    for name in report['golden_preregistered']:print(name,records[name],flush=True)


if __name__=='__main__':main()
