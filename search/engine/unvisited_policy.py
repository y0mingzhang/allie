"""Root-critic fallback and visit shrinkage, with no additional model nodes.

Preregistered family: unchanged Q, root-critic fallback, and 1/4/16 pseudo-visits
toward that root critic. Fit/selection uses only August fold0 and game CV.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from .service import ROOT,atomic
from .fit_policy import fit,loss_gradient
from .analyze_august import means


def shrink(q,visits,root,pseudovisits):
    assert pseudovisits>=0
    return np.divide(visits*q+pseudovisits*root[:,None],visits+pseudovisits,
        out=np.broadcast_to(root[:,None],q.shape).copy(),where=(visits+pseudovisits)>0)


def main():
    started=time.monotonic();folder=ROOT/'aug-search-v1';dest=folder/'unvisited-results.json'
    assert not dest.exists(),'Do not overwrite an experiment result'
    sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions'];n=len(rows);ar=np.arange(n)
    cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);target=np.zeros(n,int);ids=np.zeros((n,k),int)
    for i,r in enumerate(rows):mask[i,:len(r['legal'])]=True;ids[i,:len(r['legal'])]=np.array(r['legal'])-378;target[i]=r['legal'].index(r['target'])
    q=np.zeros((4,n,k));v=np.zeros_like(q);root=np.zeros((n,2432));cost=np.zeros((4,n))
    for lo in range(0,n,128):
        with np.load(folder/f'mcts-{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1];assert list(z['game'])==list(games[lo:hi])
            root[lo:hi]=z['root'];q[:,lo:hi,:kk]=z['q'][:,0];v[:,lo:hi,:kk]=z['visits'];cost[:,lo:hi]=z['evaluated_nodes']
    logits=np.where(mask,root[:,378:2346][ar[:,None],ids],0.);wdl=softmax(root[:,2413:2416],axis=1);value=wdl[:,0]-wdl[:,2]
    prior=softmax(np.where(mask,logits,-np.inf),axis=1);legal_loss=-np.log(prior[ar,target])
    records={};losses={}
    def calibrate(values,direction,group,train,pred):
        p=np.zeros((n,k));params={}
        for g in np.unique(group):
            a=train&(group==g);b=pred&(group==g);count=np.bincount(cells[a],minlength=16)
            t=fit(logits[a],values[a],mask[a],target[a],direction,1/count[cells[a]])
            p[b],_=loss_gradient([t['alpha'],t['beta']],logits[b],values[b],mask[b],target[b],direction,return_policy=True)
            params[str(int(g))]=t
        return p,params
    for j,budget in enumerate((16,64,256,1000)):
        versions={'unchanged':q[j],**{f'critic_nu{nu}':shrink(q[j],v[j],value,nu) for nu in (0,1,4,16)}}
        for variant,values in versions.items():
            assert np.isfinite(values).all() and np.max(abs(values))<=1.000001
            for direction in ('forward','reverse'):
                for grouping,group in [('global',np.zeros(n,int)),('elo',cells%4)]:
                    name=f'{budget}_{variant}_{direction}_{grouping}'
                    p,params=calibrate(values,direction,group,fm,np.ones(n,bool));nll=-np.log(p[ar,target]);losses[name]=nll
                    oof=np.full(n,np.nan)
                    for fold in range(3):
                        val=fm&(cv==fold);cp,_=calibrate(values,direction,group,fm&(cv!=fold),val)
                        oof[val]=-np.log(cp[ar[val],target[val]])
                    c=means(oof[fm],cells[fm]);d=means(nll[~fm],cells[~fm])
                    records[name]=dict(budget=budget,variant=variant,parameters=params,training_equivalent_cm=None,
                        mean_nodes=float(means(cost[j],cells).mean()),expert_mean_nodes=float(means(cost[j],cells)[3::4].mean()),
                        fit_game_cv=dict(macro_ce=float(c.mean()),expert_ce=float(c[3::4].mean())),
                        confirmation=dict(macro_ce=float(d.mean()),expert_ce=float(d[3::4].mean()),cells=d.tolist()))
        print('Unvisited fallback calibrated',budget,flush=True)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);g=ix.max()+1
    count=np.zeros((g,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(613814).multinomial(g,np.full(g,1/g),size=1000).astype(float);den=w@count;assert (den>0).all()
    for name in set(selected.values()):
        sums=np.zeros((g,16));np.add.at(sums,(ix,cells[~fm]),(losses[name]-legal_loss)[~fm]);delta=(w@sums)/den
        records[name]['confirmation_delta_vs_legal_ci95']=dict(macro=np.quantile(delta.mean(1),[.025,.975]).tolist(),expert=np.quantile(delta[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August potentially training-seen tuning: equal-node root-value fallback/shrinkage; no golden CM.',
        input_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fit_cv_selected=selected,results=records,analysis_seconds=time.monotonic()-started)
    atomic(dest,report)
    with (folder/'unvisited-scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,fit=fm,games=games)
    for name in set(selected.values()):print(name,records[name]['confirmation'],flush=True)


if __name__=='__main__':main()
