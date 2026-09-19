"""Game-CV calibration of matched deeper adaptive-backup trees."""
import json,time
from pathlib import Path
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .fit_policy import fit,loss_gradient
from .analyze_august import means
from .retrieval_common import data

def main():
    start=time.monotonic();out=ROOT/'aug-adaptive-deep-v1';plan=json.loads((out/'plan.json').read_text())
    d=data();cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];ids=d['ids'];mask=d['mask'];target=d['target'];n=len(target);ar=np.arange(n)
    assert plan['sample_sha256']==digest(ROOT/'aug-tune-v1/sample.json')
    root=np.zeros((n,2432));q=np.zeros((len(plan['budgets']),len(plan['methods']),n,mask.shape[1]));cost=np.zeros((len(plan['budgets']),n))
    for lo in range(0,n,plan['roots_per_batch']):
        with np.load(out/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1]
            np.testing.assert_array_equal(z['game'],games[lo:hi]);np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk]);np.testing.assert_array_equal(z['mask'],mask[lo:hi,:kk])
            q[:,:,lo:hi,:kk]=z['q'];cost[:,lo:hi]=z['evaluated_nodes'];root[lo:hi]=z['z']
    logits=root[:,378:2346][ar[:,None],ids];group=cells%4;records={};losses={}
    for b,budget in enumerate(plan['budgets']):
        for j,name in enumerate(plan['methods']):
            def predict(train):
                p=np.zeros_like(logits);params={}
                for g in range(4):
                    tr=train&(group==g);pred=group==g;counts=np.bincount(cells[tr],minlength=16)
                    f=fit(logits[tr],q[b,j,tr],mask[tr],target[tr],'forward',1/counts[cells[tr]]);assert f['converged'];params[str(g)]=f
                    p[pred],_=loss_gradient([f['alpha'],f['beta']],logits[pred],q[b,j,pred],mask[pred],target[pred],'forward',return_policy=True)
                return p,params
            p,params=predict(fm);loss=-np.log(p[ar,target]);oof=np.full(n,np.nan)
            for fold in range(3):
                pp,_=predict(fm&(cv!=fold));val=fm&(cv==fold);oof[val]=-np.log(pp[ar[val],target[val]])
            conf=means(loss[~fm],cells[~fm]);cc=means(oof[fm],cells[fm]);label=f'{name}{budget}'
            records[label]=dict(parameters=params,method=name,budget=budget,training_equivalent_cm=None,
                mean_nodes=float(means(cost[b],cells).mean()),expert_mean_nodes=float(means(cost[b],cells)[3::4].mean()),
                fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),
                confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
            losses[label]=loss
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,rec in records.items():
        reference='constant'+str(rec['budget']);delta=losses[name]-losses[reference]
        point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['confirmation_delta_vs_matched_constant']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August game-disjoint confirmation, potentially training-seen; no golden CM.',results=records,fit_cv_selected=selected,
        analysis_seconds=time.monotonic()-start,plan_sha256=digest(out/'plan.json'),source_sha256=digest(Path(__file__)))
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,x in records.items():print(name,x['confirmation']['macro_ce'],x['confirmation']['expert_ce'],x['mean_nodes'],x['confirmation_delta_vs_matched_constant'],flush=True)
    print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
