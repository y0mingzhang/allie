"""CPU-only adaptive soft-backup temperature screen on frozen coverage trees."""
import json,time
from pathlib import Path
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .adaptive_temperature_native import load,test
from .fit_policy import fit,loss_gradient
from .analyze_august import means
from .retrieval_common import data

VARIANTS=[('constant005',0,.05,16),('constant010',0,.1,16),('constant020',0,.2,16),
    ('depth_grow',1,.05,16),('depth_shrink',2,.2,16),('subtree',3,.2,16),('action_scale',4,.5,16)]
def main():
    start=time.monotonic();module=load();test(module);out=ROOT/'aug-adaptive-temperature-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-coverage-v1';original=json.loads((source/'plan.json').read_text())
    plan=dict(variants=VARIANTS,sample_sha256=original['sample_sha256'],control='same frozen coverage_bernoulli1000 trees, constant tau.1',
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('adaptive_temperature.cpp'),Path(__file__).with_name('adaptive_temperature_native.py')]},
        semantics='Same nodes, altered Bellman softness. Descendant count is an exploration proxy, NOT independent statistical evidence. Depth and child-value spread variants. No added GPU nodes or future moves.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(plan))
    else:atomic(pp,plan)
    d=data();rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];ids=d['ids'];mask=d['mask'];target=d['target'];n=len(rows);ar=np.arange(n)
    q=np.zeros((len(VARIANTS),*mask.shape));root=np.zeros((n,2432));cost=np.zeros(n)
    for lo in range(0,n,512):
        with np.load(source/'coverage_bernoulli'/f'{lo:06d}.npz') as z:
            data_={key:z[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            hi=lo+len(z['game']);np.testing.assert_array_equal(z['game'],games[lo:hi]);root[lo:hi]=z['z'];cost[lo:hi]=z['evaluated_nodes'][-1]
            for j,(name,mode,tau,scale) in enumerate(VARIANTS):
                q[j,lo:hi]=module.reduce(data_,1000,mode,tau,scale)[np.arange(hi-lo)[:,None],ids[lo:hi]]
            kk=z['ids'].shape[1];np.testing.assert_allclose(q[1,lo:hi,:kk][z['mask']],z['q'][-1][z['mask']],atol=1e-9,rtol=1e-9)
        print('Temperature reduced',hi,'/',n,flush=True)
    logits=root[:,378:2346][ar[:,None],ids];group=cells%4;records={};losses={}
    for j,(name,*_) in enumerate(VARIANTS):
        def predict(train):
            p=np.zeros_like(logits);params={}
            for g in range(4):
                tr=train&(group==g);pred=group==g;count=np.bincount(cells[tr],minlength=16)
                f=fit(logits[tr],q[j,tr],mask[tr],target[tr],'forward',1/count[cells[tr]]);assert f['converged'];params[str(g)]=f
                p[pred],_=loss_gradient([f['alpha'],f['beta']],logits[pred],q[j,pred],mask[pred],target[pred],'forward',return_policy=True)
            return p,params
        p,params=predict(fm);loss=-np.log(p[ar,target]);oof=np.full(n,np.nan)
        for f in range(3):
            p_,_=predict(fm&(cv!=f));val=fm&(cv==f);oof[val]=-np.log(p_[ar[val],target[val]])
        conf=means(loss[~fm],cells[~fm]);cc=means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,training_equivalent_cm=None,mean_nodes=float(cost.mean()),
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        losses[name]=loss
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    report=dict(stage='CPU-only August alternate Bellman temperatures, no golden CM.',fit_cv_selected=selected,results=records,seconds=time.monotonic()-start,plan_sha256=digest(pp))
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,x in records.items():print(name,x['confirmation']['macro_ce'],x['confirmation']['expert_ce'],flush=True)
    print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
