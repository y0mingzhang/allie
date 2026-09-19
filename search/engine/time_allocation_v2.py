"""Predicted-time stopping on current cached search; no change to tree or output."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT,atomic
from .sample_data import read
from .balanced_eval import digest
from .analyze_august import means
BUDGETS=np.array([128,256,512,1000])

def choose(signal,scale):
    return np.searchsorted(np.sqrt(BUDGETS[:-1]*BUDGETS[1:]),np.asarray(signal)*scale)

def calibrate(signal,weights,cap=512.):
    lo,hi=0.,1000/np.min(signal)
    for _ in range(80):
        mid=(lo+hi)/2
        if weights@BUDGETS[choose(signal,mid)]>cap:hi=mid
        else:lo=mid
    assert weights@BUDGETS[choose(signal,lo)]<=cap+1e-8
    return lo

def permutation(cells,groups):
    rng=np.random.default_rng(724093);p=np.arange(len(cells))
    for g in np.unique(groups):
        for c in range(16):
            ix=np.flatnonzero((groups==g)&(cells==c));p[ix]=rng.permutation(ix)
    assert np.array_equal(cells[p],cells) and np.array_equal(groups[p],groups)
    return p

def test():
    x=np.exp(np.linspace(-4,4,10000));w=np.full(len(x),1/len(x));a=choose(x,calibrate(x,w))
    assert np.all(np.diff(a)>=0) and 511.8<w@BUDGETS[a]<=512
    assert np.all(choose(np.ones(100),calibrate(np.ones(100),np.full(100,.01)))==2)
    cells=np.tile(np.arange(16),30);groups=np.repeat(np.arange(3),160)
    assert sorted(permutation(cells,groups))==list(range(len(cells)))
    print('PASS monotone allocation, mean cap, constant fixed512, within-cell/fold shuffle',flush=True)

def run(oracle=None,spec=None):
    test();start=time.monotonic();out=ROOT/'aug-time-allocation-v2';out.mkdir(exist_ok=True)
    d=read('aug-tune-expanded-v1');cells,games,fm,cv,target=(d[k] for k in ('cells','games','fit','cv','target'))
    n=len(cells);ar=np.arange(n);surface=ROOT/'aug-budget-surface-v2/policies.npz'
    plan=dict(source_sha256=digest(Path(__file__)),surface_sha256=digest(surface),sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),budgets=BUDGETS.tolist(),cap=512,
        methods=['fixed512','fixed1000','time_raw_p1','time_raw_p05','time_cell_p1'],
        control='Each time arm has a within-cell and game-fold shuffled-time null, identical budget histograms within partitions. Same calibrated budget policies for all arms.',
        fitting='Proportional predicted seconds, quantized at geometric midpoints of budget ladder. Scale uses training-fold root predictions only (no labels) to average at most512 nominal simulations. Powers1/.5 and within-cell normalization selected by gameCV, all reported on separate August confirmation. Budget policies independently fold-fit.',
        limitations='Cached-prefix stopping only. Modern root-coverage tree and soft Bellman values, not released Allie traversal. Actual nodes can differ despite identical nominal distributions. August potentially training-seen, CM pending golden. No target/future/outcome in allocation.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    with np.load(surface) as f:
        for key,x in [('games',games),('target',target),('budgets',BUDGETS)]:np.testing.assert_array_equal(f[key],x)
        policy,root,nodes=(f[k] for k in ('policy','root','nodes'))
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    seconds=softmax(root[:,2350:2413].astype(float),axis=1)@tc;assert (seconds>0).all()
    perm=permutation(cells,np.where(fm,cv,3))
    menu=[('fixed512',None,0),('fixed1000',None,0),('time_raw_p1',False,1.),('time_raw_p05',False,.5),('time_cell_p1',True,1.)]
    records,losses,choices={},{},{}
    for name,normal,power in menu:
        arms=[name] if normal is None else [name,name+'_shuffled']
        pars=[];oofs={a:np.full(n,np.nan) for a in arms};oofcost={a:np.full(n,np.nan) for a in arms}
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            if normal is None:ix=np.full(n,2 if name=='fixed512' else 3);pinfo=dict(fold=fold)
            else:
                signal=seconds**power;div=np.ones(16)
                if normal:div=np.array([signal[train&(cells==c)].mean() for c in range(16)]);signal=signal/div[cells]
                ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
                scale=calibrate(signal[train],w);ix=choose(signal,scale)
                pinfo=dict(fold=fold,power=power,divisor_by_cell=div.tolist(),scale=scale)
            pars.append(pinfo)
            for arm in arms:
                a=ix[perm] if arm.endswith('_shuffled') else ix
                prob=policy[a,vi,ar].astype(float);prob/=prob.sum(1,keepdims=True)
                loss=-np.log(prob[ar,target]);cost=nodes[a,ar]
                if fold is None:losses[arm]=loss;choices[arm]=a
                else:
                    take=fm&(cv==fold);oofs[arm][take]=loss[take];oofcost[arm][take]=cost[take]
        for arm in arms:
            conf=means(losses[arm][~fm],cells[~fm]);cc=means(oofs[arm][fm],cells[fm]);a=choices[arm]
            cost=means(nodes[a,ar][~fm],cells[~fm]);nom=means(BUDGETS[a][~fm],cells[~fm])
            records[arm]=dict(parameters=pars,training_equivalent_cm=None,
                confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()),
                fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean()),mean_nodes=float(means(oofcost[arm][fm],cells[fm]).mean())),
                mean_nodes=float(cost.mean()),expert_mean_nodes=float(cost[3::4].mean()),nominal_mean=float(nom.mean()),nominal_by_cell=nom.tolist(),
                budget_fractions={str(b):float(np.mean(BUDGETS[a][~fm]==b)) for b in BUDGETS})
        if normal is not None:
            for take in (fm,~fm):
                for c in range(16):
                    t=take&(cells==c);np.testing.assert_array_equal(np.sort(choices[name][t]),np.sort(choices[name+'_shuffled'][t]))
    _,gi=np.unique(games[~fm],return_inverse=True);ng=gi.max()+1
    counts=np.zeros((ng,16));np.add.at(counts,(gi,cells[~fm]),1)
    boot=np.random.default_rng(24693).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=boot@counts;assert (den>0).all()
    for name,r in records.items():
        refs=['fixed512']
        if name.startswith('time_') and not name.endswith('_shuffled'):refs.append(name+'_shuffled')
        for ref in refs:
            delta=losses[name]-losses[ref];dc=means(delta[~fm],cells[~fm]);s=np.zeros((ng,16));np.add.at(s,(gi,cells[~fm]),delta[~fm]);bs=boot@s/den
            r['delta_vs_'+ref]=dict(macro=float(dc.mean()),expert=float(dc[3::4].mean()),macro_ci95=np.quantile(bs.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(bs[:,3::4].mean(1),[.025,.975]).tolist())
    eligible=[n for n in records if n!='fixed1000' and not n.endswith('_shuffled')]
    selected={m:min(eligible,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    result=dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp),predicted_seconds_quantiles=np.quantile(seconds,[0,.1,.5,.9,.99,1]).tolist())
    atomic(out/'results.json',result)
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),choice=np.stack(list(choices.values())),cells=cells,games=games,fit=fm)
    for name,r in records.items():print(name,r['confirmation']['macro_ce'],r['confirmation']['expert_ce'],r['mean_nodes'],r['fit_game_cv'],flush=True)
    print('SELECTED',selected,flush=True)
    return dict(output=str(out),selected=selected,seconds=result['seconds'])

if __name__=='__main__':run()
