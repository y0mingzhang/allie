"""Selective depth with nested game-cross-fitted router targets, no new NN calls."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .scaled_count_native import load as backup_load
from .budget_surface import counts
from .stack_fit import fit_stack
from .analyze_august import means
from .value_of_compute import choose,solve,design

BUDGETS=np.array([128,256,512,1000,4000,16000])


def atomic_npz(path,**values):
    temp=path.with_suffix('.partial')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)


def run(oracle,spec):
    start=time.monotonic();source=ROOT/'aug-deep-scale-v1';out=ROOT/'aug-deep-frontier-v1';out.mkdir(exist_ok=True)
    assert json.loads((source/'worker.json').read_text())['completed']==4096
    d=read(source.name);rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n,k=mask.shape;ar=np.arange(n)
    plan=dict(budgets=BUDGETS.tolist(),sample_sha256=digest(source/'sample.json'),collection_plan_sha256=digest(source/'plan.json'),
        sources={f:digest(Path(__file__).with_name(f)) for f in ('deep_frontier.py','scaled_count.cpp','diff_backup.cpp','stack_fit.py','value_of_compute.py')},
        backup='Current count normalization16 through1000 simulations;16*B/1000 above1000. Tau=.2*(1+(descendants-1)/scale)^-.5. Lower budgets are exact birth-prefix snapshots from the same independent-root-action traces.',
        menus='Fixed128/256/512/1000/4000/16000, plus Elo and state ridge.1 routers at nominal mean caps512/1000/2000. Choose from development only; no golden tuning.',
        validation='Nested game cross-fitting: each router training row receives target losses from policy calibration excluding its game fold. For an outer held-out game fold, inner calibration excludes both that outer fold and the router-training row fold. Router state inputs use raw model/root128 predictions only, never fitted policy outputs or future search. Full-fit router targets are original3fold out-of-fold losses. Final checking games are separate, reused August development.',
        router_features='Elo group; state adds format, root prior entropy,128-Q spread,ply,model-predicted thinking time,pre-move remaining clock,low-time flag,clock missingness. No actual think time,played move,future values,or external memory.',
        cost='Choose with nominal simulations; report actual neural-node means by cell and expert subset. Root prefills identical. Any candidate needs a live mixed-budget test because cached NN outputs can depend on batching. Cost-matched randomized fixed controls compare expected losses, not policy ensembles.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    reducer=backup_load()
    for path in sorted(source.glob('[0-9]*.npz')):
        dest=out/path.name
        if dest.exists():continue
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            data={key:f[key] for key in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            q=np.stack([reducer.Backup(data,int(b),16*max(1,b/1000)).reduce(np.log(.2),-.5,ids[lo:hi].astype(np.int32))[0] for b in BUDGETS])
            nc=counts(data,BUDGETS)
            np.testing.assert_array_equal(q[3],f['q'][0,1]);np.testing.assert_array_equal(q[4:],f['q'][1:,2])
            np.testing.assert_array_equal(nc[3:],f['nodes'])
            atomic_npz(dest,q=q,nodes=nc,root=f['z'],game=f['game'],ply=f['ply'])
        print('frontier-prefixes',hi,n,flush=True)
    q=np.zeros((6,n,k));nodes=np.zeros((6,n));root=np.zeros((n,2432))
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);q[:,lo:hi]=f['q'];nodes[:,lo:hi]=f['nodes'];root[lo:hi]=f['root']
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text());assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:side=f['feats']
    seconds=np.array([side[r['row'],r['column']-1,0] for r in rows])
    # Held-out/router-target loss arrays: outer0 means the full training set;
    # outer1..3 leave out original game folds0..2.
    served=np.zeros((4,6,n));training_targets=np.full((4,6,n),np.nan)
    for oi,outer in enumerate([None,0,1,2]):
        train=fm if outer is None else fm&(cv!=outer)
        for bi,budget in enumerate(BUDGETS):
            path=out/f'calibration-{oi}-{budget}.npz';meta=out/f'calibration-{oi}-{budget}.json'
            if path.exists() and meta.exists():
                with np.load(path) as f:served[oi,bi]=f['served'];training_targets[oi,bi]=f['targets']
                continue
            rec,ll,policies=fit_stack(rows,root,q[bi],ids,mask,target,cells,train,cv,seconds)
            targets=np.full(n,np.nan)
            for fold in range(3):
                take=train&(cv==fold)
                if not take.any():continue
                assert not set(games[take])&set(games[train&(cv!=fold)])
                targets[take]=-np.log(policies[fold+1][ar[take],target[take]])
            assert np.isfinite(targets[train]).all()
            served[oi,bi]=ll;training_targets[oi,bi]=targets
            atomic_npz(path,served=ll,targets=targets);atomic(meta,rec)
        print('frontier-calibration-outer',outer,flush=True)
    # Raw root features avoid target-dependent calibration in the router inputs.
    z=np.where(mask,root[:,378:2346][ar[:,None],ids],-np.inf);prior=softmax(z,axis=1)
    entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    qm=(prior*q[0]).sum(1);spread=np.sqrt((prior*(q[0]-qm[:,None])**2).sum(1))
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)];known=seconds>=0
    feat=np.c_[entropy,np.log(.01+spread),[len(r['prefix'])-11 for r in rows],np.log1p(softmax(root[:,2350:2413],axis=1)@tc),
               np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known]
    menus=[(f'fixed{b}','fixed',int(b)) for b in BUDGETS]+[(f'{kind}{cap}',kind,cap) for cap in (512,1000,2000) for kind in ('elo','state')]
    records,losses,choices={}, {}, {}
    for name,kind,setting in menus:
        parameters=[];oof=np.full(n,np.nan);oofnodes=np.full(n,np.nan)
        for oi,outer in enumerate([None,0,1,2]):
            train=fm if outer is None else fm&(cv!=outer)
            if kind=='fixed':
                selected=np.full(n,int(np.flatnonzero(BUDGETS==setting)[0]));pars=dict(outer=outer,budget=setting)
            else:
                x,mean,scale=design(cells,feat,train,kind);ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]];w/=w.sum()
                coef,penalty=solve(x[train],training_targets[oi,:,train],w,.1,BUDGETS,setting)
                selected=choose(x,coef,penalty,BUDGETS)
                pars=dict(outer=outer,kind=kind,cap=setting,ridge=.1,coef=coef.tolist(),penalty=penalty,mean=mean.tolist(),scale=scale.tolist())
            ll=served[oi,selected,ar];cost=nodes[selected,ar];parameters.append(pars)
            if outer is None:losses[name]=ll;choices[name]=selected
            else:
                take=fm&(cv==outer);oof[take]=ll[take];oofnodes[take]=cost[take]
        a,b=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm]);selection=choices[name]
        nc=means(nodes[selection,ar][~fm],cells[~fm]);fc=means(oofnodes[fm],cells[fm])
        records[name]=dict(parameters=parameters,training_equivalent_cm=None,mean_nodes=float(nc.mean()),expert_mean_nodes=float(nc[3::4].mean()),
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean()),mean_nodes=float(fc.mean())),
            budget_fractions={str(b):float((BUDGETS[selection][~fm]==b).mean()) for b in BUDGETS})
        print(name,records[name]['confirmation']['macro_ce'],records[name]['confirmation']['expert_ce'],records[name]['mean_nodes'],flush=True)
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(429614).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        rec['comparisons']={}
        refs={'fixed1000':losses['fixed1000']}
        if not name.startswith('fixed'):
            costs=np.array([records[f'fixed{b}']['mean_nodes'] for b in BUDGETS]);cost=rec['mean_nodes']
            upper=min(5,max(1,int(np.searchsorted(costs,cost))))
            lower=upper-1;fraction=(cost-costs[lower])/(costs[upper]-costs[lower]);assert -.001<=fraction<=1.001
            fraction=float(np.clip(fraction,0,1));lo,hi=int(BUDGETS[lower]),int(BUDGETS[upper])
            refs['equal_expected_nodes']=(1-fraction)*losses[f'fixed{lo}']+fraction*losses[f'fixed{hi}']
            rec['randomized_control']=dict(lower=lo,upper=hi,upper_probability=fraction,mean_nodes=float((1-fraction)*costs[lower]+fraction*costs[upper]),semantics='Independent coin selects one fixed-budget execution; reference is expected CE, not the CE of an ensembled policy.')
        for key,base in refs.items():
            delta=losses[name]-base;pt=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
            rec['comparisons'][key]=dict(macro=float(pt.mean()),expert=float(pt[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    selected={str(cap):{metric:min([name for name,kind,setting in menus if (kind!='fixed' and setting==cap) or (kind=='fixed' and setting<=cap)],key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')} for cap in (512,1000,2000)}
    result=dict(results=records,fit_cv_selected_by_cap=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp));atomic(out/'results.json',result)
    atomic_npz(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),choices=np.stack(list(choices.values())),cells=cells,games=games,fit=fm,cv=cv)
    print('SELECTED',selected,flush=True)
    return dict(study=out.name,fit_cv_selected_by_cap=selected,seconds=result['seconds'])
