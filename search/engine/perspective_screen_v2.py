"""Modern, cost-identical own/opponent selectivity ablation on cached trees."""
import json,time
from pathlib import Path
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .perspective_native import load,test
from .stack_fit import fit_stack
from .analyze_august import means


def run(oracle,spec):
    module=load();test(module);start=time.monotonic()
    out=ROOT/'aug-perspective-v2';out.mkdir(exist_ok=True)
    source=ROOT/'aug-expanded-search-v1';surface=ROOT/'aug-budget-surface-v2'
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    variants=[('symmetric',1.,1.),('opponent_hot',1.,2.),('opponent_cold',1.,.5),('own_hot',2.,1.),('own_cold',.5,1.),('opponent_prior',1.,'inf')]
    plan=dict(variants=variants,budget=1000,base_tau=.2,count_exponent=-.5,
        sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),surface_sha256=digest(surface/'results.json'),
        sources={f:digest(Path(__file__).with_name(f)) for f in ('perspective_screen_v2.py','perspective_backup.cpp','perspective_native.py','diff_backup.cpp','stack_fit.py')},
        input_blocks={p.name:digest(p) for p in sorted(source.glob('[0-9]*.npz'))},
        hypothesis='Root player and opponent need not use equally selective continuation policies. Higher tau approaches the human-policy expectation; lower tau approaches best response. Change one side only. Same root coverage tree, same model, no extra NN queries. Prior asymmetric grid used a weaker output recipe and constant temperature; this is a bounded recheck with current count-dependent backup.',
        validation='Independent recursive tests, baseline identity and side-parity test first. Refit the identical modern output pipeline within each of three training-game folds. Selection uses fit CV only. August reporting is reused development, not fresh confirmation or golden CM.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games);root=f['root'];nodes=f['nodes'][-1];base=f['q'][-1]
    q=np.zeros((len(variants),*mask.shape));q[0]=base
    for path in sorted(source.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            data={k:f[k] for k in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
        b=module.Backup(data,1000)
        np.testing.assert_allclose(b.reduce(1,1,ids[lo:hi].astype(np.int32)),base[lo:hi],atol=2e-14)
        for j,(_,own,opp) in enumerate(variants[1:],1):
            q[j,lo:hi]=b.reduce(own,float(opp),ids[lo:hi].astype(np.int32))
        assert np.isfinite(q[:,lo:hi]).all()
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text())
    assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:feats=f['feats']
    seconds=np.array([feats[r['row'],r['column']-1,0] for r in rows])
    records,losses={},{}
    for j,(name,own,opp) in enumerate(variants):
        rec,ll,_=fit_stack(rows,root,q[j],ids,mask,target,cells,fm,cv,seconds)
        rec.update(mean_nodes=float(nodes.mean()),own_multiplier=own,opponent_multiplier=opp)
        records[name]=rec;losses[name]=ll
        print(name,rec['confirmation']['macro_ce'],rec['confirmation']['expert_ce'],rec['fit_game_cv'],flush=True)
    ref=json.loads((surface/'results.json').read_text())['results']['1000_temperature']
    drift={}
    for split in ('confirmation','fit_game_cv'):
        for metric in ('macro_ce','expert_ce'):
            drift[split+'/'+metric]=records['symmetric'][split][metric]-ref[split][metric]
            np.testing.assert_allclose(records['symmetric'][split][metric],ref[split][metric],atol=1e-9,rtol=0)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    draws=np.random.default_rng(95185).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@count
    for name,rec in records.items():
        delta=losses[name]-losses['symmetric'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_symmetric']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,baseline_calibration_drift=drift,seconds=time.monotonic()-start,plan_sha256=digest(pp)))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)
    return dict(study=out.name,fit_cv_selected=selected)
