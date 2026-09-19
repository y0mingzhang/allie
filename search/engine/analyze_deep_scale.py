"""Same-game inference scaling curves, independently calibrated inside game CV."""
import json,time
from pathlib import Path
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .stack_fit import fit_stack
from .analyze_august import means


def run(oracle,spec):
    start=time.monotonic();out=ROOT/'aug-deep-scale-v1'
    worker=json.loads((out/'worker.json').read_text());plan=json.loads((out/'plan.json').read_text())
    limit=int(spec.get('limit',4096));assert limit in (2048,4096)
    assert worker['completed']>=limit and worker['positions']==4096
    tag='' if limit==4096 else f'-prefix{limit}'
    assert worker['plan_sha256']==digest(out/'plan.json')
    d={k:v[:limit] for k,v in read(out.name).items()}
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n,k=mask.shape;root=np.zeros((n,2432));q=np.zeros((3,3,n,k));nodes=np.zeros((3,n));seen=np.zeros(n,bool)
    stats=[]
    for path in sorted(out.glob('[0-9]*.npz')):
        if int(path.stem)>=limit:continue
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);assert not seen[lo:hi].any()
            np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ids'],ids[lo:hi])
            np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
            root[lo:hi]=f['z'];q[:,:,lo:hi]=f['q'];nodes[:,lo:hi]=f['nodes'];seen[lo:hi]=True
            stats.append(json.loads(str(f['stats'])))
    assert seen.all();np.testing.assert_array_equal(q[0,1],q[0,2])
    report_plan=dict(positions=limit,partial_sample='First N positions of the score-independent interleaved cohort; macro still averages per-cell losses equally.',source_sha256=digest(Path(__file__)),collection_plan_sha256=digest(out/'plan.json'),
        sample_sha256=digest(out/'sample.json'),fit_source_sha256=digest(Path(__file__).with_name('stack_fit.py')),
        validation='Identical current output pipeline fitted separately within three game folds for each arm. Report all budgets/temperature rules. Pick by fit CV only; paired whole-game bootstrap on reused game-disjoint development. No golden conversion.',
        cost='Each budget shares the same growing tree. Measured cumulative stage search time plus root prefill; backup/calibration/serialization separately. Count actual nonterminal NN queries and report expert costs. Higher budget alone does not establish algorithmic Pareto gain.')
    pp=out/f'analysis{tag}-plan.json'
    if pp.exists():assert json.loads(pp.read_text())==report_plan
    else:atomic(pp,report_plan)
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text())
    assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:feats=f['feats']
    seconds=np.array([feats[r['row'],r['column']-1,0] for r in rows])
    records,losses={},{}
    for bi,budget in enumerate(plan['budgets']):
        for mi,method in enumerate(plan['methods']):
            name=f'{budget}_{method}'
            # At1000 the normalized and current rules are mathematically identical.
            if bi==0 and mi==2:continue
            rec,ll,_=fit_stack(rows,root,q[bi,mi],ids,mask,target,cells,fm,cv,seconds)
            nc=means(nodes[bi],cells)
            rec.update(budget=budget,method=method,mean_nodes=float(nc.mean()),expert_mean_nodes=float(nc[3::4].mean()),
                search_seconds=sum(s['prefill_seconds']+sum(t['search_seconds'] for t in s['stages'][:bi+1]) for s in stats),
                nonroot_nn_queries=int(nodes[bi].sum()),prefill_tokens=sum(s['prefill_tokens'] for s in stats))
            records[name]=rec;losses[name]=ll
            print(name,rec['confirmation']['macro_ce'],rec['confirmation']['expert_ce'],rec['fit_game_cv'],flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    draws=np.random.default_rng(99622).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@count;assert (den>0).all()
    for name,rec in records.items():
        rec['comparisons']={}
        references=['1000_count_decay']
        if rec['budget']!=1000:references.append(f"{rec['budget']}_count_decay")
        for ref in references:
            delta=losses[name]-losses[ref];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16))
            np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
            rec['comparisons'][ref]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
                macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    result=dict(results=records,fit_cv_selected=selected,analysis_seconds=time.monotonic()-start,analysis_plan_sha256=digest(pp))
    atomic(out/f'results{tag}.json',result)
    np.savez_compressed(out/f'scores{tag}.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    np.savez_compressed(out/f'root-values{tag}.npz',root=root,q=q,nodes=nodes,cells=cells,games=games,fit=fm,cv=cv,ids=ids,mask=mask,target=target)
    print('SELECTED',selected,flush=True)
    return dict(study=out.name,positions=limit,fit_cv_selected=selected,analysis_seconds=result['analysis_seconds'])
