"""Reverse-KL vs forward-KL Bellman values, identical trees and output recipe."""
import json
import time
from pathlib import Path
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .reverse_backup_native import load
from .stack_fit import fit_stack
from .analyze_august import means

def main():
    start=time.monotonic();out=ROOT/'aug-reverse-bellman-v1';out.mkdir(exist_ok=True)
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    variants=[('forward',None,False),('reverse01',.1,False),('reverse02',.2,False),('reverse04',.4,False),('reverse02_visited',.2,True)]
    surface=ROOT/'aug-budget-surface-v2';source=ROOT/'aug-expanded-search-v1'
    plan=dict(variants=variants,count_exponent=-.5,budget=1000,
        sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),surface_sha256=digest(surface/'results.json'),
        input_blocks={p.name:digest(p) for p in sorted(source.glob('[0-9]*.npz'))},
        sources={p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(s) for s in
            ('reverse_backup.cpp','reverse_backup_native.py','diff_backup.cpp','stack_fit.py')]]},
        formula='V=max_pi E_pi[q]-lambda KL(p||pi), restricted to positive-prior support. pi=lambda*p/(nu-q); V=nu-lambda-lambda*sum p log((nu-q)/lambda). Unvisited prior mass takes current node critic, except explicit visited-only diagnostic.',
        fit='Same root Elo alpha/beta, conditional strength mixture and residual temperature, refitted within3 game folds for each backup. Forward control must reproduce the latest corrected-surface recipe. No neural-weight changes, same NN nodes.',
        semantics='CPU-only fixed-tree backup; different returned policies do not change expansion. Reused August tuning and disjoint confirmation, no golden conversion. Report all arms and unseen-mass diagnostics; visited-only is selected-branch renormalization with possible selection bias.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    with np.load(surface/'policies.npz') as f:
        root=f['root'];forward=f['q'][-1];nodes=f['nodes'][-1]
        np.testing.assert_array_equal(f['games'],games)
    module=load();q=np.zeros((len(variants),*mask.shape));q[0]=forward;diagnostics=[];reduction=[]
    for path in sorted(source.glob('[0-9]*.npz')):
        t=time.monotonic()
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            data={k:f[k] for k in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
        reducer=module.Backup(data,1000)
        for j,(_,lam,visited) in enumerate(variants[1:],1):
            q[j,lo:hi]=reducer.reduce(lam,-.5,ids[lo:hi].astype(np.int32),visited)
        assert np.isfinite(q[:,lo:hi]).all()
        assert q[:,lo:hi][np.broadcast_to(mask[None,lo:hi],q[:,lo:hi].shape)].min()>=-1-1e-10
        assert q[:,lo:hi][np.broadcast_to(mask[None,lo:hi],q[:,lo:hi].shape)].max()<=1+1e-10
        par=data['parent'];active=(par>=0)&(data['born']<=1000)
        seen=np.bincount(par[active],weights=data['prior'][active],minlength=len(par))
        child=np.bincount(par[active],minlength=len(par))
        nonleaf=(child>0)&(data['terminal']<0)&(data['mass']>0)
        rest=np.where(child==data['degree'],0.,np.maximum(0.,data['mass']-seen))
        tail=rest[nonleaf]/data['mass'][nonleaf]
        diagnostics.append(dict(block=lo,internal_nodes=int(nonleaf.sum()),unvisited_prior_mass_mean=float(tail.mean()),
            unvisited_prior_mass_quantiles=np.quantile(tail,[0,.1,.5,.9,1]).tolist()))
        reduction.append(time.monotonic()-t);print('reduced',hi,reduction[-1],flush=True)
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text())
    assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:feats=f['feats']
    seconds=np.array([feats[r['row'],r['column']-1,0] for r in rows])
    records,losses,policies={}, {}, []
    for j,(name,lam,visited) in enumerate(variants):
        rec,ll,p=fit_stack(rows,root,q[j],ids,mask,target,cells,fm,cv,seconds)
        rec.update(mean_nodes=float(nodes.mean()),lambda0=lam,visited_only=visited)
        records[name]=rec;losses[name]=ll;policies.append(np.array(p,dtype=np.float32))
        print(name,rec['confirmation']['macro_ce'],rec['confirmation']['expert_ce'],rec['fit_game_cv'],flush=True)
    reference=json.loads((surface/'results.json').read_text())['results']['1000_temperature']
    for field in ('confirmation','fit_game_cv'):
        for metric in ('macro_ce','expert_ce'):np.testing.assert_allclose(records['forward'][field][metric],reference[field][metric],atol=1e-12)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(95834).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count
    for name,rec in records.items():
        delta=losses[name]-losses['forward'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['delta_vs_forward']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,
        reduction_seconds=sum(reduction),unvisited_mass=diagnostics,plan_sha256=digest(pp)))
    np.savez_compressed(out/'policies.npz',names=list(records),policy=np.array(policies),q=q,root=root,ids=ids,mask=mask,target=target,cells=cells,games=games,fit=fm,cv=cv,nodes=nodes)
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
