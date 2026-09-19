"""Output-only critic consistency, on existing immutable query-local trees."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .bellman_projection_test import load, test
from .engine.service import ROOT, GLOBAL_STOP, atomic
from .engine.balanced_eval import digest
from .engine.sample_data import read
from .engine.stack_fit import fit_stack
from .engine.analyze_august import means

STRENGTHS = (0., .3, 1., 3.)
OUT = ROOT / 'aug-bellman-projection-v1'


def main(expanded=False):
    start=time.monotonic()
    out=ROOT/'aug-bellman-projection-expanded-v1' if expanded else OUT
    strengths=(0.,1.,3.,10.) if expanded else STRENGTHS
    out.mkdir(exist_ok=True)
    source=ROOT/('aug-expanded-search-v1' if expanded else 'aug-deep-scale-v1')
    sample=ROOT/('aug-tune-expanded-v1' if expanded else 'aug-deep-scale-v1')
    reference_dir=ROOT/'aug-budget-surface-v2' if expanded else source
    reference_name='1000_temperature' if expanded else '1000_count_decay'
    d=read(sample.name)
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    plan=dict(strengths=strengths,budget=1000,sample_sha256=digest(sample/'sample.json'),
        tree_plan_sha256=digest(source/'plan.json'),reference_sha256=digest(reference_dir/'results.json'),
        expanded=expanded,prior_screen_sha256=digest(OUT/'results.json') if expanded else None,
        sources={str(p.relative_to(Path(__file__).parent)):digest(p) for p in (
            Path(__file__),Path(__file__).with_name('bellman_projection_test.py'),
            Path(__file__).parent/'engine/bellman_projection.cpp',Path(__file__).parent/'engine/diff_backup.cpp',
            Path(__file__).parent/'engine/stack_fit.py')},
        formula='Mover-perspective y=-boot. Objective sum_nonterminal(v-y)^2 + sum_expanded (v_i+sum_j p_ij*v_j-r_i*y_i)^2/(1/lambda+r_i^2). Terminal values fixed; original actual-Elo legal priors. r is unexpanded prior mass. Unknown aggregate has mean y_i and variance1. Exact Gaussian factor-tree mean; clip posterior to[-1,1] before original soft backup. Lambda0 must recover current backup exactly.',
        limits='Independent-noise assumptions are not calibrated confidence. Correlated network errors and self-referential unseen fallback may defeat denoising. Report clipping and residuals by coverage/time-control. This fixed checkpoint has no clock input.',
        evaluation=f'Reused August{len(rows)} with original cached hardware, controls from these identical trees.3fold gameCV and game-disjoint checking, potentially training-seen. Four fixed strengths; no neural weight updates, outcome labels, external memory, new GPU calls or golden conversion. Expanded study includes the smaller cohort; it is expanded development, not fresh confirmation.',
        cost='Output-only O(nodes) processing after unchanged raw-critic search; same actual node count, CPU processing time additional. No hypothetical rerun or tree changes.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():
        saved=json.loads(pp.read_text())
        if saved!=plan:
            revision=json.loads((out/'analysis-revision.json').read_text())
            assert digest(pp)==revision['original_plan_sha256']
            assert saved|{'sources':plan['sources']}==plan
            assert plan['sources']==revision['sources']
    else:atomic(pp,plan)
    native=load();test(native)
    n=len(rows);q=np.zeros((len(strengths),*mask.shape));root=np.zeros((n,2432));nodes=np.zeros(n)
    if expanded:
        with np.load(reference_dir/'policies.npz') as f:
            np.testing.assert_array_equal(f['games'],games)
            reference_q=f['q'][-1];reference_nodes=f['nodes'][-1]
    root_residual=np.zeros(n);root_tail=np.zeros(n);diagnostics=[]
    for path in sorted(source.glob('[0-9]*.npz')):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        lo=int(path.stem);cache=out/path.name;tick=time.monotonic()
        if not cache.exists():
            with np.load(path) as f:
                hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
                np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
                data={k:f[k] for k in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
                zz=f['z']
                nn=reference_nodes[lo:hi] if expanded else f['nodes'][0]
                reference=reference_q[lo:hi] if expanded else f['q'][0,1]
            active=data['born']<=1000;active[data['roots']]=True
            mapping=np.full(len(active),-1,np.int32);mapping[active]=np.arange(active.sum())
            roots=mapping[data['roots']];par=data['parent'][active];positive=par>=0
            par[positive]=mapping[par[positive]];assert (par[positive]>=0).all()
            compact={k:v[active] for k,v in data.items() if k not in ('roots','parent')}
            compact.update(roots=roots,parent=par)
            y=-compact['boot'].copy();term=compact['terminal']>=0;y[term]=np.where(compact['terminal'][term]==.5,0.,-1.)
            observed=np.bincount(par[positive],weights=compact['prior'][positive],minlength=len(par))
            count=np.bincount(par[positive],minlength=len(par))
            mass=compact['mass'];tail=np.where(count==compact['degree'],0.,np.maximum(0.,1.-observed/np.maximum(mass,1e-30)))
            child=np.bincount(par[positive],weights=compact['prior'][positive]*y[positive],minlength=len(par))/np.maximum(mass,1e-30)
            residual=y+child-tail*y;residual[(count==0)|term]=0.
            engine=native.Projection(compact,1000);values=[];clip=[];posterior_roots=[];processing=[]
            for strength in strengths:
                t=time.monotonic()
                info=engine.project(strength);values.append(engine.reduce(np.log(.2),-.5,ids[lo:hi].astype(np.int32))[0])
                processing.append(time.monotonic()-t)
                clip.append(int(info['clipped']));posterior_roots.append(info['mean'][roots])
            np.testing.assert_array_equal(values[0],reference)
            internal=(count>0)&~term;err=residual[internal];tails=tail[internal]
            strata=[]
            for lower,upper in ((0.,.1),(.1,.5),(.5,1.00001)):
                take=(tails>=lower)&(tails<upper)
                strata.append(dict(lower=lower,upper=upper,count=int(take.sum()),sum=float(err[take].sum()),sumsq=float((err[take]**2).sum())))
            diag=dict(block=lo,seconds=time.monotonic()-tick,processing_seconds=processing,active_nodes=len(par),clipped=clip,unseen_strata=strata)
            temp=cache.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,q=values,root=zz,nodes=nn,game=games[lo:hi],
                root_residual=residual[roots],root_tail=tail[roots],root_posterior=posterior_roots,diagnostics=json.dumps(diag))
            temp.replace(cache)
        with np.load(cache) as f:
            hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            q[:,lo:hi]=f['q'];root[lo:hi]=f['root'];nodes[lo:hi]=f['nodes']
            root_residual[lo:hi]=f['root_residual'];root_tail[lo:hi]=f['root_tail'];diagnostics.append(json.loads(str(f['diagnostics'])))
        print('Bellman projection',hi,n,flush=True)
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text())
    assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:side=f['feats']
    seconds=np.array([side[r['row'],r['column']-1,0] for r in rows])
    records={};losses={};cost=means(nodes[~fm],cells[~fm])
    for i,strength in enumerate(strengths):
        name='unchanged' if strength==0 else f'lambda{strength:g}'
        rec,ll,_=fit_stack(rows,root,q[i],ids,mask,target,cells,fm,cv,seconds)
        rec.update(strength=strength,mean_nodes=float(cost.mean()),expert_mean_nodes=float(cost[3::4].mean()),
                   cpu_project_and_backup_seconds=sum(a['processing_seconds'][i] for a in diagnostics),
                   clipping_fraction=sum(a['clipped'][i] for a in diagnostics)/sum(a['active_nodes'] for a in diagnostics))
        records[name]=rec;losses[name]=ll
        print('Bellman fitted',name,rec['confirmation'],rec['fit_game_cv'],flush=True)
    ref=json.loads((reference_dir/'results.json').read_text())['results'][reference_name]
    identity_gaps={}
    for field in ('confirmation','fit_game_cv'):
        for metric in ('macro_ce','expert_ce'):
            gap=abs(records['unchanged'][field][metric]-ref[field][metric])
            identity_gaps[field+'/'+metric]=gap
            # Nonlinear fitting under different BLAS thread counts can differ
            # at ~1e-12; cached action-value identity above remains exact.
            np.testing.assert_allclose(records['unchanged'][field][metric],ref[field][metric],rtol=0,atol=1e-9)
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(190982).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['unchanged'];pt=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_parent']=dict(macro=float(pt.mean()),expert=float(pt[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(),cells=pt.tolist())
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    strata={}
    for fmt in range(4):
        for pressure,take in [('all',np.ones(n,bool)),('low_clock',(seconds>=0)&(seconds<=15))]:
            take=take&(cells//4==fmt)&fm
            if take.any():strata[f'format{fmt}_{pressure}']=dict(count=int(take.sum()),mean=float(root_residual[take].mean()),rms=float(np.sqrt((root_residual[take]**2).mean())),unseen_mass=float(root_tail[take].mean()))
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,
        reduction_seconds=sum(a['seconds'] for a in diagnostics),projection_diagnostics=diagnostics,
        fit_root_residual_strata=strata,plan_sha256=digest(pp),baseline_metric_absolute_gaps=identity_gaps,
        analysis_revision_sha256=digest(out/'analysis-revision.json') if (out/'analysis-revision.json').exists() else None))
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('Bellman selected',selected,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expanded',action='store_true')
    main(parser.parse_args().expanded)
