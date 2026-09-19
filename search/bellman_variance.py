"""Development-only confidence-weighted Bellman projection and outcome diagnostic."""
import json,time
from pathlib import Path
import numpy as np
from .bellman_variance_test import load,test
from .engine.service import ROOT,GLOBAL_STOP,atomic
from .engine.balanced_eval import digest
from .engine.sample_data import read
from .engine.stack_fit import fit_stack
from .engine.analyze_august import means

OUT=ROOT/'aug-bellman-variance-v1'
ARMS=[('unchanged',0.,0),('uniform1',1.,0),('uniform3',3.,0),('uniform10',10.,0),('confidence1',1.,1),('confidence3',3.,1),('confidence10',10.,1)]


def main():
    begin=time.monotonic();OUT.mkdir(exist_ok=True)
    source=ROOT/'aug-expanded-search-v1';reference=ROOT/'aug-bellman-projection-expanded-v1'
    d=read('aug-tune-expanded-v1');rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    labels=ROOT/'aug-value-outcomes-v1';lm=json.loads((labels/'manifest.json').read_text())
    assert digest(labels/'labels.npz')==lm['labels_sha256']
    assert digest(ROOT/'aug-tune-expanded-v1/sample.json')==lm['sample_sha256']
    with np.load(labels/'labels.npz') as f:
        np.testing.assert_array_equal(f['game'],games);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows])
        outcome=np.array([1.,.5,0.])[f['mover_wdl']]
    plan=dict(outcomes_sha256=lm['labels_sha256'],arms=ARMS,budget=1000,sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        reference_sha256=digest(reference/'results.json'),tree_plan_sha256=digest(source/'plan.json'),
        sources={str(p.relative_to(Path(__file__).parent)):digest(p) for p in (
            Path(__file__),Path(__file__).with_name('bellman_variance_test.py'),
            *[Path(__file__).parent/'engine'/s for s in ('bellman_variance.cpp','bellman_projection.cpp','diff_backup.cpp','stack_fit.py')])},
        mechanisms='Uniform measurement variance1 versus max(0.1,1-y*y), where y is raw W-L. The latter is an outcome-variance upper-bound heuristic, NOT calibrated critic-error uncertainty. Unknown-tail variance uses the same measurement variance. Strength1/3/10 in both arms isolates global shrinkage. Same Gaussian factor objective with per-node measurement variances; terminal constraints exact. Expansion and soft backup unchanged.',
        evaluation='16384 reused August positions, game-disjoint fit/check and3gameCV. All7 arms output-calibrated inside identical game folds. No golden reading, new NN or external memory. Existing August outcome sidecar is ONLY for post-hoc root outcome scoring; labels are never prediction inputs. No outcome-based parameter selection. This shares current known development data; this is attribution, not independent confirmation.',
        cost='Same actual1000sim expansion and model queries. Report per-arm CPU projection+backup separately from cache loading.')
    plan=json.loads(json.dumps(plan));pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    native=load();test(native);n=len(rows)
    q=np.zeros((len(ARMS),*mask.shape));root=np.zeros((n,2432));nodes=np.zeros(n);diagnostics=[];values_root=np.zeros((len(ARMS),n));coverage=np.zeros(n);residual=np.zeros(n)
    for path in sorted(source.glob('[0-9]*.npz')):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        lo=int(path.stem);dest=OUT/path.name
        if not dest.exists():
            tick=time.monotonic()
            with np.load(path) as f:
                hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
                data={k:f[k] for k in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')};zz=f['z']
            active=data['born']<=1000;active[data['roots']]=True
            mapping=np.full(len(active),-1,np.int32);mapping[active]=np.arange(active.sum())
            par=data['parent'][active];pos=par>=0;par[pos]=mapping[par[pos]];assert (par[pos]>=0).all()
            compact={k:v[active] for k,v in data.items() if k not in ('parent','roots')}
            compact.update(parent=par,roots=mapping[data['roots']]);worker=native.Weighted(compact,1000)
            values=[];stats=[];rv=[]
            raw=-compact['boot'];ones=np.ones(len(raw))
            heuristic=np.maximum(.1,1-raw**2)
            for name,strength,mode in ARMS:
                t=time.monotonic();info=worker.project(strength,heuristic if mode else ones);rv.append(info['mean'][compact['roots']]);values.append(worker.reduce(np.log(.2),-.5,ids[lo:hi].astype(np.int32))[0])
                stats.append(dict(name=name,seconds=time.monotonic()-t,clipped=int(info['clipped']),active=int(info['active'])))
            with np.load(reference/path.name) as f:
                np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(zz,f['root'])
                np.testing.assert_array_equal(values[0],f['q'][0]);np.testing.assert_array_equal(values[2],f['q'][2]);nn=f['nodes']
            root_ix=compact['roots'];cov=info['coverage'][root_ix]
            child_sum=np.zeros(len(raw));child=compact['parent']>=0
            np.add.at(child_sum,compact['parent'][child],compact['prior'][child]*raw[child])
            resid=cov*raw[root_ix]+child_sum[root_ix]/np.maximum(compact['mass'][root_ix],1e-30)
            tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,q=values,root_values=rv,coverage=cov,residual=resid,root=zz,nodes=nn,game=games[lo:hi],stats=json.dumps(stats),elapsed=time.monotonic()-tick)
            tmp.replace(dest)
        with np.load(dest) as f:
            hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            q[:,lo:hi]=f['q'];root[lo:hi]=f['root'];nodes[lo:hi]=f['nodes'];diagnostics.append(json.loads(str(f['stats'])))
            values_root[:,lo:hi]=f['root_values'];coverage[lo:hi]=f['coverage'];residual[lo:hi]=f['residual']
        print('variance',hi,n,flush=True)
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text());assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:side=f['feats']
    seconds=np.array([side[r['row'],r['column']-1,0] for r in rows])
    cost=means(nodes[~fm],cells[~fm]);records={};losses={}
    for i,(name,strength,mode) in enumerate(ARMS):
        rec,ll,_=fit_stack(rows,root,q[i],ids,mask,target,cells,fm,cv,seconds)
        rec.update(strength=strength,mode=mode,mean_nodes=float(cost.mean()),expert_mean_nodes=float(cost[3::4].mean()),
                   cpu_project_backup_seconds=sum(a[i]['seconds'] for a in diagnostics),
                   clipping_fraction=sum(a[i]['clipped'] for a in diagnostics)/sum(a[i]['active'] for a in diagnostics))
        records[name]=rec;losses[name]=ll
        print(name,rec['confirmation'],rec['fit_game_cv'],flush=True)
    ref=json.loads((reference/'results.json').read_text())['results']
    for now,previous in [('unchanged','unchanged'),('uniform3','lambda3')]:
        for field in ('confirmation','fit_game_cv'):
            for m in ('macro_ce','expert_ce'):np.testing.assert_allclose(records[now][field][m],ref[previous][field][m],rtol=0,atol=1e-9)
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    draws=np.random.default_rng(427580).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@count;assert (den>0).all()
    for name,rec in records.items():
        for parent in ('unchanged','uniform3'):
            delta=losses[name]-losses[parent];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
            rec['delta_vs_'+parent]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
                macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(),cells=point.tolist())
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    # Root-outcome diagnostics use only the checking games. They are not used to pick an arm.
    outcome_records={};score_losses={}
    for i,(name,_,_) in enumerate(ARMS):
        prob=np.clip((values_root[i]+1)/2,1e-8,1-1e-8)
        score_loss=-outcome*np.log(prob)-(1-outcome)*np.log1p(-prob)
        squared=(prob-outcome)**2
        ce=means(score_loss[~fm],cells[~fm]);sq=means(squared[~fm],cells[~fm])
        score_losses[name]=score_loss
        outcome_records[name]=dict(score_ce=float(ce.mean()),expert_score_ce=float(ce[3::4].mean()),
            squared_score_error=float(sq.mean()),expert_squared_score_error=float(sq[3::4].mean()))
    for name,rec in outcome_records.items():
        delta=score_losses[name]-score_losses['unchanged'];sums=np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den;point=means(delta[~fm],cells[~fm])
        rec['delta_vs_raw']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    groups={}
    for label,take in [(f'cell{i}',cells==i) for i in range(16)]+[
        ('low_clock',(seconds>=0)&(seconds<=15)),('high_clock',seconds>15),
        ('coverage_below_half',coverage<.5),('coverage_half_to_90',(coverage>=.5)&(coverage<.9)),('coverage_at_least90',coverage>=.9)]:
        take=take&~fm
        if not take.any():continue
        groups[label]=dict(n=int(take.sum()),coverage_mean=float(coverage[take].mean()),
            residual_mean=float(residual[take].mean()),residual_rms=float(np.sqrt(np.mean(residual[take]**2))),
            move_ce_delta_uniform3=float((losses['uniform3']-losses['unchanged'])[take].mean()))
    atomic(OUT/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-begin,plan_sha256=digest(pp),
        root_outcomes=outcome_records,raw_consistency_diagnostics=groups,
        outcome_semantics='Binary CE of expected score (win1,draw0.5,loss0), not move CE or categorical WDL log loss. Only August checking games, no fits from outcome labels.'))
    np.savez_compressed(OUT/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm,
        root_values=values_root,coverage=coverage,residual=residual)
    print('VARIANCE SELECTED',selected,flush=True)
    print('ROOT OUTCOME',json.dumps(outcome_records),flush=True)


if __name__=='__main__':main()
