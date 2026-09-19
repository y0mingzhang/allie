"""Development-only attribution of critic consistency; fixed lambda3."""
import json,time
from pathlib import Path
import numpy as np
from .bellman_modes_test import load,test
from .engine.service import ROOT,GLOBAL_STOP,atomic
from .engine.balanced_eval import digest
from .engine.sample_data import read
from .engine.stack_fit import fit_stack
from .engine.analyze_august import means

OUT=ROOT/'aug-bellman-mechanisms-v1'
ARMS=[('unchanged',0.,0),('full',3.,0),('root_only',3.,1),('no_root',3.,2),('upward_only',3.,3)]


def main():
    begin=time.monotonic();OUT.mkdir(exist_ok=True)
    source=ROOT/'aug-expanded-search-v1';reference=ROOT/'aug-bellman-projection-expanded-v1'
    d=read('aug-tune-expanded-v1');rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    plan=dict(arms=ARMS,budget=1000,sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        reference_sha256=digest(reference/'results.json'),tree_plan_sha256=digest(source/'plan.json'),
        sources={str(p.relative_to(Path(__file__).parent)):digest(p) for p in (
            Path(__file__),Path(__file__).with_name('bellman_modes_test.py'),
            *[Path(__file__).parent/'engine'/s for s in ('bellman_modes.cpp','bellman_projection.cpp','diff_backup.cpp','stack_fit.py')])},
        mechanisms='Lambda3 fixed from prior August selection. full=all consistency factors and both passes; root_only=only root factors; no_root=every factor except roots; upward_only=each node uses only its own subtree, with no ancestor correction. Same initial Gaussian measurement terms and unknown-mass noise. Unchanged/full controls must recover prior action values and fitted metrics.',
        evaluation='16384 reused August positions, game-disjoint fit/check and3gameCV. All5 arms output-calibrated inside identical game folds. No golden reading, new NN, outcomes, or external memory. Full shares current known development data; this is attribution, not independent confirmation.',
        cost='Same actual1000sim expansion and model queries. Report per-arm CPU projection+backup separately from cache loading.')
    plan=json.loads(json.dumps(plan));pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    native=load();test(native);n=len(rows)
    q=np.zeros((len(ARMS),*mask.shape));root=np.zeros((n,2432));nodes=np.zeros(n);diagnostics=[]
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
            compact.update(parent=par,roots=mapping[data['roots']]);worker=native.Modes(compact,1000)
            values=[];stats=[]
            for name,strength,mode in ARMS:
                t=time.monotonic();info=worker.project(strength,mode);values.append(worker.reduce(np.log(.2),-.5,ids[lo:hi].astype(np.int32))[0])
                stats.append(dict(name=name,seconds=time.monotonic()-t,clipped=int(info['clipped']),active=int(info['active'])))
            with np.load(reference/path.name) as f:
                np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(zz,f['root'])
                np.testing.assert_array_equal(values[0],f['q'][0]);np.testing.assert_array_equal(values[1],f['q'][2]);nn=f['nodes']
            tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,q=values,root=zz,nodes=nn,game=games[lo:hi],stats=json.dumps(stats),elapsed=time.monotonic()-tick)
            tmp.replace(dest)
        with np.load(dest) as f:
            hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            q[:,lo:hi]=f['q'];root[lo:hi]=f['root'];nodes[lo:hi]=f['nodes'];diagnostics.append(json.loads(str(f['stats'])))
        print('mechanisms',hi,n,flush=True)
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
    for now,previous in [('unchanged','unchanged'),('full','lambda3')]:
        for field in ('confirmation','fit_game_cv'):
            for m in ('macro_ce','expert_ce'):np.testing.assert_allclose(records[now][field][m],ref[previous][field][m],rtol=0,atol=1e-9)
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    draws=np.random.default_rng(427580).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@count;assert (den>0).all()
    for name,rec in records.items():
        for parent in ('unchanged','full'):
            delta=losses[name]-losses[parent];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
            rec['delta_vs_'+parent]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
                macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(),cells=point.tolist())
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    atomic(OUT/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-begin,plan_sha256=digest(pp)))
    np.savez_compressed(OUT/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('MECHANISM SELECTED',selected,flush=True)


if __name__=='__main__':main()
