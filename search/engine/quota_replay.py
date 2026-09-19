"""Screen value-instability quotas using independent cached branch traces."""
import hashlib,importlib,json,subprocess,sys,sysconfig,time
from pathlib import Path
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .diff_backup_native import load as backup_load
from .native_board import from_prefix
from .stack_fit import fit_stack
from .analyze_august import means


def load():
    import pybind11
    src=Path(__file__).with_suffix('.cpp');key={p.name:digest(p) for p in (src,src.with_name('diff_backup.cpp'))}|dict(python=sys.version)
    out=ROOT/'runtime/quota-replay';out.mkdir(exist_ok=True)
    target=out/('_allie_quota_replay'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=out/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(src),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-o',str(temp)],check=True)
        temp.replace(target);atomic(stamp,key)
    sys.path.insert(0,str(out));return importlib.import_module('_allie_quota_replay')


def run(oracle,spec):
    if not spec.get('smoke'):
        receipt=ROOT/'engine-queue/096-quota-replay-smoke.result.json'
        assert receipt.exists() and json.loads(receipt.read_text()).get('smoke')=='passed','Require completed replay smoke'
    start=time.monotonic();native=load();reducer=backup_load();source=ROOT/'aug-deep-scale-v1'
    out=ROOT/'aug-quota-replay-v1';out.mkdir(exist_ok=True);d=read(source.name)
    n=len(d['rows']);assert n==4096 and json.loads((source/'worker.json').read_text())['completed']==n
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    names=['baseline','instability','sample_scaled'];plan=dict(sample_sha256=digest(source/'sample.json'),source_plan_sha256=digest(source/'plan.json'),
        sources={f:digest(Path(__file__).with_name(f)) for f in ('quota_replay.py','quota_replay.cpp','diff_backup.cpp','stack_fit.py')},methods=names,
        formula='First256simulations unchanged. Then sqrt(p*(1-p))*sigma/(1+visits), sigma=sqrt(.03^2+(Q256-Q128)^2); sample_scaled additionally multiplies sigma by sqrt(1+branch_visits_at256). Normalize sigma by prior-weighted RMS and clip multiplier to[.25,4].1000total sims. No targets or later values enter allocation.',
        proof='Reconstruct original16K per-branch pull schedule in native double math and board legal order; every recorded birth must belong to its branch. Retain only a prefix of each chosen branch. Baseline must reproduce same-batch1000 Q and node counts exactly. Reject any cache-exhausting arm instead of silently censoring it. Independent subtrees make this conditional on the cached NN outputs; a live rerun is required because GPU batching can alter predictions.',
        validation='Same4096 August cohort, independently refitted modern output pipeline inside game CV. Reused development. No golden conversion or new NN queries. Actual retained NN nodes reported, not assumed equal from simulations.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    for path in sorted(source.glob('[0-9]*.npz')):
        dest=out/path.name
        if dest.exists():continue
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi])
            data={k:f[k] for k in ('parent','move','depth','born','degree','prior','boot','mass','terminal','roots')}
            order=[[t-378 for t in from_prefix(r['prefix']).legal()] for r in rows[lo:hi]]
            replay=native.Replay(data,f['z'].astype(float),order,16000);ii=ids[lo:hi].astype(np.int32)
            q128=reducer.Backup(data,128).reduce(np.log(.2),-.5,ii)[0]
            q256=reducer.Backup(data,256).reduce(np.log(.2),-.5,ii)[0]
            values=[];counts=[]
            for mode in range(3):
                cut=replay.cut(q128,q256,ii,1000,mode)
                q=reducer.Backup(data|dict(born=cut['born']),0).reduce(np.log(.2),-.5,ii)[0]
                if mode==0:
                    np.testing.assert_array_equal(q,f['q'][0,1]);np.testing.assert_array_equal(cut['nodes'],f['nodes'][0])
                values.append(q);counts.append(cut['nodes'])
            temp=dest.with_suffix('.partial')
            with temp.open('wb') as stream:np.savez_compressed(stream,q=values,nodes=counts,z=f['z'],game=f['game'],ply=f['ply'])
            temp.replace(dest)
        print('quota-replay',hi,n,flush=True)
        if spec.get('smoke'):return dict(smoke='passed',positions=hi-lo,seconds=time.monotonic()-start)
    q=np.zeros((3,n,mask.shape[1]));nodes=np.zeros((3,n));root=np.zeros((n,2432))
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);q[:,lo:hi]=f['q'];nodes[:,lo:hi]=f['nodes'];root[lo:hi]=f['z']
    inventory=ROOT/'aug-tune-v1';manifest=json.loads((inventory/'manifest.json').read_text());assert digest(inventory/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inventory/'feats.npz') as f:feats=f['feats']
    seconds=np.array([feats[r['row'],r['column']-1,0] for r in rows]);records,losses={},{}
    for i,name in enumerate(names):
        rec,ll,_=fit_stack(rows,root,q[i],ids,mask,target,cells,fm,cv,seconds);nc=means(nodes[i],cells)
        rec.update(mean_nodes=float(nc.mean()),expert_mean_nodes=float(nc[3::4].mean()),total_nn_queries=int(nodes[i].sum()))
        records[name]=rec;losses[name]=ll
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(74115).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['baseline'];pt=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_parent']=dict(macro=float(pt.mean()),expert=float(pt[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    result=dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp));atomic(out/'results.json',result)
    np.savez_compressed(out/'scores.npz',names=names,loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    return dict(study=out.name,fit_cv_selected=selected,seconds=result['seconds'])
