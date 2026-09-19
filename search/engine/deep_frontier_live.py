"""Actual two-stage mixed-budget execution of frozen development routers."""
import json,time,os
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .sample_data import read
from .growforest_native import load
from .scaled_count_native import load as backup_load
from .handles import HandleOracle
from .budget_policy import policy,route
from .analyze_august import means

NAMES=('fixed1000','state512','state1000','state2000')


def features(rows,z,q,ids,mask,seconds):
    n=len(rows);ar=np.arange(n)
    p=softmax(np.where(mask,z[:,378:2346][ar[:,None],ids],-np.inf),axis=1)
    entropy=-(p*np.log(np.maximum(p,1e-300))).sum(1)
    qm=(p*q).sum(1);spread=np.sqrt((p*(q-qm[:,None])**2).sum(1))
    tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)];known=seconds>=0
    return np.c_[entropy,np.log(.01+spread),[len(r['prefix'])-11 for r in rows],np.log1p(softmax(z[:,2350:2413],axis=1)@tc),
        np.where(known,np.log1p(np.maximum(seconds,0)),0.),known&(seconds<=15),~known]


def freeze():
    out=ROOT/'aug-deep-frontier-live-v1';out.mkdir(exist_ok=True);source=ROOT/'aug-deep-frontier-v1'
    result=json.loads((source/'results.json').read_text());frozen=json.loads((source/'plan.json').read_text())
    assert all(result['fit_cv_selected_by_cap'][str(c)]==dict(macro_ce=f'state{c}',expert_ce=f'state{c}') for c in (512,1000,2000))
    budgets=frozen['budgets'];calibration={str(b):json.loads((source/f'calibration-0-{b}.json').read_text())['parameters'][0] for b in budgets}
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text())
    plan=dict(methods={name:result['results'][name]['parameters'][0] for name in NAMES},budget_policies=calibration,budgets=budgets,
        sample_sha256=digest(ROOT/'aug-deep-scale-v1/sample.json'),dev_sha256=digest(source/'results.json'),sidecars=manifest['files_sha256'],
        sources={f:digest(Path(__file__).with_name(f)) for f in ('deep_frontier_live.py','growforest.cpp','threadforest.cpp','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','scaled_count.cpp','diff_backup.cpp','handles.py','budget_policy.py')},
        roots_per_block=256,threads=4,initial_simulations=128,
        execution='Run every prefix to128, choose its budget from raw root/128 features, and grow the same tree and KV without repeated queries. Per-root scale16 through1000 and16*B/1000 above1000. Forced moves0. Fail on capacity overflow; never silently lower a chosen budget. Atomic completed blocks resume after preemption.',
        selection='Three cap-specific CV winners plus fixed1000. Frozen before new live outputs. This is the same reused August development cohort, not a fresh confirmation. No coefficient changes or golden evaluation.',
        comparison='Report actual counts and timing, per-position cached/live policy differences and routing changes. Include same-batch fixed control. Aggregate batching drift is not equivalent to exact per-position invariance.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return out,plan


def run(oracle,spec):
    if not spec.get('smoke'):
        receipt=ROOT/'engine-queue/099-deep-frontier-live-smoke.result.json'
        assert receipt.exists() and json.loads(receipt.read_text()).get('smoke')=='passed','Require mixed-depth smoke'
    out,plan=freeze();begin_all=time.monotonic();d=read('aug-deep-scale-v1')
    rows,cells,games,fm,cv,ids,mask,target=(d[k] for k in ('rows','cells','games','fit','cv','ids','mask','target'))
    n=len(rows);ar=np.arange(n);budgets=np.array(plan['budgets']);source=ROOT/'aug-deep-frontier-v1'
    inv=ROOT/'aug-tune-v1'
    for name in ('strat.npz','feats.npz'):assert digest(inv/name)==plan['sidecars'][name]
    with np.load(inv/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(inv/'feats.npz') as f:side=f['feats']
    seconds=[]
    for r in rows:
        rr,cc=r['row'],r['column'];assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(r['prefix']):cc],r['prefix']);seconds.append(side[rr,cc-1,0])
    seconds=np.array(seconds);module,reducer=load(),backup_load()
    for name,config in plan['methods'].items():
        folder=out/name;folder.mkdir(exist_ok=True)
        for lo in range(0,n,plan['roots_per_block']):
            if spec.get('smoke') and lo>0:break
            path=folder/f'{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            hi=min(n,lo+plan['roots_per_block']);part=rows[lo:hi];size=len(part);ii=ids[lo:hi].astype(np.int32);mm=mask[lo:hi];sec=seconds[lo:hi]
            forced=mm.sum(1)==1;initial=np.where(forced,0,128)
            tick=time.monotonic();oracle.reset();bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits.astype(float)
            prefix_tokens=oracle.new_tokens;prefill_seconds=time.monotonic()-tick
            tree=module.Tree([r['prefix'] for r in part],z,initial.tolist(),[2.5]*size,plan['threads'])
            def advance():
                while not tree.done:
                    h=tree.select()
                    if len(h):tree.update(bridge(h))
            advance();early=tree.compact();q128=reducer.Backup(early,128,16.).reduce(np.log(.2),-.5,ii)[0];del early
            if name=='fixed1000':index=np.full(size,3)
            else:index=route(cells[lo:hi],features(part,z,q128,ii,mm,sec),config,budgets)
            chosen=budgets[index].copy();chosen[forced]=0
            assert sum(chosen)+sum(len(r['prefix']) for r in part)<oracle.runner.max_total_num_tokens,'Mixed block exceeds KV capacity'
            assert sum(chosen)+size<oracle.capacity,'Mixed block exceeds node capacity'
            tree.grow(chosen.tolist());advance();compact=tree.compact()
            q=np.zeros(mm.shape)
            for scale in np.unique(16*np.maximum(1,budgets[index]/1000)):
                take=16*np.maximum(1,budgets[index]/1000)==scale
                q[take]=reducer.Backup(compact,16000,float(scale)).reduce(np.log(.2),-.5,ii)[0][take]
            nodes=np.array(tree.evals);assert int(nodes.sum())==bridge.queries;stat=tree.stats();del tree,compact
            p=np.zeros(mm.shape)
            for bi,b in enumerate(budgets):
                take=index==bi
                if take.any():p[take]=policy([part[i] for i in np.flatnonzero(take)],z[take],q[take],ii[take],mm[take],sec[take],plan['budget_policies'][str(b)])
            stat.update(seconds=time.monotonic()-tick,prefill_tokens=prefix_tokens,prefill_seconds=prefill_seconds,forward_seconds=oracle.forward_seconds,job=os.environ.get('SLURM_JOB_ID'),unique_nonroot_nn_requests=bridge.queries)
            temp=path.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,policy=p,root=z,q=q,q128=q128,nodes=nodes,budgets=chosen,index=index,stats=json.dumps(stat),game=games[lo:hi],ply=[r['ply'] for r in part])
            temp.replace(path);print('frontier-live',name,hi,n,stat['seconds'],nodes.mean(),flush=True)
    if spec.get('smoke'):return dict(smoke='passed',methods=list(NAMES),positions_per_method=plan['roots_per_block'],seconds=time.monotonic()-begin_all)
    records,losses={},{}
    with np.load(source/'scores.npz') as f:
        np.testing.assert_array_equal(f['games'],games);cached_choices=dict(zip(f['names'],f['choices']));cached_losses=dict(zip(f['names'],f['loss']))
    cached_q=np.zeros((6,n,mask.shape[1]));cached_root=np.zeros((n,2432))
    for path in sorted(source.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);cached_q[:,lo:hi]=f['q'];cached_root[lo:hi]=f['root']
    for name in NAMES:
        p=np.zeros(mask.shape);nodes=np.zeros(n);chosen=np.zeros(n,int);stats=[]
        for path in sorted((out/name).glob('[0-9]*.npz')):
            with np.load(path) as f:
                lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi]);p[lo:hi]=f['policy'];nodes[lo:hi]=f['nodes'];chosen[lo:hi]=f['index'];stats.append(json.loads(str(f['stats'])))
        assert (p[mask]>0).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-13)
        cached=np.zeros(mask.shape)
        for bi,b in enumerate(budgets):
            take=cached_choices[name]==bi
            if take.any():cached[take]=policy([rows[i] for i in np.flatnonzero(take)],cached_root[take],cached_q[bi,take],ids[take],mask[take],seconds[take],plan['budget_policies'][str(b)])
        ll=-np.log(p[ar,target]);cached_reconstruction_error=float(np.max(np.abs(-np.log(cached[ar,target])-cached_losses[name])))
        assert cached_reconstruction_error<1e-9,'Cached policy calibration reconstruction mismatch'
        ce,nc=means(ll[~fm],cells[~fm]),means(nodes[~fm],cells[~fm]);gap=means((ll-cached_losses[name])[~fm],cells[~fm])
        records[name]=dict(training_equivalent_cm=None,confirmation=dict(macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),cells=ce.tolist()),mean_nodes=float(nc.mean()),expert_mean_nodes=float(nc[3::4].mean()),seconds=sum(s['seconds'] for s in stats),prefill_tokens=sum(s['prefill_tokens'] for s in stats),
            cached_to_live=dict(macro_ce=float(gap.mean()),expert_ce=float(gap[3::4].mean()),max_policy_difference=float(np.max(np.abs(p-cached))),mean_kl=float(np.mean(np.sum(cached*(np.log(np.maximum(cached,1e-300))-np.log(np.maximum(p,1e-300))),axis=1))),route_changed=float(np.mean(chosen!=cached_choices[name])),calibration_reconstruction_max_ce_error=cached_reconstruction_error))
        losses[name]=ll
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(51092).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['fixed1000'];pt=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_parent']=dict(macro=float(pt.mean()),expert=float(pt[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    result=dict(results=records,seconds=time.monotonic()-begin_all,plan_sha256=digest(out/'plan.json'));atomic(out/'results.json',result)
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('LIVE', {name:rec['confirmation'] for name,rec in records.items()},flush=True)
    return dict(study=out.name,seconds=result['seconds'])
