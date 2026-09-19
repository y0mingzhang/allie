"""Entire-tree counterfactual strength: actual root policy, altered continuation model."""
import json, time, os
from pathlib import Path
import numpy as np
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .sample_data import read
from .growforest_native import load
from .scaled_count_native import load as backup_load
from .handles import HandleOracle
from .stack_fit import fit_stack
from .analyze_august import means

NAMES = ('actual', 'plus200', 'plus400', 'equal2400')
OUT = ROOT/'aug-conditioned-search-v1'


def transform(prefix,name):
    assert name in NAMES
    p=list(prefix); assert len(p)>=11 and all(0<=t<=9 for t in p[3:11])
    if name=='actual': return p
    ratings=[int(''.join(map(str,p[start:start+4]))) for start in (3,7)]
    # Use the same nonnegative shift for BOTH players, so their gap is preserved.
    delta=0 if name=='equal2400' else max(0,min(int(name[4:]),3200-max(ratings)))
    for start,old in zip((3,7),ratings):
        p[start:start+4]=list(map(int,f'{2400 if name=="equal2400" else old+delta:04d}'))
    assert p[:3]==prefix[:3] and p[11:]==prefix[11:]
    return p


def freeze():
    OUT.mkdir(exist_ok=True)
    plan = dict(methods=NAMES, simulations=1000, initial_simulations=128, roots_per_block=256, threads=4,
        sample_sha256=digest(ROOT/'aug-deep-scale-v1/sample.json'),
        sources={name:digest(Path(__file__).with_name(name)) for name in
                 ('conditioned_search.py','growforest.cpp','threadforest.cpp','handleforest.cpp','coverage.cpp','compact.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py','scaled_count.cpp','diff_backup.cpp','stack_fit.py','budget_policy.py')},
        hypothesis='Earlier counterfactual strength root/one-ply screens were null. Changing the whole continuation policy can change which replies search discovers, which a one-ply critic test cannot measure. This separately tests that mechanism before dismissing the family.',
        semantics='Actual human root move logits set identical coverage quotas and output prior. The hypothetical header changes all continuation priors and critics, including the root fallback critic. Same1000 simulations and count-normalized soft backup. Actual root time predictions/header supply calibration features. No other-game information; no target/future move enters search.',
        transforms='plus200/plus400 add the SAME nonnegative shift to both ratings, reducing the common shift if needed to keep max rating<=3200 (zero if already above). The original rating gap is preserved. equal2400 sets both to2400. Original time-control tokens and complete move history unchanged. Header intervention is an internal reasoning query, not a claim about actual player strength.',
        costs='actual pays1 full-prefix query; interventions pay2 full-prefix queries per position, plus all actual nonroot NN evaluations. Cost is not assumed equal merely from1000 simulations. Store prefill and wall time separately. Forced moves use0 search nodes.',
        selection='Same4096 reused August cohort. Refit identical output-calibration pipeline inside game folds, select on fit-CV only, report every arm on disjoint checking games. Potentially training-seen. No golden conversion; no automatic promotion while numerical audit is unresolved.')
    plan=json.loads(json.dumps(plan));pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return plan


def run(oracle,spec):
    plan=freeze(); d=read('aug-deep-scale-v1');rows=d['rows'];n=len(rows);begin=time.monotonic()
    if not spec.get('smoke'):
        p=ROOT/'engine-queue/102-conditioned-search-smoke.result.json'
        assert p.exists() and json.loads(p.read_text()).get('smoke')=='passed'
    native,reducer=load(),backup_load()
    for lo in range(0,n,plan['roots_per_block']):
        if spec.get('smoke') and lo>0:break
        hi=min(n,lo+plan['roots_per_block']);part=rows[lo:hi];size=len(part)
        ids=d['ids'][lo:hi].astype(np.int32);mask=d['mask'][lo:hi];forced=mask.sum(1)==1
        for name in NAMES:
            folder=OUT/name;folder.mkdir(exist_ok=True);path=folder/f'{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            tick=time.monotonic();oracle.reset();actual_extra=0;extra_forward=0.
            if name!='actual':
                actual_root=oracle([r['prefix'] for r in part]).astype(float)
                actual_extra=oracle.new_tokens;extra_forward=oracle.forward_seconds;oracle.reset()
            prefixes=[transform(r['prefix'],name) for r in part]
            for r,p in zip(part,prefixes):assert r['prefix'][:3]==p[:3] and r['prefix'][11:]==p[11:]
            bridge=HandleOracle(oracle,prefixes);context_root=bridge.root_logits.astype(float)
            if name=='actual':actual_root=context_root.copy()
            prefill_tokens=actual_extra+oracle.new_tokens;prefill_seconds=time.monotonic()-tick
            seed=context_root.copy();seed[:,378:2346]=actual_root[:,378:2346]
            tree=native.Tree(prefixes,seed,np.where(forced,0,128).tolist(),[2.5]*size,plan['threads'])
            def advance():
                while not tree.done:
                    h=tree.select()
                    if len(h):tree.update(bridge(h))
            advance();tree.grow(np.where(forced,0,1000).tolist());advance();compact=tree.compact()
            q=reducer.Backup(compact,1000,16.).reduce(np.log(.2),-.5,ids)[0]
            nodes=np.array(tree.evals);assert int(nodes.sum())==bridge.queries
            stats=tree.stats();del tree,compact
            if spec.get('smoke') and name=='actual':
                # New-allocation reference: avoid comparing A100 math to old Ada math.
                reference=ROOT/'aug-deep-order-audit-v1/fixed1000-repeat'/path.name
                assert reference.exists(),'Require the new-allocation control before this smoke'
                with np.load(reference) as f:
                    np.testing.assert_array_equal(actual_root,f['root'])
                    np.testing.assert_array_equal(q,f['q'])
                    np.testing.assert_array_equal(nodes,f['nodes'])
            stats.update(seconds=time.monotonic()-tick,forward_seconds=extra_forward+oracle.forward_seconds,
                prefill_tokens=prefill_tokens,prefill_seconds=prefill_seconds,root_queries=size*(1 if name=='actual' else 2),job=os.environ.get('SLURM_JOB_ID'))
            temp=path.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,q=q,root=actual_root,context_root=context_root,nodes=nodes,stats=json.dumps(stats),game=d['games'][lo:hi],ply=[r['ply'] for r in part])
            temp.replace(path);print('conditioned-search',name,hi,n,stats['seconds'],flush=True)
    if spec.get('smoke'):return dict(smoke='passed',methods=NAMES,positions_per_method=plan['roots_per_block'],seconds=time.monotonic()-begin)
    return analyze(time.monotonic()-begin)


def analyze(elapsed=None):
    plan=freeze();d=read('aug-deep-scale-v1');rows=d['rows'];n=len(rows);cells=d['cells'];fm=d['fit'];games=d['games']
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text());assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:side=f['feats']
    seconds=np.array([side[r['row'],r['column']-1,0] for r in rows]);records={};losses={}
    for name in NAMES:
        q=np.zeros(d['mask'].shape);root=np.zeros((n,2432));nodes=np.zeros(n);stats=[];seen=np.zeros(n,bool)
        for path in sorted((OUT/name).glob('[0-9]*.npz')):
            with np.load(path) as f:
                lo=int(path.stem);hi=lo+len(f['game']);assert not seen[lo:hi].any();seen[lo:hi]=True
                np.testing.assert_array_equal(f['game'],games[lo:hi]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows[lo:hi]])
                q[lo:hi]=f['q'];root[lo:hi]=f['root'];nodes[lo:hi]=f['nodes'];stats.append(json.loads(str(f['stats'])))
        assert seen.all()
        rec,ll,_=fit_stack(rows,root,q,d['ids'],d['mask'],d['target'],cells,fm,d['cv'],seconds)
        costs=means(nodes[~fm],cells[~fm]);rec.update(mean_nodes=float(costs.mean()),expert_mean_nodes=float(costs[3::4].mean()),seconds=sum(s['seconds'] for s in stats),prefill_tokens=sum(s['prefill_tokens'] for s in stats),root_queries=sum(s['root_queries'] for s in stats));records[name]=rec;losses[name]=ll
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(539226).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses['actual'];pt=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_parent']=dict(macro=float(pt.mean()),expert=float(pt[3::4].mean()),macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    atomic(OUT/'results.json',dict(results=records,fit_cv_selected=selected,plan_sha256=digest(OUT/'plan.json'),elapsed_seconds=elapsed))
    np.savez_compressed(OUT/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print('CONDITIONED',json.dumps({k:v['confirmation'] for k,v in records.items()}),selected,flush=True)
    return dict(study=OUT.name,fit_cv_selected=selected,seconds=elapsed)


if __name__=='__main__':freeze()
