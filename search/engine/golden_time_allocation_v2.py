"""Frozen predicted-time versus shuffled-time versus fixed modern search."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .growforest_native import load
from .diff_backup_native import load as backup_load
from .handles import HandleOracle
from .budget_policy import policy,route
from .value_of_compute import features
from .golden_metrics import summarize

OUT=ROOT/'golden-time-allocation-v2'


def freeze():
    OUT.mkdir(exist_ok=True)
    dev=ROOT/'aug-time-allocation-v2/results.json';surface=ROOT/'aug-budget-surface-v2/results.json'
    d=json.loads(dev.read_text());s=json.loads(surface.read_text())
    assert d['fit_cv_selected']==dict(macro_ce='time_cell_p1',expert_ce='time_cell_p1')
    pars=d['results']['time_cell_p1']['parameters'][0]
    inventory=Path('/data/group_data/dei-group/yimingz3/allie/strat-eval-v1')
    manifest=json.loads((inventory/'manifest.json').read_text())
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ('growforest.cpp','growforest_native.py','threadforest.cpp','handleforest.cpp','coverage.cpp',
         'compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','diff_backup.cpp','diff_backup_native.py',
         'handles.py','budget_policy.py','time_allocation_v2.py','analyze_expanded.py','analyze_player_search.py',
         'temperature_stack.py','golden_metrics.py','direct.py')]]
    plan=dict(methods={name:pars for name in ['fixed512','predicted_time','shuffled_time']},
        budget_policies={b:v[0] for b,v in s['fold_parameters'].items()},budgets=[128,256,512,1000],
        dev_sha256=digest(dev),surface_sha256=digest(surface),sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),
        laws_sha256=digest(ROOT/'training-cm-laws.json'),sidecars=dict(strat=manifest['sha256'],feats=manifest['feats_sha256']),
        old1000_scores_sha256=digest(ROOT/'golden-temperature-stack-v1/scores.npz'),roots_per_block=1024,threads=4,
        initial_simulations=128,skip_forced=True,sources={p.name:digest(p) for p in files},
        selection='August fit-gameCV selects time_cell_p1 for both metrics. All3 arms frozen before golden; no selection among them on golden.',
        execution='Common root-prediction prepass enables an identical within-cell shuffled signal histogram. Then same128-stage plus continuation for each arm; root prepass charged separately to every method. Actual NN nodes and realized nominal budgets reported, not assumed equal.',
        semantics='Only predicted thinking time divided by its August-fit mean in the query Elo/format cell chooses budget; scale and quantization frozen. Same modern root-coverage/soft-Bellman tree and calibrated output for every arm, not faithful released Allie traversal. Reused golden exploratory; conditional training-equivalent CM; fresh final confirmation still required.')
    pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return plan


def clocks(rows,plan):
    inventory=Path('/data/group_data/dei-group/yimingz3/allie/strat-eval-v1')
    assert digest(inventory/'strat.npz')==plan['sidecars']['strat']
    assert digest(inventory/'feats.npz')==plan['sidecars']['feats']
    with np.load(inventory/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(inventory/'feats.npz') as f:feat=f['feats']
    seconds=[]
    for r in rows:
        rr,cc=r['row'],r['column']
        assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(r['prefix']):cc],r['prefix'])
        seconds.append(feat[rr,cc-1,0])
    return np.array(seconds)


def metrics(p,ids,target):
    ar=np.arange(len(p));chosen=np.where(p==p.max(1)[:,None],ids,1968).min(1)
    return np.stack([-np.log(p[ar,target]),chosen==ids[ar,target],p.max(1)],axis=1)


def run(oracle,spec):
    plan=freeze();start=time.monotonic();module,backup=load(),backup_load()
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    seconds=clocks(rows,plan)
    prepass=OUT/'root-time-prepass.npz'
    if not prepass.exists():
        allz=[];ptokens=0;begin=time.monotonic()
        for lo in range(0,len(rows),plan['roots_per_block']):
            oracle.reset();allz.append(oracle([r['prefix'] for r in rows[lo:lo+plan['roots_per_block']]]));ptokens+=oracle.new_tokens
        z=np.concatenate(allz);tc=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
        pt=softmax(z[:,2350:2413].astype(float),axis=1)@tc
        tmp=prepass.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,predicted_seconds=pt,game=[r['game'] for r in rows],ply=[r['ply'] for r in rows],tokens=ptokens,seconds=time.monotonic()-begin)
        tmp.replace(prepass)
    with np.load(prepass) as f:
        np.testing.assert_array_equal(f['game'],[r['game'] for r in rows]);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows])
        predicted=f['predicted_seconds']
    from .time_allocation_v2 import permutation,choose
    cells_all=np.array([r['cell'] for r in rows]);perm=permutation(cells_all,np.zeros(len(rows),int))
    for name,config in plan['methods'].items():
        folder=OUT/name;folder.mkdir(exist_ok=True)
        for lo in range(0,len(rows),plan['roots_per_block']):
            path=folder/f'{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            part=rows[lo:lo+plan['roots_per_block']];n=len(part);ar=np.arange(n);cells=np.array([r['cell'] for r in part])
            forced=np.array([len(r['legal'])==1 for r in part]);sec=seconds[lo:lo+n]
            initial=np.where(forced,0,128).tolist()
            assert 1000*n+sum(len(r['prefix']) for r in part)<oracle.runner.max_total_num_tokens
            begin=time.monotonic();oracle.reset()
            bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits.astype(float)
            tree=module.Tree([r['prefix'] for r in part],z,initial,[2.5]*n,plan['threads'])
            def advance():
                while not tree.done:
                    h=tree.select()
                    if len(h):tree.update(bridge(h))
            advance()
            k=max(len(r['legal']) for r in part);ids=np.zeros((n,k),np.int32);mask=np.zeros((n,k),bool)
            for i,r in enumerate(part):
                ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
            target=np.array([r['legal'].index(r['target']) for r in part])
            if name=='fixed512':index=np.full(n,2)
            else:
                sig=predicted[lo:lo+n] if name=='predicted_time' else predicted[perm[lo:lo+n]]
                index=choose(sig/np.asarray(config['divisor_by_cell'])[cells],config['scale'])
            budgets=np.asarray(plan['budgets'])[index];budgets[forced]=0
            tree.grow(budgets.tolist());advance()
            compact=tree.compact()
            q=backup.Backup(compact,1000).reduce(np.log(.2),-.5,ids)[0]
            nodes=np.array(tree.evals);stats=tree.stats()
            assert int(nodes.sum())==bridge.queries
            del tree,compact
            p=np.zeros(mask.shape)
            for bi,b in enumerate(plan['budgets']):
                take=index==bi
                if not take.any():continue
                p[take]=policy([part[i] for i in np.flatnonzero(take)],z[take],q[take],ids[take],mask[take],sec[take],plan['budget_policies'][str(b)])
            logits=z[:,378:2346][ar[:,None],ids]
            raw=metrics(softmax(z[:,378:2346],axis=1),np.broadcast_to(np.arange(1968),(n,1968)),np.array([r['target']-378 for r in part]))
            legal=metrics(softmax(np.where(mask,logits,-np.inf),axis=1),ids,target)
            stats.update(seconds=time.monotonic()-begin,forward_seconds=oracle.forward_seconds,new_tokens=oracle.new_tokens,
                unique_nonroot_nn_requests=bridge.queries,nominal_mean=float(budgets.mean()))
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez_compressed(f,scores=metrics(p,ids,target),raw=raw,legal=legal,policy=p,ids=ids,mask=mask,z=z,q=q,
                    nodes=nodes,budgets=budgets,stats=json.dumps(stats),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
            tmp.replace(path);print(name,lo+n,'/',len(rows),stats['seconds'],float(nodes.mean()),flush=True)
    return analyze(time.monotonic()-start)


def analyze(elapsed=None):
    begin=time.monotonic();plan=freeze()
    rows=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    scores,costs,times={}, {}, {}
    for name in plan['methods']:
        blocks=[];nodes=[];timings=[];raw=[];legal=[]
        for path in sorted((OUT/name).glob('[0-9]*.npz')):
            with np.load(path) as f:
                lo=int(path.stem);part=rows[lo:lo+len(f['game'])]
                np.testing.assert_array_equal(f['game'],[r['game'] for r in part])
                np.testing.assert_array_equal(f['ply'],[r['ply'] for r in part])
                blocks.append(f['scores']);nodes.append(f['nodes']);timings.append(json.loads(str(f['stats']))['seconds'])
                raw.append(f['raw']);legal.append(f['legal'])
        assert sum(len(x) for x in blocks)==len(rows)
        scores[name]=np.concatenate(blocks);costs[name]=np.concatenate(nodes);times[name]=sum(timings)
        if name=='fixed512':
            scores['port_raw']=np.concatenate(raw);scores['new_legal']=np.concatenate(legal)
            costs['port_raw']=costs['new_legal']=np.zeros(len(rows))
    with np.load(ROOT/'golden-temperature-stack-v1/scores.npz') as f:
        np.testing.assert_array_equal(f['game'],[r['game'] for r in rows])
        scores['old1000']=f['scores'][list(f['names']).index('temperature')];costs['old1000']=f['nodes']
    result=summarize(rows,scores,costs,references=('new_legal','fixed512','shuffled_time','old1000'))
    with np.load(OUT/'root-time-prepass.npz') as pre:
        prepass_cost=dict(extra_full_prefix_queries_per_position=1,seconds=float(pre['seconds']),tokens=int(pre['tokens']),charged_to_each_method=True)
    atomic(OUT/'results.json',dict(root_prepass=prepass_cost,methods=result,positions=len(rows),plan_sha256=digest(OUT/'plan.json'),
        scoring_seconds=elapsed,method_seconds=times,analysis_seconds=time.monotonic()-begin,caveat=plan['semantics']))
    np.savez_compressed(OUT/'scores.npz',names=list(scores),scores=np.stack(list(scores.values())),nodes=np.stack(list(costs.values())),
        game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    for name in plan['methods']:
        print(name,{k:result[name][k] for k in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm','mean_nodes')},flush=True)
    return dict(output=str(OUT),scoring_seconds=elapsed,method_seconds=times)


if __name__=='__main__':
    import sys
    if '--freeze' in sys.argv:freeze()
    else:analyze()
