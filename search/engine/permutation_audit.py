"""Fixed-order numerical reproducibility audit of the frozen live Elo router.

No parameters are fit and no new labels guide selection. All positions are
retained and restored to canonical order before paired whole-game analysis.
"""
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from .balanced_eval import ROOT, GLOBAL_STOP, atomic, digest
from .backup_native import load
from .fit_policy import loss_gradient
from .golden_metrics import summarize, controls

OUT = ROOT / 'golden-permutation-v1'
PARENT = ROOT / 'golden-dynamic-router-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    parent = json.loads((PARENT / 'plan.json').read_text())
    n = len(json.loads((ROOT / 'golden-balanced-v1/sample.json').read_text())['positions'])
    order = np.random.default_rng(1938043).permutation(n).tolist()
    plan = dict(method=parent['method'], cpuct=1.25, roots_per_block=512,
        permutation_seed=1938043, position_order=order,
        parent_plan_sha256=digest(PARENT / 'plan.json'),
        parent_results_sha256=digest(PARENT / 'results.json'),
        sample_sha256=parent['sample_sha256'], laws_sha256=parent['laws_sha256'],
        sources={p.name: digest(p) for p in [Path(__file__), *[Path(__file__).with_name(s) for s in
            ('golden_metrics.py', 'direct.py', 'backups.cpp', 'mcts_native.hpp', 'board.cpp', 'fit_policy.py')]]},
        purpose='Numerical audit of already frozen actual mixed-budget router; same512 batch size, different fixed position permutation. No choice between orders by quality.',
        threshold_ce=0.001, stage='Reused golden; diagnostic repeat, not an independent scientific replication.')
    dest = OUT / 'plan.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == plan
    else:
        atomic(dest, plan)
    return plan


def run(oracle, spec):
    plan = freeze(); module = load(); start = time.monotonic()
    rows = json.loads((ROOT / 'golden-balanced-v1/sample.json').read_text())['positions']
    order = np.array(plan['position_order']); bs = plan['roots_per_block']
    for lo in range(0, len(rows), bs):
        dest = OUT / f'{lo:06d}.npz'
        if dest.exists():
            continue
        if GLOBAL_STOP.exists() or (ROOT / 'STOP').exists():
            raise RuntimeError('STOP')
        ix = order[lo:lo+bs]; part = [rows[i] for i in ix]; n = len(part); ar = np.arange(n)
        groups = np.array([r['cell'] % 4 for r in part])
        budgets = [plan['method']['budgets_by_elo'][g] for g in groups]
        oracle.reset(); begin = time.monotonic(); root = oracle([r['prefix'] for r in part])
        unique = 0
        tree = module.Tree([r['prefix'] for r in part], root, budgets, [plan['cpuct']]*n)
        for _ in range(max(budgets)):
            prefixes = tree.select()
            if prefixes:
                before = oracle.next_row; tree.update(oracle(prefixes)); unique += oracle.next_row-before
        k = max(len(r['legal']) for r in part)
        ids = np.zeros((n,k), int); mask = np.zeros((n,k), bool); target = np.zeros(n,int); q = np.zeros((n,k))
        for i, ((moves, values), r) in enumerate(zip(tree.backups([.1])[0], part)):
            ids[i,:len(r['legal'])] = np.array(r['legal'])-378; mask[i,:len(r['legal'])] = True
            target[i] = r['legal'].index(r['target']); lookup = dict(zip(moves, values))
            q[i,mask[i]] = [lookup[a] for a in ids[i,mask[i]]]
        nodes = np.array(tree.evals); stats = tree.stats(); del tree
        assert nodes.sum() == stats['evaluated_leaves']
        z = root[:,378:2346][ar[:,None],ids].astype(float); p = np.zeros_like(q)
        for g in np.unique(groups):
            selected = groups == g; f = plan['method']['parameters'][str(int(g))]
            p[selected], _ = loss_gradient([f['alpha'], f['beta']], z[selected], q[selected], mask[selected], target[selected], 'forward', return_policy=True)
        prediction = np.where(mask & (p == p.max(1)[:,None]), ids, 1968).min(1)
        score = np.stack([-np.log(p[ar,target]), prediction == ids[ar,target], p.max(1)], axis=1)
        stats.update(seconds=time.monotonic()-begin, new_tokens=oracle.new_tokens,
            forward_seconds=oracle.forward_seconds, unique_nonroot_nn_requests=unique)
        tmp = dest.with_suffix('.partial')
        with tmp.open('wb') as f:
            np.savez_compressed(f, index=ix, score=score, policy=p, ids=ids, mask=mask, root=root,
                nodes=nodes, stats=json.dumps(stats), game=np.array([r['game'] for r in part]), ply=np.array([r['ply'] for r in part]))
        tmp.replace(dest)
        print('Permutation audit', lo+n, '/', len(rows), stats['seconds'], flush=True)
    analyze()
    return dict(positions=len(rows), elapsed_seconds=time.monotonic()-start)


def read_cache(folder, rows, permuted):
    n = len(rows); scores = np.zeros((n,3)); nodes = np.zeros(n); root = np.zeros((n,2432)); policies = [None]*n; stats = []; seen = []
    for lo in range(0,n,512):
        with np.load(folder / f'{lo:06d}.npz') as z:
            ix = z['index'] if permuted else np.arange(lo,lo+len(z['game']))
            assert list(z['game']) == [rows[i]['game'] for i in ix]
            np.testing.assert_array_equal(z['ply'], [rows[i]['ply'] for i in ix])
            scores[ix] = z['score']; nodes[ix] = z['nodes']; root[ix] = z['root']; seen.extend(ix)
            stats.append(json.loads(str(z['stats'])))
            for j,i in enumerate(ix):
                np.testing.assert_array_equal(z['ids'][j,z['mask'][j]],np.array(rows[i]['legal'])-378)
                policies[i] = z['policy'][j,z['mask'][j]]
    assert sorted(seen) == list(range(n))
    return scores,nodes,root,policies,stats


def analyze():
    start = time.monotonic(); plan = json.loads((OUT / 'plan.json').read_text())
    assert plan['sample_sha256'] == digest(ROOT / 'golden-balanced-v1/sample.json')
    assert plan['laws_sha256'] == digest(ROOT / 'training-cm-laws.json')
    rows = json.loads((ROOT / 'golden-balanced-v1/sample.json').read_text())['positions']
    old = read_cache(PARENT,rows,False); new = read_cache(OUT,rows,True)
    scores,costs = controls(rows)
    scores.update(original_order=old[0],permuted_order=new[0]); costs.update(original_order=old[1],permuted_order=new[1])
    methods = summarize(rows,scores,costs,references=('legal','four_ply','soft1000','original_order'))
    prior = json.loads((PARENT / 'results.json').read_text())['methods']['dynamic']
    for key in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm'):
        np.testing.assert_allclose(methods['original_order'][key],prior[key],atol=1e-12,rtol=0)
    delta = {metric: methods['permuted_order'][metric]-methods['original_order'][metric] for metric in ('macro','expert_macro')}
    diffs = np.array([[np.max(np.abs(p-q)),np.sum(np.abs(p-q)),np.sum(p*np.log(p/q))] for p,q in zip(old[3],new[3])])
    rp = softmax(old[2][:,378:2346],axis=1); rq = softmax(new[2][:,378:2346],axis=1)
    root_kl = np.sum(rp*np.log(rp/rq),axis=1)
    report = dict(methods=methods, positions=len(rows), order_delta_ce=delta,
        exceeds_preregistered_threshold=any(abs(x)>plan['threshold_ce'] for x in delta.values()),
        policy_diff=dict(max_abs=float(diffs[:,0].max()),max_l1=float(diffs[:,1].max()),mean_kl=float(diffs[:,2].mean()),max_kl=float(diffs[:,2].max())),
        raw_root_diff=dict(max_logit=float(np.abs(old[2]-new[2]).max()),mean_kl=float(root_kl.mean()),max_kl=float(root_kl.max())),
        scoring_seconds=sum(s['seconds'] for s in new[4]),analysis_seconds=time.monotonic()-start,
        plan_sha256=digest(OUT/'plan.json'), caveat='One fixed permutation diagnoses numerical sensitivity; it is not a variance estimate or a fresh-data replication. Do not select the better order. Training CM is law-shape conditional.')
    atomic(OUT/'results.json',report)
    print('Permutation CE difference',delta,'policy diff',report['policy_diff'],flush=True)


if __name__ == '__main__':
    import sys
    freeze() if '--freeze' in sys.argv else analyze()
