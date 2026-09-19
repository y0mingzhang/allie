"""Live golden evaluation of the two expanded-August CV-selected stacks."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .threadforest_native import load
from .diff_backup_native import load as backup_load
from .handles import HandleOracle
from .analyze_expanded import components
from .golden_metrics import summarize, controls

OUT = ROOT / 'golden-expanded-stack-v2'


def freeze():
    OUT.mkdir(exist_ok=True)
    dev = ROOT / 'aug-expanded-stack-v1/results.json'
    report = json.loads(dev.read_text())
    assert report['fit_cv_selected'] == dict(macro_ce='subtree_sigma10', expert_ce='joint_sigma05')
    names = ('constant_single', 'subtree_sigma10', 'joint_sigma05')
    files = [Path(__file__), *[Path(__file__).with_name(s) for s in (
        'diff_backup.cpp', 'diff_backup_native.py', 'threadforest.cpp', 'threadforest_native.py',
        'handleforest.cpp', 'coverage.cpp', 'compact.cpp', 'backups.cpp', 'mcts_native.hpp',
        'board.cpp', 'handles.py', 'direct.py', 'analyze_expanded.py', 'analyze_player_search.py', 'golden_metrics.py')]]
    plan = dict(budget=1000, roots_per_block=1024, threads=4, skip_forced=True,
        methods={name: {k: report['results'][name][k] for k in ('backup', 'parameters', 'weights')} for name in names},
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'),
        laws_sha256=digest(ROOT/'training-cm-laws.json'), dev_sha256=digest(dev),
        sources={p.name: digest(p) for p in files},
        old_mixture_scores_sha256=digest(ROOT/'golden-player-search-v1/scores.npz'),
        selection='The macro and expert winners of three-way GAME CV within expanded August fit games, plus their matched constant-temperature control. Parameters frozen before golden; report both selected arms, no golden selection.',
        semantics='Actual live 1000-simulation root-coverage trees with cp2.5 PUCT below root. All methods share exactly the same NN queries; output backup temperature does not affect expansion. Complete policy vectors retained for a later order audit. Golden sample reused; final success requires fresh-game confirmation. CM is conditional on the frozen anchored training law, not inference acceleration.')
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic(path, plan)
    return plan


def run(oracle, spec):
    plan = freeze()
    tree_module, backup_module = load(), backup_load()
    rows = json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    start = time.monotonic()
    for lo in range(0, len(rows), plan['roots_per_block']):
        path = OUT/f'{lo:06d}.npz'
        if path.exists():
            continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():
            raise RuntimeError('STOP')
        part = rows[lo:lo+plan['roots_per_block']]
        n, ar = len(part), np.arange(len(part))
        groups = np.array([r['cell'] % 4 for r in part])
        budgets = [0 if len(r['legal']) == 1 else plan['budget'] for r in part]
        assert sum(budgets)+sum(len(r['prefix']) for r in part) < oracle.runner.max_total_num_tokens
        oracle.reset()
        begin = time.monotonic()
        bridge = HandleOracle(oracle, [r['prefix'] for r in part])
        z = bridge.root_logits
        tree = tree_module.Tree([r['prefix'] for r in part], z, budgets, [2.5]*n, plan['threads'])
        while not tree.done:
            h = tree.select()
            if len(h):
                tree.update(bridge(h))
        k = max(len(r['legal']) for r in part)
        ids = np.zeros((n, k), np.int32)
        mask = np.zeros((n, k), bool)
        target = np.zeros(n, int)
        for i, r in enumerate(part):
            ids[i, :len(r['legal'])] = np.array(r['legal'])-378
            mask[i, :len(r['legal'])] = True
            target[i] = r['legal'].index(r['target'])
        compact = tree.compact()
        backup = backup_module.Backup(compact, 1000)
        nodes, stats = np.array(tree.evals), tree.stats()
        assert int(nodes.sum()) == bridge.queries
        del tree
        logits = z[:, 378:2346][ar[:, None], ids].astype(float)
        payload = {}

        def metrics(p, moves, targets):
            predicted = np.where(p == p.max(1)[:, None], moves, 1968).min(1)
            return np.stack([-np.log(p[ar, targets]), predicted == moves[ar, targets], p.max(1)], axis=1)

        payload['port_raw'] = metrics(softmax(z[:, 378:2346].astype(float), axis=1),
            np.broadcast_to(np.arange(1968), (n, 1968)), np.array([r['target']-378 for r in part]))
        payload['new_legal'] = metrics(softmax(np.where(mask, logits, -np.inf), axis=1), ids, target)
        for name, config in plan['methods'].items():
            q = backup.reduce(np.log(config['backup']['tau0']), config['backup']['count_exponent'], ids)[0]
            if name == 'constant_single':
                reference_q = tree_module.reduce(compact, 1000, .1, .1)[ar[:, None], ids]
                stats['constant_backup_max_abs_diff'] = float(np.max(np.abs(q-reference_q)))
                np.testing.assert_allclose(q, reference_q, atol=1e-10, rtol=0)
            p = np.einsum('a,ank->nk', np.array(config['weights']), components(logits, q, mask, config['parameters'], groups))
            assert (p[mask] > 0).all()
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-14)
            payload[name] = metrics(p, ids, target)
            payload[name+'_policy'] = p
            payload[name+'_q'] = q
        del backup, compact
        stats.update(seconds=time.monotonic()-begin, new_tokens=oracle.new_tokens,
            forward_seconds=oracle.forward_seconds, unique_nonroot_nn_requests=bridge.queries)
        tmp = path.with_suffix('.partial')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **payload, z=z, ids=ids, mask=mask, nodes=nodes, stats=json.dumps(stats),
                game=[r['game'] for r in part], ply=[r['ply'] for r in part])
        tmp.replace(path)
        print('Golden expanded stack', lo+n, '/', len(rows), stats['seconds'], flush=True)
    return analyze(time.monotonic()-start)


def analyze(elapsed=None):
    begin = time.monotonic()
    plan = freeze()
    rows = json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    names = ['port_raw', 'new_legal', *plan['methods']]
    scores, costs = controls(rows)
    stats, node_parts = [], []
    for name in names:
        scores[name] = []
    for lo in range(0, len(rows), plan['roots_per_block']):
        with np.load(OUT/f'{lo:06d}.npz') as f:
            part = rows[lo:lo+len(f['game'])]
            np.testing.assert_array_equal(f['game'], [r['game'] for r in part])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in part])
            for name in names:
                scores[name].append(f[name])
            node_parts.append(f['nodes'])
            stats.append(json.loads(str(f['stats'])))
    nodes = np.concatenate(node_parts)
    for name in names:
        scores[name] = np.concatenate(scores[name])
        costs[name] = nodes if name in plan['methods'] else np.zeros(len(rows))
    old = ROOT/'golden-player-search-v1/scores.npz'
    assert digest(old) == plan['old_mixture_scores_sha256']
    with np.load(old) as f:
        np.testing.assert_array_equal(f['game'], [r['game'] for r in rows])
        np.testing.assert_array_equal(f['ply'], [r['ply'] for r in rows])
        for name in ('prior_s05', 'prior_s10'):
            i = list(f['names']).index(name)
            scores['old_'+name], costs['old_'+name] = f['scores'][i], f['nodes'][i]
    old_scores, old_nodes = [], []
    for path in sorted((ROOT/'golden-fast-coverage-v1').glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo = int(path.stem)
            np.testing.assert_array_equal(f['game'], [r['game'] for r in rows[lo:lo+len(f['game'])]])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in rows[lo:lo+len(f['ply'])]])
            old_scores.append(f['fast4000'])
            old_nodes.append(f['fast4000_nodes'])
    scores['old4000'], costs['old4000'] = np.concatenate(old_scores), np.concatenate(old_nodes)
    result = summarize(rows, scores, costs, references=('new_legal', 'constant_single', 'old_prior_s05', 'old_prior_s10', 'old4000'))
    report = dict(methods=result, positions=len(rows), plan_sha256=digest(OUT/'plan.json'),
        scoring_seconds=sum(s['seconds'] for s in stats),
        analysis_seconds=time.monotonic()-begin, elapsed_seconds=elapsed,
        new_tokens=sum(s['new_tokens'] for s in stats),
        caveat=plan['semantics']+' Paired CIs do not correct repeated benchmark use or include scaling-law uncertainty.')
    atomic(OUT/'results.json', report)
    with (OUT/'scores.npz').open('wb') as f:
        np.savez_compressed(f, names=names, scores=np.stack([scores[k] for k in names]), nodes=nodes,
            game=[r['game'] for r in rows], ply=[r['ply'] for r in rows])
    for name in plan['methods']:
        print(name, {k: result[name][k] for k in ('macro', 'expert_macro', 'macro_training_eq_cm', 'expert_macro_training_eq_cm', 'mean_nodes')}, flush=True)
    return report


if __name__ == '__main__':
    freeze()
