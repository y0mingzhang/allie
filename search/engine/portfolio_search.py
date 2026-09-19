"""Independent search portfolios; every component query and prefill is charged."""
import json
import os
import time
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


OUT = ROOT / 'aug-portfolio-search-v1'
COMPONENTS = {
    'single1000': (1000, 2.5),
    'half500': (500, 2.5),
    'half500_permuted': (500, 2.5),
    'narrow500': (500, .5),
    'wide500': (500, 5.),
    'mild_narrow500': (500, 1.),
    'mild_wide500': (500, 4.),
}
PAIRS = {
    'pair_same_c': ('half500', 'half500_permuted'),
    'pair_extreme': ('narrow500', 'wide500'),
    'pair_mild': ('mild_narrow500', 'mild_wide500'),
}


def freeze():
    OUT.mkdir(exist_ok=True)
    plan = dict(
        components=COMPONENTS, pairs=PAIRS, roots_per_block=256, threads=4,
        sample_sha256=digest(ROOT / 'aug-deep-scale-v1/sample.json'),
        export_sha256=digest(ROOT / 'serving-export/provenance.json'),
        sources={name: digest(Path(__file__).with_name(name)) for name in (
            'portfolio_search.py', 'growforest.cpp', 'threadforest.cpp',
            'handleforest.cpp', 'coverage.cpp', 'compact.cpp', 'backups.cpp',
            'mcts_native.hpp', 'board.cpp', 'handles.py', 'direct.py',
            'scaled_count.cpp', 'diff_backup.cpp', 'stack_fit.py')},
        hypothesis='Complementary internal exploration may reduce action-value error at fixed nominal budget. Test before investing in union-tree caching.',
        combination='Fixed arithmetic mean of the two soft-backed Q vectors, followed by the same independently game-CV-fitted output pipeline. Root action simulation quotas are identical for unpermuted components; assert visit-weighted mean equals arithmetic mean wherever visited.',
        backup='Both original count scale16 and budget-normalized scale8 at500 are computed. Pair arms report both (suffix _normalized for scale8). No parameter chosen on checking games. Single1000 uses scale16.',
        diversity_control='Same c2.5, with a fixed global root permutation from seed246851 independent of all scores. Restore canonical sample order before averaging. This isolates batch-order numerical diversity, not stochastic search. Root drift and quota changes are reported.',
        costs='Two independent500 searches pay all actual NN queries including duplicates, two root queries and two prefills. No cross-component or cross-position search reuse. Count terminal simulations separately. A pair is not assumed equal-cost merely because nominal simulations sum to1000.',
        evaluation='Reused August4096, potentially training-seen. Game-disjoint fit/check and3 gameCV folds. All arms reported; no golden or CM conversion. Fresh single1000 control must match the same-A100 conditioned actual control; fail rather than pool after hardware migration.',
    )
    plan = json.loads(json.dumps(plan))
    path = OUT / 'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic(path, plan)
    return plan


def run(oracle, spec):
    plan = freeze()
    d = read('aug-deep-scale-v1')
    rows, n = d['rows'], len(d['rows'])
    if not spec.get('smoke'):
        receipt = ROOT / 'engine-queue/107-portfolio-smoke.result.json'
        assert receipt.exists() and json.loads(receipt.read_text()).get('smoke') == 'passed'
    begin = time.monotonic()
    native, reducer = load(), backup_load()
    permutation = np.random.default_rng(246851).permutation(n)
    for lo in range(0, n, plan['roots_per_block']):
        if spec.get('smoke') and lo > 0:
            break
        hi = min(n, lo + plan['roots_per_block'])
        for name, (budget, cpuct) in COMPONENTS.items():
            folder = OUT / name
            folder.mkdir(exist_ok=True)
            dest = folder / f'{lo:06d}.npz'
            if dest.exists():
                continue
            if GLOBAL_STOP.exists() or (ROOT / 'STOP').exists():
                raise RuntimeError('STOP')
            index = permutation[lo:hi] if name.endswith('_permuted') else np.arange(lo, hi)
            part = [rows[i] for i in index]
            ids = d['ids'][index].astype(np.int32)
            forced = d['mask'][index].sum(1) == 1
            tick = time.monotonic()
            oracle.reset()
            bridge = HandleOracle(oracle, [r['prefix'] for r in part])
            root = bridge.root_logits.astype(float)
            prefill = oracle.new_tokens
            prefill_seconds = time.monotonic() - tick
            tree = native.Tree([r['prefix'] for r in part], root,
                               np.where(forced, 0, 128).tolist(),
                               [cpuct] * len(part), plan['threads'])

            def advance():
                while not tree.done:
                    handles = tree.select()
                    if len(handles):
                        tree.update(bridge(handles))

            advance()
            tree.grow(np.where(forced, 0, budget).tolist())
            advance()
            compact = tree.compact()
            q = reducer.Backup(compact, budget, 16.).reduce(np.log(.2), -.5, ids)[0]
            q_normalized = reducer.Backup(compact, budget, 16. * budget / 1000).reduce(np.log(.2), -.5, ids)[0]
            nodes = np.asarray(tree.evals)
            assert int(nodes.sum()) == bridge.queries
            visits = np.zeros(ids.shape, dtype=np.int32)
            for i, (moves, counts, _, _) in enumerate(tree.snapshot()):
                lookup = dict(zip(moves, counts))
                for j in np.flatnonzero(d['mask'][index[i]]):
                    visits[i, j] = lookup[int(ids[i, j])]
            np.testing.assert_array_equal(visits.sum(1), np.where(forced, 0, budget))
            stats = tree.stats()
            del tree, compact
            assert np.isfinite(q).all() and np.max(abs(q)) <= 1 + 1e-12
            assert np.isfinite(q_normalized).all() and np.max(abs(q_normalized)) <= 1 + 1e-12
            if name == 'single1000':
                with np.load(ROOT / 'aug-conditioned-search-v1/actual' / dest.name) as f:
                    np.testing.assert_array_equal(root, f['root'])
                    np.testing.assert_array_equal(q, f['q'])
                    np.testing.assert_array_equal(nodes, f['nodes'])
            stats.update(seconds=time.monotonic() - tick, prefill_tokens=prefill,
                         prefill_seconds=prefill_seconds, forward_seconds=oracle.forward_seconds,
                         root_queries=len(part), job=os.environ.get('SLURM_JOB_ID'))
            tmp = dest.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez_compressed(f, q=q, q_normalized=q_normalized, root=root, nodes=nodes,
                                    visits=visits, stats=json.dumps(stats), index=index,
                                    game=d['games'][index], ply=[r['ply'] for r in part])
            tmp.replace(dest)
            print('portfolio', name, hi, n, stats['seconds'], flush=True)
    if spec.get('smoke'):
        return dict(smoke='passed', positions_per_component=plan['roots_per_block'],
                    components=list(COMPONENTS), seconds=time.monotonic()-begin)
    return analyze(time.monotonic()-begin)


def analyze(elapsed=None):
    plan, d = freeze(), read('aug-deep-scale-v1')
    rows, cells, games, fm = (d[k] for k in ('rows', 'cells', 'games', 'fit'))
    n = len(rows)
    inv = ROOT / 'aug-tune-v1'
    manifest = json.loads((inv / 'manifest.json').read_text())
    assert digest(inv / 'feats.npz') == manifest['files_sha256']['feats.npz']
    with np.load(inv / 'feats.npz') as f:
        side = f['feats']
    seconds = np.array([side[r['row'], r['column']-1, 0] for r in rows])
    data = {}
    for name in COMPONENTS:
        item = dict(q=np.zeros(d['mask'].shape), q_normalized=np.zeros(d['mask'].shape),
                    root=np.zeros((n,2432)), visits=np.zeros(d['mask'].shape, np.int32),
                    nodes=np.zeros(n), stats=[])
        seen = np.zeros(n, bool)
        for path in sorted((OUT / name).glob('[0-9]*.npz')):
            with np.load(path) as f:
                idx = f['index']
                assert not seen[idx].any() and len(np.unique(idx)) == len(idx)
                seen[idx] = True
                np.testing.assert_array_equal(f['game'], games[idx])
                np.testing.assert_array_equal(f['ply'], [rows[i]['ply'] for i in idx])
                for key in ('q', 'q_normalized', 'root', 'nodes', 'visits'):
                    item[key][idx] = f[key]
                item['stats'].append(json.loads(str(f['stats'])))
        assert seen.all(), name
        data[name] = item
    root = data['single1000']['root']
    for name, item in data.items():
        if name != 'half500_permuted':
            np.testing.assert_array_equal(item['root'], root)
            if name != 'single1000':
                np.testing.assert_array_equal(item['visits'], data['half500']['visits'])
    records, losses = {}, {}

    def score(name, q, parts):
        rec, loss, _ = fit_stack(rows, root, q, d['ids'], d['mask'], d['target'],
                                 cells, fm, d['cv'], seconds)
        nodes = sum(data[k]['nodes'] for k in parts)
        cost = means(nodes[~fm], cells[~fm])
        stats = [s for k in parts for s in data[k]['stats']]
        rec.update(mean_nodes=float(cost.mean()), expert_mean_nodes=float(cost[3::4].mean()),
                   seconds=sum(s['seconds'] for s in stats),
                   prefill_tokens=sum(s['prefill_tokens'] for s in stats),
                   root_queries=sum(s['root_queries'] for s in stats), components=list(parts))
        records[name], losses[name] = rec, loss

    for name, item in data.items():
        score(name, item['q'], [name])
    score('half500_normalized', data['half500']['q_normalized'], ['half500'])
    for name, (a,b) in PAIRS.items():
        for key, suffix in [('q', ''), ('q_normalized', '_normalized')]:
            q = (data[a][key] + data[b][key]) / 2
            if name != 'pair_same_c':
                va, vb = data[a]['visits'], data[b]['visits']
                visited = (va + vb) > 0
                weighted = (data[a][key]*va + data[b][key]*vb)[visited] / (va+vb)[visited]
                np.testing.assert_allclose(q[visited], weighted, rtol=0, atol=3e-16)
            score(name+suffix, q, [a,b])
    _, ix = np.unique(games[~fm], return_inverse=True)
    ng = ix.max()+1
    ct = np.zeros((ng,16))
    np.add.at(ct, (ix,cells[~fm]), 1)
    draws = np.random.default_rng(720855).multinomial(ng, np.full(ng,1/ng), size=2000).astype(float)
    den = draws@ct
    assert (den>0).all()
    for name, rec in records.items():
        delta = losses[name] - losses['single1000']
        pt = means(delta[~fm],cells[~fm])
        sums = np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm])
        boot = draws@sums/den
        rec['delta_vs_parent'] = dict(macro=float(pt.mean()), expert=float(pt[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),
            expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(), cells=pt.tolist())
    selected = {m:min(records,key=lambda k:records[k]['fit_game_cv'][m])
                for m in ('macro_ce','expert_ce')}
    normal, perm = data['half500'],data['half500_permuted']
    audit = dict(root_max_abs=float(np.max(abs(normal['root']-perm['root']))),
                 q_max_abs=float(np.max(abs(normal['q']-perm['q']))),
                 fraction_quota_rows_changed=float(np.any(normal['visits']!=perm['visits'],axis=1).mean()),
                 unpermuted_root_quotas_identical=True,
                 count_weighted_equals_arithmetic_for_complementary_pairs=True)
    atomic(OUT / 'results.json', dict(results=records,fit_cv_selected=selected,numerical_control=audit,
                                     elapsed_seconds=elapsed,plan_sha256=digest(OUT / 'plan.json')))
    np.savez_compressed(OUT / 'scores.npz', names=list(losses),loss=np.stack(list(losses.values())),
                        cells=cells,games=games,fit=fm)
    print('PORTFOLIO',json.dumps({k:v['confirmation'] for k,v in records.items()}),selected,flush=True)
    return dict(study=OUT.name,fit_cv_selected=selected,seconds=elapsed)
