"""Game-fold-calibrated large-bank retrieval, with shared-player ablation."""
import json
import time
from pathlib import Path
import numpy as np
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .fit_policy import fit, loss_gradient
from .analyze_august import means
from .analyze_expanded import components
from .analyze_retrieval import distribution, fit_mix, test


def main():
    test()
    start = time.monotonic()
    out, source = ROOT/'aug-retrieval-large-v1', ROOT/'aug-expanded-search-v1'
    plan, worker = (json.loads((out/f).read_text()) for f in ('plan.json', 'worker.json'))
    stack = ROOT/'aug-expanded-stack-v1/results.json'
    stack_report = json.loads(stack.read_text())
    analysis_plan = dict(parent_plan_sha256=digest(out/'plan.json'), stack_sha256=digest(stack),
        worker_sha256=digest(out/'worker.json'),
        sources={p.name: digest(p) for p in [Path(__file__), *[Path(__file__).with_name(s) for s in
            ('analyze_retrieval.py', 'analyze_expanded.py', 'sample_data.py', 'fit_policy.py')]]},
        base='direct Elo temperature and subtree_sigma10 selected on expanded fit-gameCV; independent nested fits already recorded per fold',
        ablation='All kernels scored both with and without shared participants; report paired changes, all16 cells, full-support interpolation and exact unchanged baseline at lambda0. Golden CM pending.')
    pp = out/'analysis-plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == analysis_plan
    else:
        atomic(pp, analysis_plan)
    assert worker['plan_sha256'] == digest(out/'plan.json')
    assert worker['neighbors_sha256'] == digest(out/'neighbors.npz')
    assert worker['root_sha256'] == digest(out/'roots.npz')
    d = read('aug-tune-expanded-v1')
    rows, cells, games, fm, cv, ids, mask, target = (d[k] for k in ('rows', 'cells', 'games', 'fit', 'cv', 'ids', 'mask', 'target'))
    assert plan['sample_sha256'] == digest(ROOT/'aug-tune-expanded-v1/sample.json')
    n, ar, groups = len(rows), np.arange(len(rows)), cells % 4
    with np.load(out/'neighbors.npz') as f:
        neighbor = {k: f[k] for k in ('similarity', 'label', 'game', 'ply')}
        np.testing.assert_array_equal(f['game'], games)
        np.testing.assert_array_equal(f['ply'], [r['ply'] for r in rows])
    root = np.zeros((n, 2432))
    q = np.zeros(mask.shape)
    nodes = np.zeros(n)
    for lo in range(0, n, 1024):
        with np.load(source/f'{lo:06d}.npz') as f:
            hi = lo+len(f['game'])
            kk = f['ids'].shape[1]
            np.testing.assert_array_equal(f['game'], games[lo:hi])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in rows[lo:hi]])
            root[lo:hi] = f['z']
            q[lo:hi, :kk] = f['q'][-1, 1]
            nodes[lo:hi] = f['evaluated_nodes'][-1]
    logits = np.where(mask, root[:, 378:2346][ar[:, None], ids], 0.)
    versions = {'direct': [], 'search': []}
    for version, (fold, train) in enumerate([(None, fm), *[(f, fm & (cv != f)) for f in range(3)]]):
        direct, direct_params = np.zeros_like(q), {}
        for g in range(4):
            tr, pred = train & (groups == g), groups == g
            counts = np.bincount(cells[tr], minlength=16)
            fp = fit(logits[tr], np.zeros_like(q[tr]), mask[tr], target[tr], 'forward', 1/counts[cells[tr]])
            assert fp['converged']
            direct_params[str(g)] = fp
            direct[pred], _ = loss_gradient([fp['alpha'], fp['beta']], logits[pred], np.zeros_like(q[pred]),
                mask[pred], target[pred], 'forward', return_policy=True)
        versions['direct'].append((direct, direct_params))
        f = stack_report['fold_parameters']['subtree'][version]
        assert f['fold'] == fold
        search = np.einsum('a,ank->nk', np.array(f['weights']['sigma10']), components(logits, q, mask, f['parameters'], groups))
        versions['search'].append((search, dict(parameters=f['parameters'], weights=f['weights']['sigma10'], backup=f['backup'])))
    records, losses = {}, {}

    def record(name, p, oof, params, base, mode, kernel):
        loss = -np.log(p[ar, target])
        conf, cc = means(loss[~fm], cells[~fm]), means(oof[fm], cells[fm])
        extra = kernel is not None and params['mixing'] != 0.
        # Root hidden state is an additional complete prefix query in this pilot.
        # An integrated deployment could reuse it, but we do not credit that yet.
        records[name] = dict(parameters=params, base=base, filter=mode, kernel=kernel,
            training_equivalent_cm=None, mean_nodes=float(nodes.mean())*(base=='search')+int(extra),
            extra_full_prefix_queries=int(extra),
            fit_game_cv=dict(macro_ce=float(cc.mean()), expert_ce=float(cc[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()), expert_ce=float(conf[3::4].mean()), cells=conf.tolist()))
        losses[name] = loss

    for base, vv in versions.items():
        p, params = vv[0]
        oof = np.full(n, np.nan)
        for f in range(3):
            val = fm & (cv == f)
            oof[val] = -np.log(vv[f+1][0][ar[val], target[val]])
        record(base, p, oof, dict(base=params, mixing=0.), base, None, None)
    for field in ('confirmation', 'fit_game_cv'):
        for metric in ('macro_ce', 'expert_ce'):
            np.testing.assert_allclose(records['search'][field][metric], stack_report['results']['subtree_sigma10'][field][metric], atol=1e-12, rtol=0)
    for mode_ix, mode in enumerate(plan['filters']):
        for kernel in plan['kernels']:
            knn, ok = distribution(neighbor['similarity'][mode_ix], neighbor['label'][mode_ix], ids, mask, kernel['k'], kernel['temperature'])
            for base, vv in versions.items():
                p, params = vv[0]
                lam = fit_mix(p, knn, ok, target, cells, fm)
                mixed = (1-lam)*p+lam*np.where(ok[:, None], knn, p)
                assert (mixed[mask] > 0).all()
                np.testing.assert_allclose(mixed.sum(1), 1., atol=1e-14)
                oof = np.full(n, np.nan)
                fold_mix = []
                for f in range(3):
                    pp_ = vv[f+1][0]
                    val, train = fm & (cv == f), fm & (cv != f)
                    l = fit_mix(pp_, knn, ok, target, cells, train)
                    fold_mix.append(l)
                    pm = (1-l)*pp_+l*np.where(ok[:, None], knn, pp_)
                    oof[val] = -np.log(pm[ar[val], target[val]])
                name = f"{base}_{mode}_k{kernel['k']}_t{kernel['temperature']}"
                record(name, mixed, oof, dict(base=params, mixing=lam, fold_mixing=fold_mix), base, mode, kernel)
            print('Large retrieval fitted', mode, kernel, flush=True)
    selected = {metric: min(records, key=lambda name: records[name]['fit_game_cv'][metric]) for metric in ('macro_ce', 'expert_ce')}
    _, ix = np.unique(games[~fm], return_inverse=True)
    ng = ix.max()+1
    count = np.zeros((ng, 16))
    np.add.at(count, (ix, cells[~fm]), 1)
    w = np.random.default_rng(98318).multinomial(ng, np.full(ng, 1/ng), size=2000).astype(float)
    den = w@count
    assert (den>0).all()
    for name, rec in records.items():
        refs = [rec['base']]
        if rec['filter'] == 'different_players':
            refs.append(name.replace('different_players', 'all'))
        for ref in refs:
            delta = losses[name]-losses[ref]
            point = means(delta[~fm], cells[~fm])
            sums = np.zeros((ng, 16))
            np.add.at(sums, (ix, cells[~fm]), delta[~fm])
            draw = w@sums/den
            rec['delta_vs_'+ref] = dict(macro=float(point.mean()), expert=float(point[3::4].mean()),
                macro_ci95=np.quantile(draw.mean(1), [.025, .975]).tolist(), expert_ci95=np.quantile(draw[:, 3::4].mean(1), [.025, .975]).tolist())
    with np.load(out/'roots.npz') as f:
        drift = np.abs(f['z']-root)
    atomic(out/'results.json', dict(stage='Expanded August confirmation, potentially training-seen. Additional June datastore memory. Golden CM pending.',
        results=records, fit_cv_selected=selected, worker=worker, analysis_plan_sha256=digest(out/'analysis-plan.json'),
        analysis_seconds=time.monotonic()-start, query_root_drift=dict(max_abs=float(drift.max()), mean_abs=float(drift.mean()))))
    with (out/'scores.npz').open('wb') as f:
        np.savez_compressed(f, names=list(losses), loss=np.stack(list(losses.values())), cells=cells, games=games, fit=fm)
    for name, r in records.items():
        print(name, r['confirmation']['macro_ce'], r['confirmation']['expert_ce'], 'lambda', r['parameters']['mixing'], flush=True)
    print('SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
