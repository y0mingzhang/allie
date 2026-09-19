"""Reusable small calibration screen on the fixed expanded August game folds."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from .sample_data import read
from .service import ROOT, atomic
from .balanced_eval import digest
from .analyze_august import means
from .analyze_header_adaptation import tilted


def analyze(study, features, costs, specification, bounds=(0., 20.)):
    start = time.monotonic(); out = ROOT / study
    d = read('aug-tune-expanded-v1')
    cells, games, fm, cv, mask, target = (d[k] for k in ('cells', 'games', 'fit', 'cv', 'mask', 'target'))
    n, k = mask.shape; ar = np.arange(n)
    surface = ROOT / 'aug-budget-surface-v2/policies.npz'
    with np.load(surface) as f:
        np.testing.assert_array_equal(f['games'], games)
        parents = f['policy'][-1].astype(float)
        nodes = f['nodes'][-1]
    assert 'parent' not in features
    features = {'parent': np.zeros((n, k)), **features}
    plan = dict(specification=specification, features=list(features), bounds=list(bounds),
                source_sha256=digest(Path(__file__)), surface_sha256=digest(surface),
                worker_sha256=digest(out / 'worker.json'),
                fitting='Each scalar fit is on full fit games or its training-game fold only. Equal-cell CE. Method selection uses three-fold game CV; report every arm on disjoint August confirmation. No golden loss or CM in fitting.')
    pp = out / 'analysis-plan.json'
    if pp.exists(): assert json.loads(pp.read_text()) == plan
    else: atomic(pp, plan)
    records, losses = {}, {}
    for name, ratio in features.items():
        assert ratio.shape == mask.shape and np.isfinite(ratio).all()
        for by_elo in ((False,) if name == 'parent' else (False, True)):
            label = name + ('_elo' if by_elo else '')
            group = cells % 4 if by_elo else np.zeros(n, int)
            oof = np.full(n, np.nan); parameters = []
            for vi, (fold, train) in enumerate([(None, fm), *[(f, fm & (cv != f)) for f in range(3)]]):
                p = parents[vi] / parents[vi].sum(1, keepdims=True)
                base = np.where(mask, np.log(np.maximum(p, 1e-300)), 0.)
                pred = np.zeros((n, k)); coefficients = {}
                for g in np.unique(group):
                    take = train & (group == g); allg = group == g
                    ct = np.bincount(cells[take], minlength=16)
                    w = 1 / ct[cells[take]]; w /= w.sum()
                    fun = lambda a: tilted(a, base[take], ratio[take], mask[take], target[take], w)
                    alpha = 0.
                    if name != 'parent':
                        opt = minimize_scalar(fun, bounds=bounds, method='bounded', options=dict(xatol=1e-7))
                        assert opt.success
                        alpha = min((0., float(opt.x), *bounds), key=fun)
                    pred[allg] = tilted(alpha, base[allg], ratio[allg], mask[allg], target[allg])
                    coefficients[str(int(g))] = alpha
                assert np.isfinite(pred).all() and (pred[mask] > 0).all()
                ll = -np.log(pred[ar, target]); parameters.append(dict(fold=fold, coefficients=coefficients))
                if fold is None: losses[label] = ll
                else: oof[fm & (cv == fold)] = ll[fm & (cv == fold)]
            a, b = means(losses[label][~fm], cells[~fm]), means(oof[fm], cells[fm])
            extra = np.zeros(n) if name == 'parent' else np.asarray(costs[name]['extra_nodes'])
            records[label] = dict(parameters=parameters, training_equivalent_cm=None,
                mean_nodes=float(np.mean(nodes + extra)), cost={} if name == 'parent' else costs[name]['summary'],
                confirmation=dict(macro_ce=float(a.mean()), expert_ce=float(a[3::4].mean()), cells=a.tolist()),
                fit_game_cv=dict(macro_ce=float(b.mean()), expert_ce=float(b[3::4].mean())))
    selected = {m: min(records, key=lambda n: records[n]['fit_game_cv'][m]) for m in ('macro_ce', 'expert_ce')}
    _, ix = np.unique(games[~fm], return_inverse=True); ng = ix.max()+1
    ct = np.zeros((ng, 16)); np.add.at(ct, (ix, cells[~fm]), 1)
    draws = np.random.default_rng(41912).multinomial(ng, np.full(ng, 1/ng), size=2000).astype(float)
    den = draws @ ct; assert (den > 0).all()
    for name, rec in records.items():
        delta = losses[name] - losses['parent']; point = means(delta[~fm], cells[~fm])
        sums = np.zeros((ng, 16)); np.add.at(sums, (ix, cells[~fm]), delta[~fm])
        boot = draws @ sums / den
        rec['delta_vs_parent'] = dict(macro=float(point.mean()), expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1), [.025,.975]).tolist(), expert_ci95=np.quantile(boot[:,3::4].mean(1), [.025,.975]).tolist())
    result = dict(results=records, fit_cv_selected=selected, analysis_seconds=time.monotonic()-start, analysis_plan_sha256=digest(pp))
    atomic(out / 'results.json', result)
    np.savez_compressed(out / 'scores.npz', names=list(losses), loss=np.stack(list(losses.values())), cells=cells, games=games, fit=fm)
    print(json.dumps({name: records[name] for name in set(selected.values()) | {'parent'}}, indent=2), flush=True)
    return result
