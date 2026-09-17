"""Lower bounds on observed-loss error for every additive size/data law.

Relax the power laws to arbitrary per-size and per-horizon offsets solely to
bound achievable error. These offsets are not a replacement predictor: they
cannot forecast new sizes or horizons. Canonical fits/weights remain unchanged.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

BASE = Path('/home/yimingz3/src/allie/results/recipe10x/tiny-scaling-v1')


def bounds(rows, metric):
    sizes = sorted({r['parameters'] for r in rows})
    horizons = sorted({r['tokens'] for r in rows})
    x = np.zeros((len(rows), len(sizes) + len(horizons)))
    for i, r in enumerate(rows):
        x[i, sizes.index(r['parameters'])] = 1
        x[i, len(sizes) + horizons.index(r['tokens'])] = 1
    y = np.array([r['ce'][metric] for r in rows])
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = x @ coef - y
    assert np.max(np.abs(x.T @ residual)) < 1e-10

    # Minimize t subject to -t <= X*offsets - observed <= t.
    a = np.vstack([np.column_stack([x, -np.ones(len(y))]),
                   np.column_stack([-x, -np.ones(len(y))])])
    b = np.concatenate([y, -y])
    objective = np.zeros(x.shape[1] + 1)
    objective[-1] = 1
    lp = linprog(objective, A_ub=a, b_ub=b,
                 bounds=[(None, None)] * x.shape[1] + [(0, None)],
                 method='highs')
    assert lp.success, lp.message
    assert np.max(a @ lp.x - b) < 1e-8
    assert abs(np.max(np.abs(x @ lp.x[:-1] - y)) - lp.fun) < 1e-8
    dual = lp.ineqlin.marginals
    assert np.max(np.abs(a[:, :-1].T @ dual)) < 1e-8
    assert abs(b @ dual - lp.fun) < 1e-8
    return dict(observations=len(y), sizes=len(sizes), horizons=len(horizons),
                design_rank=int(rank), residual_degrees_of_freedom=len(y)-int(rank),
                minimum_rmse=float(np.sqrt(np.mean(residual**2))),
                minimum_max_abs_error=float(lp.fun),
                squared_error_floor=float(residual @ residual),
                primal_dual_gap=float(abs(b @ dual - lp.fun)),
                least_squares_residuals=residual.tolist(),
                minimax_residuals=(x @ lp.x[:-1]-y).tolist(),
                dual_upper=dual[:len(y)].tolist(), dual_lower=dual[len(y):].tolist(),
                cells=[r['name'] for r in rows])


def verify():
    rows = [dict(name=f'{i}-{j}', parameters=i, tokens=j,
                 ce={'move': float(2*i+3*j)}) for i in (1, 2) for j in (1, 2)]
    exact = bounds(rows, 'move')
    assert exact['minimum_rmse'] < 1e-12
    assert exact['minimum_max_abs_error'] < 1e-12
    rows[-1]['ce']['move'] += 1
    perturbed = bounds(rows, 'move')
    assert abs(perturbed['minimum_rmse']-.25) < 1e-12
    assert abs(perturbed['minimum_max_abs_error']-.25) < 1e-12


def main():
    verify()
    source = BASE/'extended-analysis/full-with-size-validation/fit.json'
    data = json.loads(source.read_text())
    assert len(data['rows']) == 42
    sources = {'full': source, 'original_tiny': BASE/'analysis/fit.json'}
    results = {}
    for domain, fit_path in sources.items():
        fits = json.loads(fit_path.read_text())
        results[domain] = {}
        for recipe in ('ours', 'qwen_recipe'):
            rows = [r for r in data['rows'] if r['recipe'] == recipe]
            if domain == 'original_tiny':
                rows = [r for r in rows if r.get('width') in (128, 256, 512)]
                assert len(rows) == 12
            results[domain][recipe] = {}
            for metric in ('move', 'expert2400', 'expert2600'):
                v = bounds(rows, metric)
                entry = fits['fits'][recipe][metric]
                fitted = entry.get('fit', entry.get('full_fit'))
                if fitted is None:
                    raise KeyError(f'Unknown canonical fit schema: {list(entry)}')
                v['canonical_power_law_rmse'] = fitted['rmse']
                assert v['minimum_rmse'] <= fitted['rmse'] + 1e-8
                v['fraction_of_canonical_squared_error_unavoidable_by_additivity'] = (
                    v['minimum_rmse'] / fitted['rmse'])**2
                results[domain][recipe][metric] = v
    result = dict(at=datetime.now(timezone.utc).isoformat(), records=results,
                  sources={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for k, p in sources.items()},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Deterministic bounds on the observed losses, not unknown seed means or future predictive errors.',
                  interpretation='Any E+A/N^alpha+B/D^beta predictor lies in the relaxed additive class. It cannot have smaller observed RMSE or maximum error than these minima. The two minima optimize different norms and need not share offsets.',
                  canonical_fit_modified=False, new_prediction_model=False, synthetic_checks_passed=True)
    out = BASE/'analysis/additive-error-floor.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# How much observed fitting error is unavoidable?', '',
             result['interpretation'], '',
             '| Data | Recipe | Metric | Canonical fit RMSE | Minimum additive RMSE | Minimum additive max error | Unavoidable fraction of squared error |',
             '|---|---|---|---:|---:|---:|---:|']
    for domain, recipes in results.items():
        for recipe, metrics in recipes.items():
            for metric in ('move', 'expert2400'):
                v = metrics[metric]
                lines.append(f"| {domain} | {recipe} | {metric} | {v['canonical_power_law_rmse']:.5f} | {v['minimum_rmse']:.5f} | {v['minimum_max_abs_error']:.5f} | {v['fraction_of_canonical_squared_error_unavoidable_by_additivity']:.1%} |")
    lines += ['', 'The original tiny grid excludes historical large-model anchors and the later384-width checks. Its bounds therefore do not rely on combining those historical runs.', '',
              'Least squares supplies the RMSE bound; a linear program with checked primal feasibility and matching dual objective supplies the maximum-error bound. Synthetic separable and known quarter-unit error cases passed.', '',
              result['scope'], 'The per-size/per-horizon offsets are only a mathematical relaxation for this audit. They do not forecast new N or D, replace the requested law, change weights, or justify narrowing the goal.']
    (BASE/'analysis/ADDITIVE_ERROR_FLOOR.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
