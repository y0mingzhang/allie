"""Validate the frozen 24 endpoints and fit only E+A/N^alpha+B/D^beta.

CPU only. No training submissions, test access, architecture selection or plots.
The fourth horizon is excluded from the first fit, irrespective of finish order.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path('/home/yimingz3/src/allie')
BASE = ROOT / 'results/recipe10x/tiny-scaling-v1'
METRICS = ('move', 'expert2400', 'expert2600')
REV = '20a899ddf344ccaea74e273509a60e5a511125f8'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix('.partial.json')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def collect(accounting, selected_names=None):
    prepared = read(BASE / 'prepared.json')
    submitted = read(BASE / 'submitted.json')
    assert sha(BASE / 'design.json') == prepared['design_sha256'] == submitted['design_sha256']
    rows = []
    for name, cell in prepared['packages'].items():
        if selected_names is not None and name not in selected_names:
            continue
        package = BASE / name
        plan = read(package / 'plan.json')
        assert sha(package / 'plan.json') == cell['plan_sha256']
        assert sha(package / 'evaluator/plan.json') == cell['eval_plan_sha256']
        ids = submitted['cells'][name]
        for job in ids.values():
            record = accounting[str(job)]
            assert record['state'] == 'COMPLETED' and record['exit_code'] == '0:0', record
            assert 'gres/gpu=1' in record['alloc_tres'].split(','), record
        run = ROOT / 'results/pretrain' / ('tiny-v1-' + name)
        complete = read(run / 'quality-complete.json')
        assert complete['complete'] and complete['done']['step'] == cell['steps']
        assert complete['plan_sha256'] == cell['plan_sha256']
        pointer_key = 'checkpoint_pointer_sha256' if cell['recipe'] == 'ours' else 'checkpoint_sha256'
        assert sha(run / 'last.pt') == complete[pointer_key]
        source = Path(plan['source']) if cell['recipe'] == 'ours' else package
        for f, h in plan['source_sha256'].items():
            assert sha(source / f) == h, (name, f)
        ep = package / 'evaluator'
        for f, h in read(ep / 'plan.json')['source_sha256'].items():
            assert sha(ep / f) == h
        if cell['recipe'] == 'ours':
            receipt = read(ep / f"completion-{ids['evaluate']}.json")
            entry = next(r for r in receipt['results'] if not r['panel'])
        else:
            receipt = read(ep / str(ids['evaluate']) / 'completion.json')
            assert receipt['exports_ready_sha256'] == sha(ep / 'exports-ready.json')
            entry = receipt['reports'][0]
        assert receipt['passed'] and receipt['plan_sha256'] == cell['eval_plan_sha256']
        report = read(entry['report'])
        assert sha(entry['report']) == entry['sha256']
        assert report['rows'] == 5371 and report['dataset_revision'] == REV
        assert report['parameters'] == cell['parameters']
        assert [report[m + '_count'] for m in METRICS] == [4616637, 396483, 151660]
        rows.append(dict(name=name, recipe=cell['recipe'], layers=cell['layers'], width=cell['width'],
                         parameters=cell['parameters'], tokens=cell['steps'] * 512 * 1024,
                         steps=cell['steps'], gpu_type=cell['gpu_type'], jobs=ids,
                         allocated_gpu_hours=sum(accounting[str(j)]['elapsed_seconds'] / 3600 for j in ids.values()),
                         ce={m: report[m + '_ce'] for m in METRICS},
                         report=entry['report'], report_sha256=entry['sha256']))
    assert len(rows) == (24 if selected_names is None else len(selected_names))
    return rows


def predict(p, n, d):
    return p['E'] + p['A'] * n ** -p['alpha'] + p['B'] * d ** -p['beta']


def fit(rows, metric, fixed_floor=None):
    n = np.array([r['parameters'] / 1e7 for r in rows])
    d = np.array([r['tokens'] / 1e8 for r in rows])
    y = np.array([r['ce'][metric] for r in rows])

    def unpack(z):
        e, rest = (z[0], z[1:]) if fixed_floor is None else (fixed_floor, z)
        a, b, alpha, beta = rest
        return dict(E=float(e), A=float(a), B=float(b), alpha=float(alpha), beta=float(beta))

    lower = ([0.] if fixed_floor is None else []) + [1e-10, 1e-10, .0001, .0001]
    upper = ([float(y.min())] if fixed_floor is None else []) + [100., 100., 5., 5.]
    solutions = []
    for e0 in (0., float(y.min() * .5), float(y.min() * .9)):
        for exponent in (.1, .4, 1.):
            guess = ([e0] if fixed_floor is None else []) + [.5, .5, exponent, exponent]
            result = least_squares(lambda z: predict(unpack(z), n, d) - y, guess,
                                   bounds=(lower, upper), max_nfev=4000,
                                   ftol=1e-11, xtol=1e-11, gtol=1e-11)
            if result.success:
                solutions.append(result)
    assert solutions, 'No converged fit'
    best = min(solutions, key=lambda s: float(s.fun @ s.fun))
    p = unpack(best.x)
    singular = np.linalg.svd(best.jac, compute_uv=False)
    return dict(parameters=p, rmse=float(np.sqrt(np.mean(best.fun ** 2))),
                residuals=best.fun.tolist(), observations=len(rows),
                fixed_floor=fixed_floor, floor_at_zero=p['E'] < 1e-6,
                exponent_at_bound=min(p['alpha'], p['beta']) < .00011 or max(p['alpha'], p['beta']) > 4.999,
                jacobian_singular_values=singular.tolist(),
                near_nonidentifiable=bool(singular[-1] < singular[0] * 1e-8))


def evaluate(p, rows, metric):
    records = []
    for r in rows:
        pred = float(predict(p, r['parameters'] / 1e7, r['tokens'] / 1e8))
        records.append(dict(name=r['name'], observed=r['ce'][metric], predicted=pred,
                            error=pred-r['ce'][metric]))
    err = np.array([r['error'] for r in records])
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), max_abs=float(np.abs(err).max()), records=records)


def optimum(p, product):
    # Dimensionless n*d = (N/1e7)*(D/1e8). Idealized ND cost, not GPU time.
    a, b = p['alpha'], p['beta']
    n = ((a * p['A']) / (b * p['B']) * product ** b) ** (1 / (a + b))
    d = product / n
    assert np.isclose(a*p['A']*n**-a, b*p['B']*d**-b, rtol=1e-8)
    return dict(N=n*1e7, D=d*1e8, D_over_N=d*1e8/(n*1e7),
                loss=float(predict(p, n, d)), ND=product*1e15,
                nominal_6ND_flops=6*product*1e15)


def analyze(rows):
    fits = {}
    for recipe in ('ours', 'qwen_recipe'):
        rr = [r for r in rows if r['recipe'] == recipe]
        assert len(rr) == 12, (recipe, len(rr))
        fits[recipe] = {}
        for metric in METRICS:
            first = fit([r for r in rr if r['steps'] < 2048], metric)
            held = evaluate(first['parameters'], [r for r in rr if r['steps'] == 2048], metric)
            full = fit(rr, metric)
            # Two remaining sizes cannot identify E and alpha separately.
            # Report explicit floor scenarios, not an arbitrary unique fit.
            size_checks = []
            for n in sorted({r['parameters'] for r in rr}):
                train = [r for r in rr if r['parameters'] != n]
                target = [r for r in rr if r['parameters'] == n]
                for floor in (0., .5, 1.):
                    if floor >= min(r['ce'][metric] for r in train):
                        continue
                    ff = fit(train, metric, floor)
                    size_checks.append(dict(omitted_parameters=n, fixed_floor=floor,
                                            training_rmse=ff['rmse'], **evaluate(ff['parameters'], target, metric)))
            floor_sensitivity = [fit(rr, metric, e) for e in (0., .5, 1.)
                                 if e < min(r['ce'][metric] for r in rr)]
            product = max(r['parameters']*r['tokens'] for r in rows)/1e15
            allocations = [optimum(full['parameters'], product*m) for m in (1., 10., 100.)]
            for v in allocations:
                v['extrapolates_N'] = not min(r['parameters'] for r in rr) <= v['N'] <= max(r['parameters'] for r in rr)
                v['extrapolates_D'] = not min(r['tokens'] for r in rr) <= v['D'] <= max(r['tokens'] for r in rr)
            fits[recipe][metric] = dict(first_three_horizons=first, fourth_horizon_check=held,
                                       full_fit=full, leave_size_out_floor_scenarios=size_checks,
                                       fixed_floor_sensitivity=floor_sensitivity, idealized_ND_allocations=allocations)
    return fits


def self_check():
    truth = dict(E=1.1, A=.4, B=.3, alpha=.45, beta=.65)
    rows = [dict(parameters=n*1e7, tokens=d*1e8, ce=dict(move=float(predict(truth, n, d))))
            for n in (.3, 1.4, 6.) for d in (1.34, 2.68, 5.36, 10.74)]
    recovered = fit(rows, 'move')
    assert max(abs(recovered['parameters'][k]-v) for k, v in truth.items()) < 1e-5
    v = optimum(truth, 100.)
    n, d = v['N']/1e7, v['D']/1e8
    for factor in (.99, 1.01):
        assert predict(truth, n*factor, d/factor) > v['loss']
    return dict(synthetic_parameter_recovery=True, analytic_optimum_checked=True)


def larger_anchor_checks(fits):
    """Out-of-fit historical checks, not blinded or used to select the law."""
    frozen = read(BASE / 'larger-anchors.json')
    for row in frozen['rows']:
        report = read(row['source_report'])
        assert sha(row['source_report']) == row['source_report_sha256']
        assert report['rows'] == 5371 and report['dataset_revision'] == REV
        assert [report[m+'_count'] for m in METRICS] == [4616637,396483,151660]
        for metric in METRICS:
            assert abs(report[metric+'_ce']-row['ce'][metric]) < 1e-12
    result = {}
    for recipe, metrics in fits.items():
        rr = [r for r in frozen['rows'] if r['recipe'] == recipe]
        result[recipe] = {metric:evaluate(value['full_fit']['parameters'],rr,metric)
                          for metric,value in metrics.items()}
    return dict(scope=frozen['scope'], results=result)


def early_forecast(accounting):
    packages=read(BASE/'prepared.json')['packages']
    selected={name for name,c in packages.items() if c['steps']<2048}
    assert len(selected)==18
    rows=collect(accounting, selected)
    fits={recipe:{m:fit([r for r in rows if r['recipe']==recipe],m) for m in METRICS}
          for recipe in ('ours','qwen_recipe')}
    predictions={name:{m:float(predict(fits[c['recipe']][m]['parameters'],c['parameters']/1e7,
                                      c['steps']*512*1024/1e8)) for m in METRICS}
                 for name,c in packages.items() if c['steps']==2048}
    out=BASE/'analysis';out.mkdir(exist_ok=True)
    with (out/'first-three-forecast.json').open('x') as f:
        json.dump(dict(at=datetime.now(timezone.utc).isoformat(),rows=rows,fits=fits,
                       predictions=predictions,source_sha256=sha(__file__),
                       longer_horizon_results_read=False,
                       scope='Only18 shorter-horizon endpoint reports are read; some long jobs may already have finished, but are not consulted.'),f,indent=2,allow_nan=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--forecast-only', action='store_true')
    args = parser.parse_args()
    checks = self_check()
    if args.self_check:
        print(json.dumps(checks)); return
    if args.forecast_only:
        early_forecast(read(BASE/'monitor/observation.json')['jobs']); return
    out = BASE / 'analysis'
    out.mkdir(exist_ok=True)
    observation = read(BASE / 'monitor/observation.json')
    rows = collect(observation['jobs'])
    fits = analyze(rows)
    early_path=BASE/'analysis/first-three-forecast.json'
    if early_path.exists():
        early=read(early_path)
        for recipe, metrics in fits.items():
            for metric,value in metrics.items():
                for key,x in value['first_three_horizons']['parameters'].items():
                    assert abs(x-early['fits'][recipe][metric]['parameters'][key])<1e-8
        checks['matches_earlier_short_horizon_fit']=True
    anchors = larger_anchor_checks(fits)
    write(out / 'fit.json', dict(at=datetime.now(timezone.utc).isoformat(), rows=rows, fits=fits,
          larger_anchor_checks=anchors, larger_anchor_manifest_sha256=sha(BASE/'larger-anchors.json'),
          formula='E+A*(N/1e7)^(-alpha)+B*(D/1e8)^(-beta)', separate_recipe_floors=True,
          objective='Unweighted squared error in full original-validation CE', checks=checks,
          analysis_source_sha256=sha(__file__), final_tests_accessed=False,
          limitations=['One seed per cell: no seed-variance estimate or unconditional confidence intervals.',
                      'Leave-size-out leaves only two sizes, so floor scenarios expose nonidentifiability.',
                      'Smallest Qwen has one KV head; old fixed16-layer large anchors are a different shape trajectory.',
                      'ND allocation is an idealized FLOP proxy; actual hardware efficiency is not inferred.',
                      'Logged ours8/12-layer FLOPs retain16-layer attention accounting and are excluded.',
                      'Fourth-horizon errors, boundary solutions and floor sensitivity must be reviewed before extrapolation.']))
    with (out / 'results.csv').open('w') as f:
        keys = ['name', 'recipe', 'layers', 'width', 'parameters', 'tokens', 'gpu_type', 'allocated_gpu_hours', *METRICS]
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for r in rows:
            writer.writerow({**{k:r[k] for k in keys if k not in METRICS}, **r['ce']})
    lines = ['# Tiny scaling results', '', '24 validated independent endpoints. Formula: `L=E+A*(N/1e7)^(-alpha)+B*(D/1e8)^(-beta)`.', '',
             '| Recipe | Metric | E | alpha | beta | Full-fit RMSE | Held-out longest-horizon RMSE |',
             '|---|---|---:|---:|---:|---:|---:|']
    for recipe, metrics in fits.items():
        for metric, v in metrics.items():
            p = v['full_fit']['parameters']
            lines.append(f"| {recipe} | {metric} | {p['E']:.6f} | {p['alpha']:.4f} | {p['beta']:.4f} | {v['full_fit']['rmse']:.5f} | {v['fourth_horizon_check']['rmse']:.5f} |")
    lines += ['', 'Read fit.json for residuals, two-size floor scenarios and idealized optimal N/D. These are forecasts, not measured optimal allocations. Hardware costs retain their actual GPU labels. No final test or new training is triggered.', '',
              'This automatic report does not declare the high-fidelity goal complete. Held-out errors, floor sensitivity and extrapolation need review.']
    lines += ['', '## Larger-model extrapolation check', '',
              'Previously observed128M/483M fixed16-layer anchors are excluded from the tiny fit. This checks transfer to their different width/depth trajectory; it is not a blinded test.', '',
              '| Recipe | Metric | Larger-anchor RMSE | Maximum absolute error |',
              '|---|---|---:|---:|']
    for recipe, metrics in anchors['results'].items():
        for metric,v in metrics.items():
            lines.append(f"| {recipe} | {metric} | {v['rmse']:.5f} | {v['max_abs']:.5f} |")
    (out / 'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
