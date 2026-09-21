"""Same additive law on all compatible measured sizes; preserve tiny-only fit.

Default: shorter new horizons + historical anchors, predicting the six new long
horizons without reading their reports. --full requires all24 new endpoints.
No new GPU work, model forms, hyperparameter search or final tests.
"""
import argparse
import importlib.util
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/tiny-scaling-v1'
spec=importlib.util.spec_from_file_location('tiny_fit',BASE/'analysis_source.py')
tiny=importlib.util.module_from_spec(spec);spec.loader.exec_module(tiny)


def records(full, include_size_validation=False):
    observation=tiny.read(BASE/'monitor/observation.json')
    packages=tiny.read(BASE/'prepared.json')['packages']
    selected=set(packages) if full else {n for n,c in packages.items() if c['steps']<2048}
    rows=tiny.collect(observation['jobs'],selected)
    assert tiny.read(BASE/'larger-anchor-compatibility.json')['passed']
    for source in tiny.read(BASE/'larger-anchors.json')['rows']:
        r=source.copy();report=tiny.read(r['source_report'])
        assert tiny.sha(r['source_report'])==r['source_report_sha256']
        assert report['parameters']==r['parameters'] and report['dataset_revision']==tiny.REV
        assert report['rows']==5371
        for m in tiny.METRICS:assert abs(report[m+'_ce']-r['ce'][m])<1e-12
        r['historical_anchor']=True;rows.append(r)
    assert len(rows)==(36 if full else 30)
    if include_size_validation:
        assert full, 'Size-validation refit requires all original endpoints'
        validation_base=BASE.parent/'tiny-size-validation-v1'
        accepted=tiny.read(validation_base/'validation.json')
        assert accepted['data_integrity_passed']
        assert accepted['prospective_forecasts_sha256']==tiny.sha(validation_base/'predictions.json')
        protocol=tiny.read(validation_base/'analysis-protocol.json')
        helper_path=validation_base/'review_helpers.py'
        assert tiny.sha(helper_path)==protocol['source_sha256']['review_helpers.py']
        helper_spec=importlib.util.spec_from_file_location('size_validation_helpers',helper_path)
        helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
        helper.BASE=validation_base
        additional=helper.collect(tiny.read(validation_base/'monitor/observation.json')['jobs'],
                                  set(tiny.read(validation_base/'prepared.json')['packages']))
        assert additional==accepted['rows'] and len(additional)==6
        rows.extend(additional)
        assert len(rows)==42
    return rows


def fit_and_check(rows):
    fits={}
    for recipe in ('ours','qwen_recipe'):
        rr=[r for r in rows if r['recipe']==recipe]
        assert len({r['parameters'] for r in rr}) in (5,6)
        fits[recipe]={}
        for metric in tiny.METRICS:
            model=tiny.fit(rr,metric)
            checks={}
            for dimension in ('parameters','tokens'):
                folds=[]
                for value in sorted({r[dimension] for r in rr}):
                    train=[r for r in rr if r[dimension]!=value]
                    target=[r for r in rr if r[dimension]==value]
                    fit=tiny.fit(train,metric)
                    interpolation=min(r[dimension] for r in train)<value<max(r[dimension] for r in train)
                    folds.append(dict(omitted=value,interpolation=interpolation,fit=fit,
                                      **tiny.evaluate(fit['parameters'],target,metric)))
                errors=[v['error'] for fold in folds for v in fold['records']]
                checks[dimension]=dict(folds=folds,pooled_rmse=(sum(e*e for e in errors)/len(errors))**.5,
                                       max_abs=max(map(abs,errors)))
                for label,interior in (('interpolation',True),('boundary_extrapolation',False)):
                    subset=[v['error'] for fold in folds if fold['interpolation']==interior for v in fold['records']]
                    checks[dimension][label]=dict(count=len(subset),
                        rmse=(sum(e*e for e in subset)/len(subset))**.5 if subset else None,
                        max_abs=max(map(abs,subset)) if subset else None)
            fits[recipe][metric]=dict(fit=model,cross_validation=checks)
    return fits


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    parser.add_argument('--include-size-validation',action='store_true');args=parser.parse_args()
    if args.include_size_validation and not args.full:
        parser.error('--include-size-validation requires --full')
    mode='full-with-size-validation' if args.include_size_validation else ('full' if args.full else 'short-horizon-forecast')
    out=BASE/'extended-analysis'/mode
    assert not out.exists(),'Preserve previous fit; no blind overwrite'
    rows=records(args.full,args.include_size_validation)
    out.mkdir(parents=True)
    fits=fit_and_check(rows)
    packages=tiny.read(BASE/'prepared.json')['packages']
    predictions={name:{m:float(tiny.predict(fits[c['recipe']][m]['fit']['parameters'],
                         c['parameters']/1e7,c['steps']*512*1024/1e8)) for m in tiny.METRICS}
                 for name,c in packages.items() if c['steps']==2048}
    early_path=BASE/'extended-analysis/short-horizon-forecast/fit.json'
    heldout=None
    if args.full and early_path.exists():
        forecast=tiny.read(early_path)
        heldout={recipe:{metric:tiny.evaluate(forecast['fits'][recipe][metric]['fit']['parameters'],
                            [r for r in rows if r['recipe']==recipe and r.get('steps')==2048],metric)
                        for metric in tiny.METRICS} for recipe in fits}
    product=max(r['parameters']*r['tokens'] for r in rows)/1e15
    allocations={}
    for recipe,metrics in fits.items():
        allocations[recipe]={}
        for metric,v in metrics.items():
            p=v['fit']['parameters'];scenarios=[]
            for scale in (.1,1.,10.):
                estimate=tiny.optimum(p,product*scale)
                sensitivity=[tiny.optimum(f['fit']['parameters'],product*scale)
                             for dimension in v['cross_validation'].values() for f in dimension['folds']]
                estimate['omitted_group_sensitivity']={key:[min(x[key] for x in sensitivity),max(x[key] for x in sensitivity)]
                                                       for key in ('N','D','D_over_N','loss')}
                estimate['extrapolates_N']=not min(r['parameters'] for r in rows if r['recipe']==recipe)<=estimate['N']<=max(r['parameters'] for r in rows if r['recipe']==recipe)
                estimate['extrapolates_D']=not min(r['tokens'] for r in rows)<=estimate['D']<=max(r['tokens'] for r in rows)
                scenarios.append(estimate)
            allocations[recipe][metric]=scenarios
    result=dict(at=datetime.now(timezone.utc).isoformat(),mode=mode,rows=rows,fits=fits,
                formula='E+A*(N/1e7)^(-alpha)+B*(D/1e8)^(-beta)',
                longest_horizon_predictions=predictions,early_forecast_validation=heldout,
                idealized_ND_allocations=allocations,
                source_sha256=tiny.sha(__file__),core_source_sha256=tiny.sha(BASE/'analysis_source.py'),
                compatibility_sha256=tiny.sha(BASE/'larger-anchor-compatibility.json'),
                limits=['Nominal optimizer/schedule/global batch match; width/depth/FF ratios/microbatch and some world sizes differ.',
                        'This is a new extended-domain analysis; original tiny-only forecasts remain unchanged and separately evaluated.',
                        'Independent positive coefficients and floor for each recipe/metric, unweighted least squares; no interaction term.',
                        'One seed per endpoint. Omitted-group ranges are stability checks, not confidence intervals.',
                        'ND allocation uses a nominal compute proxy and is not an actual GPU-time optimum or guarantee of optimal shape.',
                        'Leave-size-out and leave-token-horizon-out prediction errors delimit usefulness; a fit alone is not high fidelity.'],
                final_tests_accessed=False)
    tiny.write(out/'fit.json',result)
    size_count=len({r['parameters'] for r in rows if r['recipe']=='ours'})
    lines=['# Extended-domain additive scaling law','',f'Mode: {mode}; {len(rows)} endpoints, {size_count} sizes per recipe. Formula and fitting objective unchanged. Tiny-only forecasts are preserved.', '',
           '| Recipe | Metric | E | alpha | beta | Fit RMSE | Leave-size-out RMSE | Leave-horizon-out RMSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for recipe,metrics in fits.items():
        for metric,v in metrics.items():
            p=v['fit']['parameters'];cv=v['cross_validation']
            lines.append(f"| {recipe} | {metric} | {p['E']:.5f} | {p['alpha']:.4f} | {p['beta']:.4f} | {v['fit']['rmse']:.5f} | {cv['parameters']['pooled_rmse']:.5f} | {cv['tokens']['pooled_rmse']:.5f} |")
    lines+=['', 'Full residuals, held-out forecasts and conditional N/D allocations are in fit.json. All omitted-fold errors are retained; interpolation and boundary extrapolation errors are additionally separated. Shape trajectory/microbatch/world-size differences are recorded in larger-anchor-compatibility.json. These surfaces compare whole recipes, not an isolated architecture intervention. No claim of final goal completion follows automatically.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
