"""Human-readable scientific review of the completed scaling study.

Requires accepted evaluations and the combined fit. Does not declare success.
"""
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/tiny-scaling-v1'
NEW=BASE.parent/'tiny-size-validation-v1'
spec=importlib.util.spec_from_file_location('tiny',BASE/'analysis_source.py')
tiny=importlib.util.module_from_spec(spec);spec.loader.exec_module(tiny)


def main():
    full=tiny.read(BASE/'extended-analysis/full-with-size-validation/fit.json')
    original=tiny.read(BASE/'analysis/fit.json')
    validation=tiny.read(NEW/'validation.json')
    bounds=tiny.read(BASE/'analysis/additive-bound-complete.json')
    assert bounds['source_sha256']==tiny.sha(BASE/'extended-analysis/full-with-size-validation/fit.json')
    assert len(full['rows'])==42 and validation['data_integrity_passed']
    allocations=tiny.read(BASE/'extended-analysis/allocation-stability.json')
    assert 'extended_full' in allocations['sources']
    assert allocations['sources']['extended_full']['sha256']==tiny.sha(BASE/'extended-analysis/full-with-size-validation/fit.json')
    lines=['# Qwen versus ours: empirical scaling review','',datetime.now(timezone.utc).isoformat(),'',
        '42 measured endpoints: 24 new grid cells, six prospective intermediate-size checks, and 12 historical larger-model anchors. Original corpus, original validation split, move-sequence inputs. Final tests remain unopened.','',
        'The fitted law is `L = E + A*(N/1e7)^(-alpha) + B*(D/1e8)^(-beta)`. N is exact total parameter count; D is processed training tokens. Each recipe and metric has its own coefficients. Unweighted squared CE residuals; no interaction term.','',
        '## Full-data coefficients','',
        '| Recipe | Metric | E | A | B | alpha | beta | Fit RMSE |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for recipe,metrics in full['fits'].items():
        for metric in ('move','expert2400'):
            fit=metrics[metric]['fit'];p=fit['parameters']
            lines.append(f"| {recipe} | {metric} | {p['E']:.5f} | {p['A']:.5f} | {p['B']:.5f} | {p['alpha']:.4f} | {p['beta']:.4f} | {fit['rmse']:.5f} |")
    lines+=['','## Predictions checked before the final refit','',
        'New-size predictions were frozen before submitting those six runs. Longer-horizon predictions excluded the long reports from fitting; some long results were already visible to the investigator, so that check was not blinded. The final42-point model includes these observations and is not independently validated by them.','',
        '| Pre-refit model | Recipe | Metric | New-size RMSE | New-size max error | Long-horizon RMSE |',
        '|---|---|---|---:|---:|---:|']
    for label in ('tiny_only','extended'):
        for recipe in ('ours','qwen_recipe'):
            for metric in ('move','expert2400'):
                v=validation['scores'][label][recipe][metric]
                long=(original['fits'][recipe][metric]['fourth_horizon_check'] if label=='tiny_only'
                      else full['early_forecast_validation'][recipe][metric])
                lines.append(f"| {label} | {recipe} | {metric} | {v['rmse']:.5f} | {v['max_abs']:.5f} | {long['rmse']:.5f} |")
    lines+=['','## Omitted-group prediction checks','',
        'Each fold refits after removing an entire size or token horizon. Interior omissions test interpolation; boundary omissions test extrapolation. All errors are retained. These are sensitivity checks, not statistical confidence intervals.','',
        '| Recipe | Metric | Size: all RMSE | Size: interior RMSE | Size: boundary RMSE | Horizon: all RMSE |',
        '|---|---|---:|---:|---:|---:|']
    for recipe,metrics in full['fits'].items():
        for metric in ('move','expert2400'):
            cv=metrics[metric]['cross_validation'];n=cv['parameters'];d=cv['tokens']
            lines.append(f"| {recipe} | {metric} | {n['pooled_rmse']:.5f} | {n['interpolation']['rmse']:.5f} | {n['boundary_extrapolation']['rmse']:.5f} | {d['pooled_rmse']:.5f} |")
    lines+=['','Boundary errors depend on direction. The largest errors above are backward extrapolations to the smallest model/data endpoint. Forward checks are better, but still have material residuals:','',
        '| Recipe | Metric | Omit largest N: RMSE | Omit largest D: RMSE |',
        '|---|---|---:|---:|']
    for recipe,metrics in full['fits'].items():
        for metric in ('move','expert2400'):
            cv=metrics[metric]['cross_validation']
            n=max(cv['parameters']['folds'],key=lambda f:f['omitted'])
            d=max(cv['tokens']['folds'],key=lambda f:f['omitted'])
            lines.append(f"| {recipe} | {metric} | {n['rmse']:.5f} | {d['rmse']:.5f} |")
    billion_path=BASE/'analysis/billion-anchor-check.json'
    if billion_path.exists():
        billion=tiny.read(billion_path)
        assert billion['passed'] and not billion['refitted']
        lines+=['','An existing1.064B-parameter/469.8M-token run, excluded from the42point fit, has move/expert2400 prediction errors '
                f"{billion['scores']['move']['error']:+.5f}/{billion['scores']['expert2400']['error']:+.5f}CE. "
                'This is a previously seen historical point, not a blinded test, and covers one short horizon only. [Audit and caveats](analysis/BILLION_ANCHOR.md).']
    lines+=['','## Conditional compute allocation','',
        'At fixed C=N*D, the optimum satisfies `alpha*A*(N/1e7)^(-alpha) = beta*B*(D/1e8)^(-beta)`. Both recipes use identical C. These are nominal compute comparisons, not actual node-day forecasts or a final-run recommendation.','',
        '| Recipe | ND | N (M) | D (B) | D/N | Predicted move CE | Extrapolates N or D |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in allocations['records']:
        if r['fit']=='extended_full' and r['metric']=='move':
            extrapolates=not (r['N_in_fitted_range'] and r['D_in_fitted_range'])
            lines.append(f"| {r['recipe']} | {r['ND']:.0e} | {r['N']/1e6:.2f} | {r['D']/1e9:.3f} | {r['D_over_N']:.2f} | {r['loss']:.5f} | {'Yes' if extrapolates else 'No'} |")
    seed_path=BASE.parent/'tiny-seed-check-v1/review.json'
    seed_limit='- One training seed per endpoint. We have not separated seed variance from systematic law error.'
    if seed_path.exists():
        seed=tiny.read(seed_path)
        seed_base=seed_path.parent
        assert seed['data_integrity_passed'] and len(seed['rows'])==6
        assert seed['prospective_predictions_sha256']==tiny.sha(seed_base/'predictions.json')
        assert tiny.read(seed_base/'predictions.json')['fit_sha256']==tiny.sha(BASE/'extended-analysis/full-with-size-validation/fit.json')
        assert tiny.read(seed_base/'monitor/done.json')['analysis_returncode']==0
        lines+=['','## Targeted seed repeats','',
            'At the two largest short-horizon residuals, the original seed42 and three new seeds43/44/45 give four observations per recipe. Same16-layer/512-width shapes,134.2Mtokens and A6000 hardware class. The42-point fit and its predictions were frozen before the new runs; replicas are not pooled into that fit.','',
            '| Recipe | Metric | Four-seed mean | Seed sample SD | Frozen prediction minus mean |',
            '|---|---|---:|---:|---:|']
        for recipe in ('ours','qwen_recipe'):
            for metric in ('move','expert2400'):
                v=seed['scores'][recipe][metric]
                lines.append(f"| {recipe} | {metric} | {v['mean']:.5f} | {v['sample_sd']:.5f} | {v['prediction_minus_mean']:+.5f} |")
        lines+=['','These deliberately selected repeats estimate local variability only. They do not measure seed variance at other grid points or establish a global statistical test of the additive law. [Seed reports and provenance](../tiny-seed-check-v1/RESULTS.md).']
        seed_limit='- Only two endpoints have four seeds; all other fitted endpoints retain one seed. The targeted repeats cannot establish grid-wide uncertainty.'
    floor_path=BASE/'analysis/additive-error-floor.json'
    if floor_path.exists():
        floor=tiny.read(floor_path)
        assert floor['sources']['full']['sha256']==tiny.sha(BASE/'extended-analysis/full-with-size-validation/fit.json')
        assert floor['sources']['original_tiny']['sha256']==tiny.sha(BASE/'analysis/fit.json')
        ours_floor=floor['records']['original_tiny']['ours']['move']
        qwen_floor=floor['records']['original_tiny']['qwen_recipe']['move']
        lines+=['','## Is the fitting procedure leaving substantial accuracy unused?','',
            'On the original small-model grid, even allowing arbitrary additive size and data effects only reduces move RMSE '
            f"from{ours_floor['canonical_power_law_rmse']:.5f} to{ours_floor['minimum_rmse']:.5f} for ours and "
            f"from{qwen_floor['canonical_power_law_rmse']:.5f} to{qwen_floor['minimum_rmse']:.5f} for Qwen. "
            'Thus almost all of that fitting error is unavoidable under additivity on these observed losses. This is a mathematical error bound, not a replacement forecasting model. It excludes the historical anchors, so mixing those anchors cannot explain this particular finding. [Derivation, full-grid bounds and numerical certificates](analysis/ADDITIVE_ERROR_FLOOR.md).']
    lines+=['','## What still limits the inference','',
        seed_limit,
        f"- Rectangle contrasts force any additive size-plus-data function to miss at least one observed move loss by at least {bounds['records']['ours']['move']['worst']['minimum_possible_max_abs_error']:.5f}CE for ours and {bounds['records']['qwen_recipe']['move']['worst']['minimum_possible_max_abs_error']:.5f} for Qwen. These are bounds on the observed single-seed losses, not on unknown multi-seed means. See [the algebraic diagnostic](analysis/ADDITIVE_BOUND.md).",
        '- Larger historical models share nominal recipes but differ in width/depth, feed-forward ratios, microbatch and some world sizes; these are recipe-family surfaces, not an isolated architecture intervention.',
        '- Ours8/12-layer logged FLOPs overcount attention/gating. Those counters are excluded. N,D and allocated GPU-hours remain usable.',
        '- An analytic optimum and a small fitting residual do not establish predictive fidelity. Consult the prospective and omitted-group errors before using these forecasts.',
        '- The original small-grid tiny-only extrapolation to the larger anchors was too optimistic by up to0.19–0.21moveCE. Those failed forecasts remain preserved.',
        '', 'Full source records, expert2600 scores, all residuals and allocation sensitivity are in the linked JSON artifacts. No goal-completion claim is made by this reporting script.','',
        '[Combined fit](extended-analysis/full-with-size-validation/fit.json) · [Prospective size checks](../tiny-size-validation-v1/RESULTS.md) · [Allocation sensitivity](extended-analysis/ALLOCATION_STABILITY.md)']
    (BASE/'SCIENTIFIC_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(BASE/'SCIENTIFIC_REVIEW.md')


if __name__=='__main__':
    main()
