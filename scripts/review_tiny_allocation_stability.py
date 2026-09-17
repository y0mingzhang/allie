"""Compare conditional allocations from preserved fits at identical ND costs."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/tiny-scaling-v1'
spec=importlib.util.spec_from_file_location('tiny',BASE/'analysis_source.py')
tiny=importlib.util.module_from_spec(spec);spec.loader.exec_module(tiny)


def main():
    paths={'tiny_short':BASE/'analysis/first-three-forecast.json',
           'extended_short':BASE/'extended-analysis/short-horizon-forecast/fit.json'}
    final=BASE/'extended-analysis/full-with-size-validation/fit.json'
    if final.exists():
        paths['extended_full']=final
    records=[]
    parameters={}
    for label,path in paths.items():
        source=tiny.read(path)
        for recipe in ('ours','qwen_recipe'):
            rr=[r for r in source['rows'] if r['recipe']==recipe]
            for metric in tiny.METRICS:
                fitted=source['fits'][recipe][metric]
                p=fitted['parameters'] if label=='tiny_short' else fitted['fit']['parameters']
                parameters[label,recipe,metric]=p
                for product in (1e17,1e18,1e19):
                    optimum=tiny.optimum(p,product/1e15)
                    records.append(dict(fit=label,recipe=recipe,metric=metric,**optimum,
                        N_cost_exponent=p['beta']/(p['alpha']+p['beta']),
                        D_cost_exponent=p['alpha']/(p['alpha']+p['beta']),
                        N_in_fitted_range=min(r['parameters'] for r in rr)<=optimum['N']<=max(r['parameters'] for r in rr),
                        D_in_fitted_range=min(r['tokens'] for r in rr)<=optimum['D']<=max(r['tokens'] for r in rr)))
    for r in records:
        p=parameters[r['fit'],r['recipe'],r['metric']]
        alternatives=[v for v in records if v['recipe']==r['recipe'] and v['metric']==r['metric']
                      and v['ND']==r['ND'] and v['fit']!=r['fit']]
        r['other_fit_allocation_regret']={v['fit']:float(tiny.predict(p,v['N']/1e7,v['D']/1e8)-r['loss'])
                                         for v in alternatives}
        assert all(regret>=-1e-10 for regret in r['other_fit_allocation_regret'].values())
    result=dict(at=datetime.now(timezone.utc).isoformat(),records=records,
        sources={k:dict(path=str(v),sha256=tiny.sha(v)) for k,v in paths.items()},
        source_sha256=tiny.sha(__file__),
        formula='C=N*D; N*=((alpha*A)/(beta*B)*(C/1e15)^beta)^(1/(alpha+beta))*1e7; D*=C/N*',
        interpretation='Conditional optimum of each fitted additive law. Common nominal ND costs, not hardware-time optima.',
        caveats=['Differences across fit domains are sensitivity, not confidence intervals.',
                 'Being inside individual N and D ranges does not guarantee dense joint coverage.',
                 'Cross-fit allocation regret evaluates choices under fitted surfaces, not observed training losses.',
                 'Predictive residuals and prospective tests must be inspected before selecting training.',
                 'No new training, test access, or goal completion.'])
    out=BASE/'extended-analysis'
    tiny.write(out/'allocation-stability.json',result)
    lines=['# Allocation sensitivity','',result['interpretation'],'',
        'Illustrative common cost: ND=1e18 parameter-tokens (nominal6ND=6e18 FLOPs).','',
        '| Recipe | Fit domain | N (M) | D (B) | D/N | Predicted move CE | N,D within fitted ranges |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in records:
        if r['metric']=='move' and r['ND']==1e18:
            lines.append(f"| {r['recipe']} | {r['fit']} | {r['N']/1e6:.2f} | {r['D']/1e9:.3f} | {r['D_over_N']:.2f} | {r['loss']:.5f} | {r['N_in_fitted_range']}, {r['D_in_fitted_range']} |")
    state=('The combined fit and prospective checks are complete; their scientific limitations remain material.'
           if final.exists() else 'The combined full fit and prospective prediction checks remain necessary.')
    lines+=['','Allocation and loss predictions change with fitted domain. The tiny-only fit extrapolates N at this cost and has already failed the larger-anchor check. The extended fit includes those anchors but misses some small-model long horizons. Neither low training error nor an analytic optimum proves reliable allocation. '+state]
    lines+=['','The loss surface can be flat around its optimum: changing the selected size need not cost much predicted CE. Cross-fit allocation regret is recorded in JSON, but this is conditional on the fitted laws and has not been measured by training at these allocations.']
    (out/'ALLOCATION_STABILITY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
