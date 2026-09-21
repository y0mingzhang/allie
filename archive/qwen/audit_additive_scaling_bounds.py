"""Coefficient-free approximation bounds for the requested additive law.

This is an algebraic diagnostic of observed losses, not another fitted model.
"""
import hashlib
import itertools
import json
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('/home/yimingz3/src/allie/results/recipe10x/tiny-scaling-v1')


def main():
    source=BASE/'extended-analysis/full-with-size-validation/fit.json'
    rows=json.loads(source.read_text())['rows'];assert len(rows)==42
    records={}
    for recipe in ('ours','qwen_recipe'):
        rr=[r for r in rows if r['recipe']==recipe]
        lookup={(r['parameters'],r['tokens']):r for r in rr}
        assert len(lookup)==len(rr)
        records[recipe]={}
        for metric in ('move','expert2400','expert2600'):
            contrasts=[]
            for ns in itertools.combinations(sorted({r['parameters'] for r in rr}),2):
                for ds in itertools.combinations(sorted({r['tokens'] for r in rr}),2):
                    keys=list(itertools.product(ns,ds))
                    if not all(k in lookup for k in keys):continue
                    values=[lookup[k]['ce'][metric] for k in keys]
                    cross=values[0]-values[1]-values[2]+values[3]
                    contrasts.append(dict(N=ns,D=ds,cells=[lookup[k]['name'] for k in keys],losses=values,
                        cross_difference=cross,minimum_possible_max_abs_error=abs(cross)/4,
                        smaller_model_data_gain=values[0]-values[1],larger_model_data_gain=values[2]-values[3]))
            records[recipe][metric]=dict(rectangles_checked=len(contrasts),
                worst=max(contrasts,key=lambda r:r['minimum_possible_max_abs_error']),rectangles=contrasts)
    result=dict(at=datetime.now(timezone.utc).isoformat(),source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records,
        proof='For every u(N)+v(D), the four-corner cross-difference is zero. Hence the observed cross-difference equals a signed sum of four errors; at least one absolute error is >=|difference|/4.',
        scope='Applies to the observed single-seed losses. It does not establish a bound on unknown multi-seed mean losses.',
        additional_compute=False,new_fitted_model=False)
    (BASE/'analysis/additive-bound-complete.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# A limit on additive-law precision','',result['proof'],'',
        '| Recipe | Metric | Observed data gain at smaller N | At larger N | Unavoidable maximum error bound |',
        '|---|---|---:|---:|---:|']
    for recipe,metrics in records.items():
        for metric,value in metrics.items():
            v=value['worst']
            lines.append(f"| {recipe} | {metric} | {v['smaller_model_data_gain']:.5f} | {v['larger_model_data_gain']:.5f} | {v['minimum_possible_max_abs_error']:.5f} |")
    lines+=['',result['scope'],'','This diagnostic does not change the requested law or its fitting objective. The seed study targets two influential endpoints; it cannot by itself estimate seed variability at every corner.']
    (BASE/'analysis/ADDITIVE_BOUND.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
