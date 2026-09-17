"""Score the targeted seed diagnostic without automatically refitting the law."""
import importlib.util
import statistics
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/tiny-seed-check-v1'


def main():
    spec=importlib.util.spec_from_file_location('helpers',BASE/'review_helpers.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.BASE=BASE
    prepared=a.read(BASE/'prepared.json')['packages']
    rows=a.collect(a.read(BASE/'monitor/observation.json')['jobs'],set(prepared));assert len(rows)==6
    prediction=a.read(BASE/'predictions.json');design=a.read(BASE/'design.json')
    assert a.sha(design['fit_path'])==prediction['fit_sha256']==design['fit_sha256']
    for r in rows:
        r['seed']=prepared[r['name']]['seed']
        assert a.read(BASE/r['name']/'plan.json')['seed']==r['seed']
    scores={}
    for recipe in ('ours','qwen_recipe'):
        reference=prediction['baselines'][recipe]
        assert a.sha(reference['report'])==reference['report_sha256']
        replicate=[r for r in rows if r['recipe']==recipe]
        assert sorted(r['seed'] for r in replicate)==[43,44,45]
        scores[recipe]={}
        for metric in a.METRICS:
            original=reference['ce'][metric];new=[r['ce'][metric] for r in replicate]
            values=[original,*new];mean=statistics.mean(values);sd=statistics.stdev(values)
            predicted=prediction['predictions'][recipe][metric]
            scores[recipe][metric]=dict(original_seed42=original,additional_seeds=new,mean=mean,sample_sd=sd,
                mean_new_seeds=statistics.mean(new),original_minus_new_mean=original-statistics.mean(new),
                frozen_prediction=predicted,prediction_minus_mean=predicted-mean,
                absolute_residual_over_sample_sd=abs(predicted-mean)/sd if sd else None)
    a.write(BASE/'review.json',dict(at=datetime.now(timezone.utc).isoformat(),data_integrity_passed=True,rows=rows,scores=scores,
        prospective_predictions_sha256=a.sha(BASE/'predictions.json'),source_sha256=a.sha(__file__),
        limits=['Two deliberately selected cells, four seeds each; not a grid-wide noise estimate.',
                'No automatic refit, promotion, law change, or goal completion.'],goal_complete=False,final_tests_accessed=False))
    lines=['# Short-horizon seed diagnostic','',
        '| Recipe | Metric | Original seed42 | Four-seed mean | Sample SD | Frozen prediction minus mean |',
        '|---|---|---:|---:|---:|---:|']
    for recipe,metrics in scores.items():
        for metric,v in metrics.items():
            lines.append(f"| {recipe} | {metric} | {v['original_seed42']:.5f} | {v['mean']:.5f} | {v['sample_sd']:.5f} | {v['prediction_minus_mean']:+.5f} |")
    lines+=['','Targets were selected after observing large residuals. This is a local seed-variance check, not an unbiased noise estimate over all sizes/data or a formal test of the entire scaling law.']
    (BASE/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
