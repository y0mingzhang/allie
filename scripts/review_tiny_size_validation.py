"""Score frozen prospective forecasts against the six new-size evaluations."""
import importlib.util,json,math
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/tiny-size-validation-v1'


def main():
    spec=importlib.util.spec_from_file_location('review_helpers',BASE/'review_helpers.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.BASE=BASE
    prepared=module.read(BASE/'prepared.json')['packages']
    rows=module.collect(module.read(BASE/'monitor/observation.json')['jobs'],set(prepared))
    assert len(rows)==6
    forecasts=module.read(BASE/'predictions.json')
    for source in forecasts['fit_sources'].values():assert module.sha(source['path'])==source['sha256']
    scores={}
    for model in ('tiny_only','extended'):
        scores[model]={}
        for recipe in ('ours','qwen_recipe'):
            scores[model][recipe]={}
            for metric in module.METRICS:
                comparisons=[dict(name=r['name'],observed=r['ce'][metric],
                    predicted=forecasts['predictions'][r['name']][model][metric],
                    error=forecasts['predictions'][r['name']][model][metric]-r['ce'][metric])
                    for r in rows if r['recipe']==recipe]
                scores[model][recipe][metric]=dict(rmse=math.sqrt(sum(c['error']**2 for c in comparisons)/len(comparisons)),
                    max_abs=max(abs(c['error']) for c in comparisons),comparisons=comparisons)
    result=dict(at=datetime.now(timezone.utc).isoformat(),data_integrity_passed=True,rows=rows,scores=scores,
                prospective_forecasts_sha256=module.sha(BASE/'predictions.json'),
                source_sha256=module.sha(__file__),final_tests_accessed=False,
                goal_complete=False,scope='Prospective new-size predictions; all6 forecasts fixed before submission. No refit or promotion performed by reviewer.')
    module.write(BASE/'validation.json',result)
    lines=['# Prospective size validation','', 'Six independently trained endpoints; forecasts frozen before submission.','',
           '| Forecast | Recipe | Metric | RMSE | Maximum error |','|---|---|---|---:|---:|']
    for model,recipes in scores.items():
        for recipe,metrics in recipes.items():
            for metric,v in metrics.items():
                lines.append(f"| {model} | {recipe} | {metric} | {v['rmse']:.5f} | {v['max_abs']:.5f} |")
    (BASE/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
