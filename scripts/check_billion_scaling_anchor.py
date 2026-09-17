"""Audit an existing larger model against the frozen42point law, without refit."""
import importlib.util
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/tiny-scaling-v1'
LARGE=BASE.parent/'large-width-quality-v1'
s=importlib.util.spec_from_file_location('tiny',BASE/'analysis_source.py')
tiny=importlib.util.module_from_spec(s);s.loader.exec_module(tiny)


def main():
    comparison=tiny.read(LARGE/'comparison.json');assert comparison['passed'] and not comparison['final_tests_accessed']
    row=comparison['endpoint'];report=tiny.read(row['report'])
    assert tiny.sha(row['report'])==row['report_sha256']
    assert report['parameters']==row['parameters']==1064057624
    assert report['rows']==5371 and report['dataset_revision']==tiny.REV
    assert [report[m+'_count'] for m in tiny.METRICS]==[4616637,396483,151660]
    for m in tiny.METRICS:assert report[m+'_ce']==row['ce'][m]
    plan=tiny.read(LARGE/'plan.json')
    assert tiny.sha(LARGE/'analysis-plan.json')==comparison['plan_sha256']
    assert tiny.sha(LARGE/'plan.json')==tiny.read(LARGE/'prepared.json')['plan_sha256']
    for f,h in plan['source_sha256'].items():assert tiny.sha(Path(plan['source'])/f)==h
    run=ROOT/'results/pretrain/large1064-896-v1'
    metadata=tiny.read(run/'config.json');a=metadata['args'];cfg=metadata['config']
    expected=dict(batch_rows=512,decay_shape='linear',final_lr=.2,mtp_steps=64,plateau=4.,split_step=65,warmup_steps=32)
    assert json.loads(a['wsd_schedule'])==expected
    assert a['seed']==42 and a['lr_scale']==1 and a['initial_batch_rows']==512
    assert a['wsd_end_step']==a['steps']==896 and a['wsd_decay_start']==32
    assert cfg['layers']==16 and cfg['width']==2304 and cfg['head_dim']==64 and not cfg['fp8']
    assert metadata['world_size']==4 and a['micro_batch']==4
    assert tiny.sha(run/'last.pt')==row['checkpoint_pointer_sha256']
    assert tiny.sha(run/'quality-complete.json')==row['quality_marker_sha256']
    completion=tiny.read(run/'quality-complete.json')
    assert completion['complete'] and completion['plan_sha256']==tiny.sha(LARGE/'plan.json')
    fit_path=BASE/'extended-analysis/full-with-size-validation/fit.json'
    fit=tiny.read(fit_path)
    assert not any(r['parameters']==row['parameters'] for r in fit['rows'])
    scores={}
    for metric in tiny.METRICS:
        p=fit['fits']['ours'][metric]['fit']['parameters']
        predicted=float(tiny.predict(p,row['parameters']/1e7,row['tokens']/1e8))
        scores[metric]=dict(observed=row['ce'][metric],predicted=predicted,error=predicted-row['ce'][metric])
    result=dict(at=datetime.now(timezone.utc).isoformat(),passed=True,parameters=row['parameters'],tokens=row['tokens'],scores=scores,
        nominal_recipe_compatible=True,refitted=False,new_gpu_work=False,final_tests_accessed=False,
        provenance={str(p):tiny.sha(p) for p in (fit_path,LARGE/'comparison.json',LARGE/'plan.json',run/'config.json',Path(row['report']))},
        limits=['Existing result was previously viewed and used by an older, different analysis; not a blinded prospective test.',
                'Excluded from fitting the current42point law. Not independent of all earlier project decisions; no refit performed here.',
                'Nominal worker recipe matches; four-rank micro4 versus tiny one-rank micro16 changes floating-point arithmetic.',
                'Three inactive scalar padding parameters included in N, retained consistently with checkpoint metadata.',
                'One model size and one horizon do not establish high-data accuracy at this size or validate arbitrary large-compute forecasts.'])
    tiny.write(BASE/'analysis/billion-anchor-check.json',result)
    lines=['# Existing billion-parameter cross-check','',
        'Our existing1.064B-parameter model at469.8Mtokens was excluded from the current42point fit. Nominal recipe and original validation identity rechecked. No training or refit.','',
        '| Metric | Observed CE | Predicted CE | Prediction error |','|---|---:|---:|---:|']
    for m,v in scores.items():lines.append(f"| {m} | {v['observed']:.5f} | {v['predicted']:.5f} | {v['error']:+.5f} |")
    lines+=['',*result['limits']]
    (BASE/'analysis/BILLION_ANCHOR.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
