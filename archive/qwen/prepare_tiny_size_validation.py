"""Freeze one prospective size-interpolation test, reusing proven workers."""
import json,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from prepare_tiny_scaling import ROOT,batch,sha,read,write,STOP

OLD=ROOT/'results/recipe10x/tiny-scaling-v1'
OUT=ROOT/'results/recipe10x/tiny-size-validation-v1'


def main():
    assert not STOP.exists() and not (OUT/'prepared.json').exists()
    cpu=read(OUT/'cpu-shape.json');assert cpu['passed']
    sys.path.insert(0,str(OLD/'qwen-l16-w512-s256'))
    from scaled_native_qwen import Shape
    shape=Shape(layers=16,width=384,ff=1344,heads=3,kv_heads=1,head_dim=128)
    counts=dict(ours=cpu['parameters'],qwen_recipe=shape.parameter_count())
    write(OUT/'counts.json',dict(parameters=counts,qwen_shape=shape.__dict__))
    (OUT/'proof').symlink_to(OLD/'proof',target_is_directory=True)
    assert read(OUT/'proof/report.json')['passed']
    assert sha(OUT/'proof/report.json')==read(OUT/'proof/accepted.json')['report_sha256']
    design=dict(at=datetime.now(timezone.utc).isoformat(),purpose='Prospective withheld-size validation of the same additive scaling law.',
        reason='Extended30point fit has intermediate-size omitted prediction RMSE about0.06-0.08CE. Test a new interior N with frozen predictions before training.',
        cells=6,layers=16,width=384,steps=[256,512,1024],tokens=[s*512*1024 for s in (256,512,1024)],
        parameters=counts,qwen_shape=shape.__dict__,gpu_type='L40S',gpus_per_cell=1,
        training_hours_per_cell=1.5,evaluation_hours_per_cell=.25,bound_gpu_hours=10.5,
        unchanged='Worker/model/objective/optimizer/schedule/global batch/microbatch/seed/corpus/splits/evaluator source. Only model size/horizon/output path changes.',
        recovery='Reuse actual production recovery10474674 with identical ours source; existing native driver recovery proof retained. Additional384wide CPU causal/gradient check passed.',
        limitations='Qwen384 has3queryheads/1KVhead under the existing rounding rule; exact parameter counts differ and enter the law.',
        final_tests_accessed=False,final_pool_used=False)
    write(OUT/'design.json',design)
    prepared={}
    for recipe in ('ours','qwen_recipe'):
        tag='ours' if recipe=='ours' else 'qwen'
        for steps in (256,512,1024):
            name=f'{tag}-l16-w384-s{steps}';src=OLD/f'{tag}-l16-w512-s{steps}'
            dest=OUT/name;dest.mkdir();ev=dest/'evaluator';ev.mkdir()
            original=read(src/'plan.json');p=json.loads(json.dumps(original))
            worker_names=list(original['launcher_sha256']) if recipe=='ours' else list(original['source_sha256'])
            for f in worker_names:
                expected=(original['launcher_sha256'] if recipe=='ours' else original['source_sha256'])[f]
                assert sha(src/f)==expected
                shutil.copyfile(src/f,dest/f)
            p.update(at=design['at'],purpose='tiny-size-validation-v1 '+name,width=384,layers=16,
                     parameters=counts[recipe],phase='scaling_wsd',gpu_type='L40S',gpus_per_trial=1,
                     max_seconds=5400,max_hours_per_trial=1.5,training_envelope_gpu_hours=1.5,
                     matching_envelope_gpu_hours=.25,design_sha256=sha(OUT/'design.json'))
            p['trials']['quality']['name']='tiny-size-v1-'+name
            if recipe=='qwen_recipe':p['shape']={k:v for k,v in shape.__dict__.items() if k!='vocab'}
            # Original source and real-recovery report stay pinned at their old immutable paths.
            batch(dest/'run.sbatch','size-check-'+name,'L40S',1.5,f'exec .venv/bin/python {dest}/run.py --trial quality')
            field='launcher_sha256' if recipe=='ours' else 'source_sha256'
            p[field]={f:sha(dest/f) for f in worker_names}
            write(dest/'plan.json',p)
            ep=read(src/'evaluator/plan.json')
            for f,h in ep['source_sha256'].items():
                assert sha(src/'evaluator'/f)==h
                shutil.copyfile(src/'evaluator'/f,ev/f)
            command=f'exec .venv/bin/python {ev}/run.py'
            if recipe=='qwen_recipe':command=f'.venv/bin/python {ev}/prepare.py\n'+command
            batch(ev/'run.sbatch','size-check-'+name+'-eval','L40S',.25,command)
            ep.update(at=design['at'],phase='scaling_wsd',gpu_type='L40S',training_plan=str(dest/'plan.json'),
                      training_plan_sha256=sha(dest/'plan.json'),source_sha256={f:sha(ev/f) for f in ep['source_sha256']})
            write(ev/'plan.json',ep)
            for folder in (dest,ev):
                for f in folder.glob('*.py'):compile(f.read_text(),str(f),'exec')
            subprocess.run([sys.executable,str(dest/'run.py'),'--trial','quality','--dry-run'],check=True,stdout=subprocess.DEVNULL)
            prepared[name]=dict(recipe=recipe,layers=16,width=384,parameters=counts[recipe],steps=steps,
                                gpu_type='L40S',phase='scaling_wsd',training_hours=1.5,evaluation_hours=.25,
                                plan_sha256=sha(dest/'plan.json'),eval_plan_sha256=sha(ev/'plan.json'))
    write(OUT/'prepared.json',dict(passed=True,design_sha256=sha(OUT/'design.json'),packages=prepared,jobs_submitted=False))
    # Freeze both existing laws' predictions before any new job is submitted.
    import importlib.util
    spec=importlib.util.spec_from_file_location('tiny_analysis',OLD/'analysis_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths={'tiny_only':OLD/'analysis/first-three-forecast.json',
           'extended':OLD/'extended-analysis/short-horizon-forecast/fit.json'}
    predictions={}
    for name,c in prepared.items():
        predictions[name]={}
        for model,path in paths.items():
            fits=read(path)['fits'][c['recipe']]
            predictions[name][model]={m:float(module.predict(
                fits[m]['parameters'] if model=='tiny_only' else fits[m]['fit']['parameters'],
                c['parameters']/1e7,c['steps']*512*1024/1e8)) for m in module.METRICS}
    write(OUT/'predictions.json',dict(at=datetime.now(timezone.utc).isoformat(),predictions=predictions,
        fit_sources={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        all_new_size_outcomes_unobserved=True,jobs_not_yet_submitted=True))
    print(json.dumps(dict(prepared=len(prepared),parameters=counts,predictions=predictions),indent=2))


if __name__=='__main__':main()
