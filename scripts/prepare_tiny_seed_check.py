"""Targeted seed replication of the two largest short-horizon residuals."""
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from prepare_tiny_scaling import ROOT,STOP,read,write,sha,batch

OLD=ROOT/'results/recipe10x/tiny-scaling-v1'
BASE=OLD.parent/'tiny-seed-check-v1'


def load_module(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def main():
    assert not STOP.exists() and not BASE.exists()
    BASE.mkdir()
    fit_path=OLD/'extended-analysis/full-with-size-validation/fit.json'
    fit=read(fit_path)
    original=read(OLD/'analysis/fit.json')
    design=dict(at=datetime.now(timezone.utc).isoformat(),purpose='Targeted seed-variance diagnostic at the short largest-tiny size.',
        cells=6,seeds=[43,44,45],original_seed=42,layers=16,width=512,steps=256,tokens=134217728,
        gpu_type='A6000',phase='science_a6000',training_hours=.75,evaluation_hours=.25,bound_gpu_hours=6,
        reason='Largest full-fit residuals occur at these two original short endpoints. Replicate before attributing the residual to sampling noise or the additive law.',
        unchanged='Worker/optimizer/objective/model/schedule/batch/corpus/split/evaluator/hardware class. Only seed and output path change.',
        changed_launcher='Read the already-supported seed flag from plan instead of hardcoded42; assert saved checkpoint seed.',
        selection_bias='These two cells were deliberately selected after observing large residuals. This is a targeted diagnostic, not an unbiased estimate of noise over the entire grid.',
        decision='Report original and three new seeds, mean, sample SD, and residual against frozen42point fit. No automatic refit, law change, promotion, or goal completion.',
        fit_path=str(fit_path),fit_sha256=sha(fit_path),final_pool_used=False,final_tests_accessed=False)
    write(BASE/'design.json',design)
    (BASE/'proof').symlink_to(OLD/'proof',target_is_directory=True)
    assert read(BASE/'proof/report.json')['passed']
    prepared={};checks=[];baselines={}
    for recipe,tag in (('ours','ours'),('qwen_recipe','qwen')):
        baseline_name=f'{tag}-l16-w512-s256'
        baseline=next(r for r in original['rows'] if r['name']==baseline_name)
        assert baseline['gpu_type']=='A6000' and sha(baseline['report'])==baseline['report_sha256']
        baselines[recipe]=baseline
        src=OLD/baseline_name
        for seed in design['seeds']:
            name=f'{baseline_name}-seed{seed}'
            dest=BASE/name;dest.mkdir();ev=dest/'evaluator';ev.mkdir()
            p=read(src/'plan.json');field='launcher_sha256' if recipe=='ours' else 'source_sha256'
            for f,h in p[field].items():
                assert sha(src/f)==h;shutil.copyfile(src/f,dest/f)
            runner=dest/'run.py';text=runner.read_text()
            if recipe=='ours':
                assert text.count("'--seed','42'")==1
                text=text.replace("'--seed','42'","'--seed',str(plan['seed'])")
                needle="        assert shared['config']['width']==plan['width']"
                assert needle in text
                text=text.replace(needle,needle+"\n        assert shared['args']['seed']==plan['seed']")
            else:
                assert text.count("'--seed', '42'")==1 and text.count('seed=42,checkpoint_every')==1
                text=text.replace("'--seed', '42'","'--seed', str(plan['seed'])")
                text=text.replace('seed=42,checkpoint_every',"seed=plan['seed'],checkpoint_every")
            runner.write_text(text)
            original_runner=load_module(src/'run.py','original_runner_'+tag)
            new_runner=load_module(runner,'new_runner_'+tag)
            # Identical argv for seed42 when package/output locations are held equal.
            new_runner.PACKAGE=original_runner.PACKAGE
            trial=p['trials']['quality'];run=ROOT/'results/pretrain'/trial['name']
            if recipe=='ours':
                args=(p,trial,p['python'],1200)
            else:
                args=(p,trial,run,1200,False)
            assert original_runner.command(*args)==new_runner.command(*args)
            p['seed']=seed
            new_command=new_runner.command(*args)
            assert new_command[new_command.index('--seed')+1]==str(seed)
            p.update(at=design['at'],purpose='tiny-seed-check-v1 '+name,seed=seed,
                max_seconds=2700,max_hours_per_trial=.75,training_envelope_gpu_hours=.75,
                matching_envelope_gpu_hours=.25,design_sha256=sha(BASE/'design.json'))
            p['trials']['quality']['name']='tiny-seed-v1-'+name
            p['assumptions']=['Exact original recipe and hardware class; only seed and run identity differ.',
                              'Targeted short-horizon diagnostic, not a new architecture or LR trial.']
            batch(dest/'run.sbatch','seed-check-'+name,'A6000',.75,f'exec .venv/bin/python {dest}/run.py --trial quality')
            p[field]={f:sha(dest/f) for f in p[field]}
            write(dest/'plan.json',p)
            ep=read(src/'evaluator/plan.json')
            for f,h in ep['source_sha256'].items():
                assert sha(src/'evaluator'/f)==h;shutil.copyfile(src/'evaluator'/f,ev/f)
            command=f'exec .venv/bin/python {ev}/run.py'
            if recipe=='qwen_recipe':command=f'.venv/bin/python {ev}/prepare.py\n'+command
            batch(ev/'run.sbatch','seed-check-'+name+'-eval','A6000',.25,command)
            ep.update(at=design['at'],training_plan=str(dest/'plan.json'),training_plan_sha256=sha(dest/'plan.json'),
                      source_sha256={f:sha(ev/f) for f in ep['source_sha256']})
            write(ev/'plan.json',ep)
            for folder in (dest,ev):
                for f in folder.glob('*.py'):compile(f.read_text(),str(f),'exec')
            cmd=json.loads(subprocess.check_output([sys.executable,str(runner),'--trial','quality','--dry-run'],text=True))
            assert cmd[cmd.index('--seed')+1]==str(seed)
            prepared[name]=dict(recipe=recipe,layers=16,width=512,steps=256,seed=seed,parameters=baseline['parameters'],
                gpu_type='A6000',phase='science_a6000',training_hours=.75,evaluation_hours=.25,
                plan_sha256=sha(dest/'plan.json'),eval_plan_sha256=sha(ev/'plan.json'))
            checks.append(dict(name=name,seed42_argv_equivalent=True,requested_seed_verified=True,
                               worker_sources_unchanged=True,evaluator_sources_unchanged=True))
    tiny=load_module(OLD/'analysis_source.py','tiny_predict')
    predictions={recipe:{m:float(tiny.predict(fit['fits'][recipe][m]['fit']['parameters'],r['parameters']/1e7,r['tokens']/1e8))
                         for m in tiny.METRICS} for recipe,r in baselines.items()}
    write(BASE/'predictions.json',dict(baselines=baselines,predictions=predictions,fit_sha256=sha(fit_path),
                                      all_new_seed_outcomes_unobserved=True))
    write(BASE/'prepared.json',dict(passed=True,design_sha256=sha(BASE/'design.json'),packages=prepared,checks=checks,jobs_submitted=False))
    shutil.copyfile(OLD.parent/'tiny-size-validation-v1/review_helpers.py',BASE/'review_helpers.py')
    shutil.copyfile(ROOT/'scripts/review_tiny_seed_check.py',BASE/'analysis_source.py')
    watch=(OLD.parent/'tiny-size-validation-v1/watch_source.py').read_text()
    watch=watch.replace('tiny-size-validation-v1','tiny-seed-check-v1').replace('tiny-size-v1-','tiny-seed-v1-')
    (BASE/'watch_source.py').write_text(watch)
    write(BASE/'analysis-protocol.json',dict(source_sha256={f:sha(BASE/f) for f in ('analysis_source.py','review_helpers.py','watch_source.py')},
        scope=design['decision']))
    print(json.dumps(dict(prepared=len(prepared),bound_A6000_GPUh=6,predictions=predictions),indent=2))


if __name__=='__main__':main()
