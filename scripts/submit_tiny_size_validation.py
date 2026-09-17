"""Submit six prospective validation chains once, within the existing cap."""
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from budget import ROOT,ledger,LIMITS
from submit_tiny_scaling import submit
from prepare_tiny_scaling import read,write,sha,STOP
BASE=ROOT/'results/recipe10x/tiny-size-validation-v1'


def main():
    assert not STOP.exists()
    prepared=read(BASE/'prepared.json');design=read(BASE/'design.json')
    assert prepared['passed'] and prepared['design_sha256']==sha(BASE/'design.json')
    assert read(BASE/'cpu-shape.json')['passed'] and read(BASE/'predictions.json')['all_new_size_outcomes_unobserved']
    jobs,account=ledger()
    registered=sum(j['gpus']*j['max_hours'] for j in jobs if j['purpose'].startswith('tiny-size-validation-v1 '))
    assert account['committed_gpu_hours']['scaling_wsd']+design['bound_gpu_hours']-registered<=LIMITS['scaling_wsd']+1e-8
    actual=subprocess.check_output(['sacct','-X','-nP','-j','10474674','--format=JobIDRaw,State,ExitCode'],text=True)
    assert '10474674|COMPLETED|0:0' in actual
    for name,c in prepared['packages'].items():
        p=BASE/name;plan=read(p/'plan.json');ev=read(p/'evaluator/plan.json')
        assert sha(p/'plan.json')==c['plan_sha256'] and sha(p/'evaluator/plan.json')==c['eval_plan_sha256']
        assert ev['training_plan_sha256']==c['plan_sha256']
        for f,h in plan['source_sha256'].items():assert sha((Path(plan['source']) if c['recipe']=='ours' else p)/f)==h
        if c['recipe']=='ours':
            for f,h in plan['launcher_sha256'].items():assert sha(p/f)==h
        for f,h in ev['source_sha256'].items():assert sha(p/'evaluator'/f)==h
    for name,c in sorted(prepared['packages'].items(),key=lambda kv:(-kv[1]['steps'],kv[1]['recipe']!='qwen_recipe')):
        folder=BASE/name
        train=submit(folder,'scaling_wsd','L40S',1.5,'tiny-size-validation-v1 '+name+' train')
        submit(folder/'evaluator','scaling_wsd','L40S',.25,'tiny-size-validation-v1 '+name+' evaluate',train)
    write(BASE/'submitted.json',dict(at=datetime.now(timezone.utc).isoformat(),proof_job_id=10474674,
        proof_reused_without_new_charge=True,cells={n:dict(train=read(BASE/n/'submission.json')['job_id'],
        evaluate=read(BASE/n/'evaluator/submission.json')['job_id']) for n in prepared['packages']},
        design_sha256=sha(BASE/'design.json'),predictions_sha256=sha(BASE/'predictions.json'),final_tests_accessed=False))
    print('ALL6SIZEVALIDATIONCHAINS_SUBMITTED',flush=True)


if __name__=='__main__':main()
