"""Admit the six bounded targeted seed checks; no duplicate or blind retry."""
import subprocess
from pathlib import Path
from datetime import datetime,timezone
from budget import ROOT,ledger,LIMITS
from submit_tiny_scaling import submit
from prepare_tiny_scaling import read,write,sha,STOP
from dei_capacity import snapshot
BASE=ROOT/'results/recipe10x/tiny-seed-check-v1'


def main():
    assert not STOP.exists()
    prepared=read(BASE/'prepared.json');design=read(BASE/'design.json')
    assert prepared['passed'] and prepared['design_sha256']==sha(BASE/'design.json')
    assert len(prepared['checks'])==6 and all(c['seed42_argv_equivalent'] and c['requested_seed_verified'] for c in prepared['checks'])
    assert read(BASE/'predictions.json')['all_new_seed_outcomes_unobserved']
    for f,h in read(BASE/'analysis-protocol.json')['source_sha256'].items():assert sha(BASE/f)==h
    capacity=snapshot();assert capacity['allowed_user_a6000']>=6,capacity
    jobs,account=ledger()
    registered=sum(j['gpus']*j['max_hours'] for j in jobs if j['purpose'].startswith('tiny-seed-check-v1 '))
    assert account['committed_gpu_hours']['science_a6000']+design['bound_gpu_hours']-registered<=LIMITS['science_a6000']+1e-8
    status=subprocess.check_output(['sacct','-X','-nP','-j','10474674','--format=JobIDRaw,State,ExitCode'],text=True)
    assert '10474674|COMPLETED|0:0' in status
    for name,c in prepared['packages'].items():
        p=BASE/name;plan=read(p/'plan.json');ev=read(p/'evaluator/plan.json')
        assert sha(p/'plan.json')==c['plan_sha256'] and sha(p/'evaluator/plan.json')==c['eval_plan_sha256']
        assert ev['training_plan_sha256']==c['plan_sha256'] and plan['seed']==c['seed']
        for f,h in plan['source_sha256'].items():assert sha((Path(plan['source']) if c['recipe']=='ours' else p)/f)==h
        if c['recipe']=='ours':
            for f,h in plan['launcher_sha256'].items():assert sha(p/f)==h
        for f,h in ev['source_sha256'].items():assert sha(p/'evaluator'/f)==h
    for name,c in sorted(prepared['packages'].items(),key=lambda kv:(kv[1]['recipe']!='qwen_recipe',kv[1]['seed'])):
        p=BASE/name
        train=submit(p,'science_a6000','A6000',.75,'tiny-seed-check-v1 '+name+' train')
        submit(p/'evaluator','science_a6000','A6000',.25,'tiny-seed-check-v1 '+name+' evaluate',train)
    write(BASE/'submitted.json',dict(at=datetime.now(timezone.utc).isoformat(),proof_job_id=10474674,
        proof_reused_without_new_charge=True,cells={n:dict(train=read(BASE/n/'submission.json')['job_id'],
        evaluate=read(BASE/n/'evaluator/submission.json')['job_id']) for n in prepared['packages']},
        design_sha256=sha(BASE/'design.json'),predictions_sha256=sha(BASE/'predictions.json'),final_tests_accessed=False))
    print('ALL6SEEDCHECKCHAINS_SUBMITTED',flush=True)


if __name__=='__main__':main()
