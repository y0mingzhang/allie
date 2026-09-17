"""Submit each approved grid cell once; all admissions use the cumulative ledger."""
import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from budget import ROOT,ledger,LIMITS
from dei_capacity import snapshot
from prepare_tiny_scaling import OUT,STOP,batch,sha,read,write

def submit(folder,phase,gpu,hours,purpose,dependency=None):
    receipt=folder/'submission.json';intent=folder/'submit-intent.json'
    jobs=json.loads((ROOT/'results/jobs.json').read_text())
    old=[j for j in jobs if j['purpose']==purpose]
    if receipt.exists():
        r=read(receipt);assert len(old)==1 and old[0]['id']==r['job_id'];return r['job_id']
    assert not old,'Submission exists in ledger; recover missing receipt explicitly'
    assert not intent.exists(),'Prior submission attempt requires inspection; no blind retry'
    assert not STOP.exists()
    cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/budget.py'),'submit','--phase',phase,
         '--gpus','1','--gpu-type',gpu,'--hours',str(hours),'--purpose',purpose]
    if dependency:cmd+=['--dependency','afterok:'+str(dependency)]
    cmd+=[str(folder/'run.sbatch')]
    with intent.open('x') as f:json.dump(dict(at=datetime.now(timezone.utc).isoformat(),command=cmd),f,indent=2)
    job=int(subprocess.check_output(cmd,text=True,timeout=180).strip())
    write(receipt,dict(job_id=job,command=cmd,dependency=dependency,at=datetime.now(timezone.utc).isoformat(),
                       script_sha256=sha(folder/'run.sbatch')))
    print(json.dumps(dict(purpose=purpose,job_id=job,dependency=dependency)),flush=True)
    return job

def main():
    assert not STOP.exists()
    prepared=read(OUT/'prepared.json');design=read(OUT/'design.json')
    assert prepared['passed'] and prepared['design_sha256']==sha(OUT/'design.json')
    assert read(OUT/'cpu-check.json')['passed']
    for name,cell in prepared['packages'].items():
        p=OUT/name;plan=read(p/'plan.json');ev=read(p/'evaluator/plan.json')
        assert sha(p/'plan.json')==cell['plan_sha256'] and sha(p/'evaluator/plan.json')==cell['eval_plan_sha256']
        assert ev['training_plan_sha256']==cell['plan_sha256']
        for n,h in plan['source_sha256'].items():assert sha((Path(plan['source']) if cell['recipe']=='ours' else p)/n)==h
        if cell['recipe']=='ours':
            for n,h in plan['launcher_sha256'].items():assert sha(p/n)==h
        for n,h in ev['source_sha256'].items():assert sha(p/'evaluator'/n)==h
    jobs,account=ledger();dei=snapshot()
    assert dei['allowed_user_a6000']==16,'Replan placements if DEI headroom changed'
    for phase,bound in design['bound_gpu_hours'].items():
        existing=sum(j['gpus']*j['max_hours'] for j in jobs if j['purpose'].startswith('tiny-scaling-v1 ') and j['phase']==phase)
        assert account['committed_gpu_hours'][phase]+bound-existing<=LIMITS[phase]+1e-8
    proof=OUT/'proof';proof.mkdir(exist_ok=True)
    if not (proof/'run.sbatch').exists():
        import shutil
        shutil.copyfile(ROOT/'scripts/proof_tiny_depth.py',proof/'proof.py')
        batch(proof/'run.sbatch','tiny-depth-recovery','A6000',.5,f'exec .venv/bin/python {proof}/proof.py')
    proofjob=submit(proof,'science_a6000','A6000',.5,'tiny-scaling-v1 depth recovery proof')
    # Start longest paths first; Qwen can run while the changed-depth ours proof runs.
    cells=sorted(prepared['packages'].items(),key=lambda kv:(kv[1]['recipe']=='ours',kv[1]['gpu_type']!='L40S',-kv[1]['steps'],-kv[1]['parameters']))
    for name,cell in cells:
        p=OUT/name
        train=submit(p,cell['phase'],cell['gpu_type'],cell['training_hours'],'tiny-scaling-v1 '+name+' train',
                     proofjob if cell['recipe']=='ours' else None)
        submit(p/'evaluator',cell['phase'],cell['gpu_type'],.25,'tiny-scaling-v1 '+name+' evaluate',train)
    write(OUT/'submitted.json',dict(at=datetime.now(timezone.utc).isoformat(),proof_job_id=proofjob,
        cells={n:dict(train=read(OUT/n/'submission.json')['job_id'],evaluate=read(OUT/n/'evaluator/submission.json')['job_id']) for n in prepared['packages']},
        design_sha256=sha(OUT/'design.json'),final_tests_accessed=False))
    print('ALL24CHAINS_SUBMITTED',flush=True)

if __name__=='__main__':main()
