"""Account allocated GPU-hours; guard new jobs against cumulative phase budgets."""
import argparse
import json
import math
import re
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
# Cumulative GPU-hour caps. Historical charges remain in results/jobs.json.
LIMITS={'preliminary':24.,'selection':96.,'big':384.,'science_a6000':135.,'scaling_wsd':455.,'hardware_h200':2.}
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}

def allocation_minutes(hours):
    """Reserve whole Slurm minutes, including its rounding of partial minutes."""
    assert math.isfinite(hours) and hours > 0
    return max(1, math.ceil(hours*60-1e-9))

def validate_phase_resources(phase, gpu_type, gpus, hours, script):
    if gpus == 0:
        assert gpu_type == 'CPU', 'Zero-GPU analysis must be labeled CPU'
        assert hours <= 48
        assert not re.search(r'^#SBATCH .*--(?:gres|gpus)', script, re.M), 'CPU analysis requests GPU resources'
        return
    if phase in ('selection','big','scaling_wsd'):
        assert gpu_type=='L40S' and gpus<=8 and hours<=48
    if phase=='science_a6000':
        assert gpu_type=='A6000' and 1<=gpus<=8 and hours<=3, 'Small A6000 science: at most one node and three hours per allocation'
        for option,value in (('partition','dei-group'),('qos','dei_group_qos'),('nodes','1')):
            assert re.search(r'^#SBATCH --'+option+'='+value+r'\s*$',script,re.M), 'A6000 science requires one DEI node and DEI QoS'
    if phase=='hardware_h200':
        assert gpu_type=='H200' and 1<=gpus<=4 and hours<=1
        for option,value in (('partition','preempt'),('qos','preempt_qos'),('nodes','1')):
            assert re.search(r'^#SBATCH --'+option+'='+value+r'\s*$',script,re.M), 'H200 calibration requires one preemptible node'

def ledger():
    path=ROOT/'results/jobs.json';jobs=json.loads(path.read_text())
    ids=','.join(str(j['id']) for j in jobs)
    out=subprocess.check_output(['sacct','-X','-nP','-j',ids,'--format=JobIDRaw,State,ElapsedRaw,AllocTRES,TimelimitRaw'],text=True)
    records={}
    for line in out.splitlines():
        jid,state,elapsed,tres,limit,*_=line.split('|')
        if not jid.isdigit():continue
        resources=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        gpu=int(resources.get('gres/gpu',0))
        kind=next((k.split(':',1)[1] for k in resources if k.startswith('gres/gpu:')),'unknown' if gpu else 'none')
        records[int(jid)]=dict(job_id=int(jid),state=state,elapsed_seconds=int(elapsed),gpus=gpu,gpu_type=kind,
            gpu_hours=gpu*int(elapsed)/3600,time_limit_minutes=int(limit))
    used={k:0. for k in LIMITS};committed=used.copy();by_type={};rows=[];active=0
    scheduled_by_phase={k:0 for k in LIMITS}
    for job in jobs:
        phase=job.get('phase','preliminary')
        r=records.get(job['id'],dict(job_id=job['id'],state='ACCOUNTING_PENDING',gpus=0,gpu_hours=0.))
        r['phase']=phase;r['purpose']=job['purpose']
        r['gpu_type']=job.get('gpu_type',r.get('gpu_type','unknown'))
        used[phase]+=r['gpu_hours'];by_type[r['gpu_type']]=by_type.get(r['gpu_type'],0)+r['gpu_hours']
        terminal=r['state'].split()[0].split('+')[0] in TERMINAL
        commitment=r['gpu_hours'] if terminal else max(r['gpu_hours'],job['gpus']*job['max_hours'],
            job['gpus']*r.get('time_limit_minutes',0)/60)
        committed[phase]+=commitment
        if not terminal:
            active+=job['gpus']
            scheduled_by_phase[phase]+=job['gpus']
        rows.append(r)
    # Pending afterok ancestors and descendants cannot overlap. Count the
    # maximum concurrent GPU demand over the dependency DAG, not every queued
    # successor as if it were already running. Unknown dependencies stay independent.
    from gpu_concurrency import peak_bound,parse_afterok,qos_peak_bound
    queue=subprocess.check_output(['squeue','-u','yimingz3','-h','-o','%i|%T|%E|%q'],text=True)
    dependencies={};job_qos={}
    for line in queue.splitlines():
        jid,state,dep,qos=line.split('|')
        if jid.isdigit():job_qos[int(jid)]=qos
        if jid.isdigit() and state=='PENDING':dependencies[int(jid)]=parse_afterok(dep)
    nonterminal={r['job_id'] for r in rows if r['state'].split()[0].split('+')[0] not in TERMINAL}
    requested={j['id']:j['gpus'] for j in jobs if j['id'] in nonterminal}
    dag_peak=peak_bound(requested,dependencies)
    caps={}
    try:
        raw_caps=subprocess.check_output(['sacctmgr','-nP','show','qos','normal','format=Name,MaxTRESPU'],text=True,timeout=10)
        for line in raw_caps.splitlines():
            qos,tres,*_=line.split('|')
            values=dict(v.split('=',1) for v in tres.split(',') if '=' in v)
            if qos=='normal' and values.get('gres/gpu','').isdigit():caps[qos]=int(values['gres/gpu'])
    except (subprocess.SubprocessError,OSError,ValueError):
        pass # Unknown/unavailable scheduler limits cannot lower the bound.
    peak=qos_peak_bound(requested,dependencies,job_qos,caps)
    report=dict(limits_gpu_hours=LIMITS,used_gpu_hours=used,committed_gpu_hours=committed,
        used_by_gpu_type=by_type,max_concurrent_gpus=24,scheduled_gpus=peak,queued_and_running_gpu_requests=active,
        concurrency_bound_method='Minimum of strict afterok DAG bound and sum of per-QoS bounds capped by live MaxTRESPU; unknown jobs/limits uncapped',
        dependency_only_gpu_bound=dag_peak,live_qos_gpu_caps=caps,
        concurrency_inputs=dict(weights=requested,dependencies=dependencies,job_qos=job_qos),
        scheduled_gpus_by_phase=scheduled_by_phase,jobs=rows)
    (ROOT/'results/compute.json').write_text(json.dumps(report,indent=2))
    return jobs,report

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command')
    sub.add_parser('ledger',help='Refresh and print accounting without submitting jobs')
    submit=sub.add_parser('submit');submit.add_argument('--phase',choices=LIMITS,required=True)
    submit.add_argument('--gpus',type=int,required=True);submit.add_argument('--gpu-type',required=True)
    submit.add_argument('--hours',type=float,required=True);submit.add_argument('--purpose',required=True)
    submit.add_argument('--dependency',help='Slurm dependency, e.g. afterok:12345')
    submit.add_argument('script');submit.add_argument('script_args',nargs=argparse.REMAINDER)
    a=p.parse_args();jobs,report=ledger()
    if a.command=='submit':
        assert not Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP').exists(), 'Controller STOP is present; no new submission'
        assert a.gpus>=0 and a.hours>0
        minutes=allocation_minutes(a.hours)
        reserved_hours=minutes/60
        assert report['committed_gpu_hours'][a.phase]+a.gpus*reserved_hours<=LIMITS[a.phase]+1e-8,'Phase compute cap exceeded'
        script=Path(a.script).read_text()
        validate_phase_resources(a.phase,a.gpu_type,a.gpus,reserved_hours,script)
        from gpu_concurrency import qos_peak_bound,peak_bound,parse_afterok
        if a.dependency:
            assert re.fullmatch(r'afterok:\d+(?::\d+)*',a.dependency),'Only explicit successful-job dependencies are supported'
        ci=report['concurrency_inputs'];candidate=-1
        weights={**ci['weights'],candidate:a.gpus}
        dependencies={**ci['dependencies'],candidate:parse_afterok(a.dependency or '')}
        match=re.search(r'^#SBATCH --qos=(\S+)\s*$',script,re.M)
        qos={**ci['job_qos'],candidate:match.group(1) if match else None}
        assert qos_peak_bound(weights,dependencies,qos,report['live_qos_gpu_caps'])<=24,'Project GPU concurrency cap exceeded'
        # The new headroom is A6000 capacity, not permission to add more L40S.
        l40s_ids={j['id'] for j in jobs if j.get('gpu_type','').lower()=='l40s'}
        if a.gpu_type.lower()=='l40s':
            l40s_ids.add(candidate)
            l40s_peak=qos_peak_bound({j:w for j,w in weights.items() if j in l40s_ids},dependencies,qos,report['live_qos_gpu_caps'])
            assert l40s_peak<=8,'At most8 L40S GPUs across QoS for new admissions'
        if a.phase=='science_a6000':
            phase_ids={j['id'] for j in jobs if j.get('phase')==a.phase}|{candidate}
            phase_peak=peak_bound({j:w for j,w in weights.items() if j in phase_ids},dependencies)
            assert phase_peak<=16,'At most16 A6000 GPUs across DEI science jobs'
            from dei_capacity import snapshot
            dei=snapshot()
            assert phase_peak<=dei['allowed_user_a6000'],f'DEI demand takes priority: {dei}'
        resource=re.search(r'^#SBATCH --gres=gpu:([^:]+):(\d+)\s*$',script,re.M)
        if a.gpus:
            assert resource and resource.group(1).lower()==a.gpu_type.lower() and int(resource.group(2))==a.gpus,'Script GPU request must match ledger declaration'
        else:
            assert not re.search(r'^#SBATCH .*--(?:gres|gpus)',script,re.M),'CPU-only ledger entry requests GPUs'
        duration=f'{minutes//60:02}:{minutes%60:02}:00'
        command=['sbatch','--parsable','--no-requeue','--time='+duration]
        if a.dependency:
            assert re.fullmatch(r'afterok:\d+(?::\d+)*',a.dependency),'Only explicit successful-job dependencies are supported'
            command+=['--dependency='+a.dependency]
        jid=int(subprocess.check_output([*command,a.script,*a.script_args],text=True).strip().split(';')[0])
        entry=dict(id=jid,purpose=a.purpose,gpus=a.gpus,gpu_type=a.gpu_type,max_hours=reserved_hours,phase=a.phase)
        jobs.append(entry)
        (ROOT/'results/jobs.json').write_text(json.dumps(jobs,indent=2));print(jid)
    else:print(json.dumps(report,indent=2))
