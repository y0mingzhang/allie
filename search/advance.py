"""Cheap pilot observer: account own allocation and analyze once; never submit jobs."""
import argparse,json,subprocess,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/search-v1'
STOP=Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')

def tick():
    path=OUT/'status.json';state=json.loads(path.read_text())
    known={j['id'] for j in state['jobs']}
    for receipt in sorted((OUT/'job-receipts').glob('*.json')):
        job=json.loads(receipt.read_text())
        if job['id'] not in known:state['jobs'].append(job);known.add(job['id'])
    if STOP.exists(): return 'controller STOP; observer exiting',True
    total=0.;live=False
    for job in state['jobs']:
        jobid=job['id'].split(';')[0]
        text=subprocess.check_output(['sacct','-D','-j',jobid,'-n','-P','-o','JobIDRaw,State,ElapsedRaw,Start,End'],text=True)
        entries=[x.split('|') for x in text.splitlines() if x.split('|')[0]==jobid]
        if not entries:
            live=True;total+=job.get('gpu_hours',0.);continue
        # Persist every start incarnation: a Slurm requeue must not erase earlier usage.
        charges=job.setdefault('allocations',{})
        for entry in entries:
            _,status,seconds,started,ended=entry[:5]
            if started not in ('Unknown','None','') and int(seconds):
                charges[started]=max(charges.get(started,0),int(seconds))
        _,status,seconds=entries[-1][:3]
        job['state']=status
        job['elapsed_seconds']=max(job.get('elapsed_seconds',0),sum(charges.values()))
        job['gpu_hours']=job['elapsed_seconds']*job['gpus']/3600.;total+=job['gpu_hours']
        live |= status in ('PENDING','RUNNING','CONFIGURING','COMPLETING','REQUEUED','SUSPENDED')
    state['gpu_hours']=total
    if all((OUT/f'cache-{i:05d}.npz').exists() for i in range(0,2048,128)):
        if not (OUT/'pilot-results.json').exists():
            with (OUT/'logs/analysis.log').open('w') as log:
                subprocess.run(['/home/yimingz3/src/allie/.venv/bin/python','-B',ROOT/'search/analyze.py'],stdout=log,stderr=subprocess.STDOUT,check=True)
        state['phase']='pilot analyzed; persistent workbench active' if live else 'pilot analyzed; allocation ended'
    elif not live:state['phase']='pilot stopped; inspect logs before any retry'
    else:state['phase']='pilot pending/running'
    if (OUT/'golden-balanced-v1/results.json').exists():
        state['phase']='balanced golden comparison analyzed; research goal remains active'
    elif (OUT/'golden-balanced-v1/sample.json').exists():
        state['phase']='balanced golden sample frozen; persistent engine active/pending' if live else 'balanced golden sample frozen; allocation ended'
    elif (OUT/'adaptive-repairs-pilot/results.json').exists():
        state['phase']='adaptive MCTS repairs analyzed; persistent engine active/pending' if live else 'adaptive MCTS repairs analyzed; allocation ended'
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    return state['phase']+' '+str([(j['id'],j.get('state')) for j in state['jobs']]),not live

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args();last=None
    lock=(OUT/'accounting.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        message,done=tick()
        if message!=last:print(message,flush=True);last=message
        if done or not args.watch:break
        time.sleep(30)
