"""Cheap pilot observer: account own allocation and analyze once; never submit jobs."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/search-v1'
STOP=Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')

def tick():
    path=OUT/'status.json';state=json.loads(path.read_text())
    if STOP.exists(): return 'controller STOP; observer exiting',True
    total=0.;live=False
    for job in state['jobs']:
        jobid=job['id'].split(';')[0]
        text=subprocess.check_output(['sacct','-j',jobid,'-n','-P','-o','JobIDRaw,State,ElapsedRaw'],text=True)
        entries=[x.split('|') for x in text.splitlines() if x.split('|')[0]==jobid]
        if not entries: live=True;continue
        _,status,seconds=entries[-1][:3]
        job['state']=status;job['elapsed_seconds']=int(seconds)
        job['gpu_hours']=int(seconds)*job['gpus']/3600.;total+=job['gpu_hours']
        live |= status in ('PENDING','RUNNING','CONFIGURING','COMPLETING','REQUEUED','SUSPENDED')
    state['gpu_hours']=total
    if all((OUT/f'cache-{i:05d}.npz').exists() for i in range(0,2048,128)):
        if not (OUT/'pilot-results.json').exists():
            with (OUT/'logs/analysis.log').open('w') as log:
                subprocess.run(['/home/yimingz3/src/allie/.venv/bin/python','-B',ROOT/'search/analyze.py'],stdout=log,stderr=subprocess.STDOUT,check=True)
        state['phase']='pilot analyzed; persistent workbench active' if live else 'pilot analyzed; allocation ended'
    elif not live:state['phase']='pilot stopped; inspect logs before any retry'
    else:state['phase']='pilot pending/running'
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    return state['phase']+' '+str([(j['id'],j.get('state')) for j in state['jobs']]),not live

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args();last=None
    while True:
        message,done=tick()
        if message!=last:print(message,flush=True);last=message
        if done or not args.watch:break
        time.sleep(30)
