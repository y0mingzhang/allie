"""Read-only completion forecast, using live progress and observed step times.

This models job-slot constraints, not Slurm priority/preemption guarantees.
No launches, cancellations or changes to any training run.
"""
import heapq
import json
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/tiny-scaling-v1'


def read(p):
    return json.loads(Path(p).read_text())


def recent(path):
    if not path.exists(): return []
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-131072))
        rows=[]
        for line in f.read().splitlines():
            try: rows.append(json.loads(line))
            except (ValueError,UnicodeDecodeError): pass
    return rows


def main():
    observation=read(BASE/'monitor/observation.json')
    packages=read(BASE/'prepared.json')['packages']
    mapping=read(BASE/'submitted.json')['cells']
    measured={};progress={}
    for name,c in packages.items():
        run=ROOT/'results/pretrain'/('tiny-v1-'+name)
        if c['recipe']=='ours':
            rows=recent(run/'train.jsonl')
            eligible=[r for r in rows if r['step']>=100][-3:]
            times=[524288/r['tokens_per_second'] for r in eligible]
            progress[name]=rows[-1]['step'] if rows else 0
        else:
            rows=recent(run/'timing-rank0.jsonl')
            times=[r['seconds'] for r in rows if r['step']>=10][-50:]
            progress[name]=read(run/'progress.json')['step'] if (run/'progress.json').exists() else 0
        if times:
            key=(c['recipe'],c['layers'],c['width'],c['gpu_type'])
            measured.setdefault(key,[]).append(statistics.median(times))
    rates={k:statistics.median(v) for k,v in measured.items()}
    ratios=[]
    for key,v in rates.items():
        other=key[:-1]+('L40S',)
        if key[-1]=='A6000' and other in rates:ratios.append(v/rates[other])
    a6000_factor=statistics.median(ratios) if ratios else 1.6
    jobs={};details={}
    for name,c in packages.items():
        key=(c['recipe'],c['layers'],c['width'],c['gpu_type'])
        if key in rates: seconds=rates[key];basis='same recipe/shape/hardware observed'
        else:
            other=key[:-1]+('L40S' if key[-1]=='A6000' else 'A6000',)
            assert other in rates, ('No matching shape timing available yet',name)
            seconds=rates[other]*(a6000_factor if key[-1]=='A6000' else 1/a6000_factor)
            basis='other hardware scaled by observed A6000/L40S median ratio'
        for kind,job in mapping[name].items():
            state=observation['jobs'][str(job)]
            if state['state']=='COMPLETED': duration=0.
            elif kind=='train':
                startup=max(0,240-state['elapsed_seconds']) if progress[name]==0 else 0
                duration=(c['steps']-progress[name])*seconds+startup+30
            else:
                duration=max(10,(300 if c['recipe']=='ours' else 90)-state['elapsed_seconds'])
            jobs[job]=dict(name=name,kind=kind,state=state['state'],duration=duration,
                           gpu=c['gpu_type'],depends=mapping[name]['train'] if kind=='evaluate' else None)
        details[name]=dict(step=progress[name],target=c['steps'],seconds_per_step=seconds,basis=basis)
    def simulate(multiplier):
        caps={'A6000':9,'L40S':8};occupied={k:0 for k in caps}
        done={j for j,v in jobs.items() if v['state']=='COMPLETED'};started=set(done);heap=[];ends={}
        for j,v in jobs.items():
            if v['state']=='RUNNING':
                started.add(j);occupied[v['gpu']]+=1
                heapq.heappush(heap,(v['duration']*multiplier,j))
        now=0.
        while len(done)<len(jobs):
            for j,v in sorted(jobs.items()):
                if j in started or v['depends'] is not None and v['depends'] not in done:continue
                if occupied[v['gpu']]>=caps[v['gpu']]:continue
                started.add(j);occupied[v['gpu']]+=1
                heapq.heappush(heap,(now+v['duration']*multiplier,j))
            assert heap,'Cannot schedule remaining jobs'
            now,j=heapq.heappop(heap);done.add(j);occupied[jobs[j]['gpu']]-=1;ends[j]=now
        last=max(ends,key=ends.get)
        return dict(remaining_minutes=now/60,last_job=last,last_cell=jobs[last]['name'],
                    last_kind=jobs[last]['kind'],
                    last_by_hardware_minutes={gpu:max([t for j,t in ends.items() if jobs[j]['gpu']==gpu],default=0)/60 for gpu in caps})
    now=datetime.now(timezone.utc)
    result=dict(at=now.isoformat(),observation_at=observation['at'],
                scheduler_slots=dict(A6000=9,L40S=8),
                median_A6000_to_L40S_step_time_ratio=a6000_factor,
                forecast=simulate(1),all_remaining_durations_50pct_slower=simulate(1.5),cells=details,
                limits=['Observed timing extrapolation, not a Slurm start-time promise.',
                        'Evaluations estimated conservatively: ours300s/Qwen90s; pending startup240s.',
                        'Oldest-job-first approximation; queue priority and other users can change placement.',
                        'Matching-architecture hardware ratio is preferred; unmeasured pairs use pooled observed ratio.'])
    result['forecast']['estimated_finish_utc']=(now+timedelta(minutes=result['forecast']['remaining_minutes'])).isoformat()
    (BASE/'eta.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':main()
