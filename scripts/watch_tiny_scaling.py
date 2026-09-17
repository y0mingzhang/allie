"""Bounded read-only Slurm observer; runs CPU analysis once after success.

No retries/submissions/cancellations and no periodic model wakeups.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/yimingz3/src/allie')
BASE = ROOT/'results/recipe10x/tiny-scaling-v1'
OUT = BASE/'monitor'
STOP = Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    temp = p.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(p)


def ids_now():
    return {str(read(p)['job_id']):str(p.parent.relative_to(BASE)) for p in BASE.glob('*/submission.json')} | {
        str(read(p)['job_id']):str(p.parent.relative_to(BASE)) for p in BASE.glob('*/evaluator/submission.json')}


def snapshot():
    ids = ids_now()
    if not ids:
        return dict(jobs={}, submission_complete=False)
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
                                   '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList'],text=True,timeout=45)
    jobs = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        job, state, code, elapsed, tres, nodes, *_ = line.split('|')
        if job in ids:
            jobs[job] = dict(package=ids[job], state=state.split()[0].rstrip('+'), exit_code=code,
                             elapsed_seconds=int(elapsed), alloc_tres=tres, nodes=nodes)
    progress = {}
    for name in read(BASE/'prepared.json')['packages']:
        run = ROOT/'results/pretrain'/('tiny-v1-'+name)
        p = run/'progress.json'
        if p.exists():
            progress[name] = read(p)
        else:
            p = run/'train.jsonl'
            if p.exists():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-16384))
                    for line in reversed(f.read().splitlines()):
                        try:
                            progress[name] = json.loads(line); break
                        except (ValueError, UnicodeDecodeError):
                            continue
    return dict(at=datetime.now(timezone.utc).isoformat(), jobs=jobs, progress=progress,
                submission_complete=(BASE/'submitted.json').exists(),
                missing_accounting=sorted(set(ids)-set(jobs)))


def main():
    OUT.mkdir(exist_ok=True)
    receipt = OUT/'receipt.json'
    if receipt.exists():
        raise RuntimeError('Existing observer receipt: inspect ownership before any restart')
    with receipt.open('x') as f:
        json.dump(dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                       interval_seconds=60, bounded_hours=6, automatic_submissions=False),f,indent=2)
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        if STOP.exists():
            write(OUT/'done.json',dict(reason='controller_STOP',training_untouched=True)); return
        try:
            obs=snapshot();write(OUT/'observation.json',obs)
            failed={j:v for j,v in obs['jobs'].items() if v['state'] in TERMINAL and
                    (v['state']!='COMPLETED' or v['exit_code']!='0:0')}
            if failed:
                write(OUT/'review-required.json',dict(reason='Terminal unsuccessful jobs; no automatic retry',jobs=failed))
            complete=obs['submission_complete'] and len(obs['jobs'])==49 and not obs['missing_accounting']
            if complete and all(v['state']=='COMPLETED' and v['exit_code']=='0:0' for v in obs['jobs'].values()):
                action=OUT/'analysis-action.json'
                if action.exists():
                    raise RuntimeError('Prior analysis action exists; inspect instead of retrying')
                cmd=[str(ROOT/'.venv/bin/python'),str(BASE/'analysis_source.py')]
                write(action,dict(started=datetime.now(timezone.utc).isoformat(),command=cmd))
                with (OUT/'analysis.log').open('w') as log:
                    proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                write(OUT/'done.json',dict(reason='all_completed',analysis_returncode=proc.returncode,
                                          goal_requires_scientific_review=True))
                if proc.returncode:
                    write(OUT/'review-required.json',dict(reason='CPU analysis failed; see analysis.log'))
                # Refresh cumulative allocation history only once at completion.
                subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/budget.py'),'ledger'],
                               stdout=subprocess.DEVNULL,timeout=180,check=True)
                return
        except Exception as exc:
            with (OUT/'errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),error=repr(exc)))+'\n')
        time.sleep(60)
    write(OUT/'done.json',dict(reason='bounded_observer_expired',training_untouched=True))


if __name__=='__main__':
    main()
