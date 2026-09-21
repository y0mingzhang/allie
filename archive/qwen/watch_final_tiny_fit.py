"""Run the frozen combined CPU analysis after both existing reviews succeed.

No Slurm submissions, retries, recipe changes, or goal-completion decisions.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/home/yimingz3/src/allie')
BASE = ROOT/'results/recipe10x/tiny-scaling-v1'
OTHER = BASE.parent/'tiny-size-validation-v1'
OUT = BASE/'extended-analysis/final-monitor'
STOP = Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def ready(bases):
    for base in bases:
        monitor = base/'monitor'
        if (monitor/'review-required.json').exists():
            raise RuntimeError(f'Upstream review required: {base}')
        if not (monitor/'done.json').exists():
            return False
        done = read(monitor/'done.json')
        if done.get('reason') != 'all_completed' or done.get('analysis_returncode') != 0:
            raise RuntimeError(f'Upstream observer ended without successful review: {base}: {done}')
    return True


def main():
    assert not STOP.exists(), 'Controller STOP present'
    OUT.mkdir(exist_ok=True)
    with (OUT/'receipt.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                       bounded_hours=6, interval_seconds=60, automatic_gpu_actions=False), f, indent=2)
    deadline = time.monotonic()+6*3600
    try:
        while time.monotonic() < deadline:
            if STOP.exists():
                write(OUT/'done.json', dict(reason='controller_STOP', training_untouched=True))
                return
            if ready((BASE, OTHER)):
                plan = read(BASE/'extended-analysis/final-analysis-plan.json')
                source = Path(plan['source'])
                assert hashlib.sha256(source.read_bytes()).hexdigest() == plan['source_sha256']
                assert read(OTHER/'validation.json')['data_integrity_passed']
                assert not (BASE/'extended-analysis/full-with-size-validation').exists()
                command = [str(ROOT/'.venv/bin/python'), str(source), '--full', '--include-size-validation']
                with (OUT/'action.json').open('x') as f:
                    json.dump(dict(command=command, started=datetime.now(timezone.utc).isoformat()), f, indent=2)
                with (OUT/'analysis.log').open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                assert result.returncode == 0, 'Combined CPU analysis failed; inspect log, no retry'
                write(OUT/'done.json', dict(reason='analysis_completed', requires_scientific_review=True,
                                          goal_complete=False, gpu_actions=False))
                return
            time.sleep(60)
        write(OUT/'done.json', dict(reason='bounded_wait_expired', training_untouched=True))
    except Exception as exc:
        write(OUT/'review-required.json', dict(error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
