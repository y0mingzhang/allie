"""One restart-safe private research chain; no Slurm submissions or GPU allocation."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from .service import ROOT,QUEUE,GLOBAL_STOP,atomic


def stopped():
    if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP requested')


def main():
    with (ROOT/'august-chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (ROOT/'aug-tune-v1/manifest.json').exists():stopped();time.sleep(10)
        stopped()
        if not (ROOT/'aug-tune-v1/sample.json').exists():
            subprocess.run([sys.executable,'-B','-m','search.engine.prepare_dev_balanced'],check=True)
        # Freeze the analysis menu before any new inference results exist.
        paths=[Path(__file__).with_name(n) for n in ('analyze_august.py','fit_policy.py','distributional_policy.py')]
        spec=dict(kind='experiment',module='search.engine.balanced_dev_pilot',
            input=str(ROOT/'aug-tune-v1/sample.json'),output=str(ROOT/'aug-search-v1'),
            analysis_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        request=QUEUE/'023-august.request.json'
        if request.exists():assert json.loads(request.read_text())==spec
        else:atomic(request,spec)
        while not (QUEUE/'023-august.result.json').exists():
            stopped()
            error=QUEUE/'023-august.error.json'
            if error.exists():raise RuntimeError(error.read_text())
            time.sleep(10)
        for module in ('analyze_august','distributional_policy'):
            stopped()
            for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['analysis_sources'][p.name]
            output=ROOT/'aug-search-v1'/('results.json' if module=='analyze_august' else 'distributional-results.json')
            if not output.exists():subprocess.run([sys.executable,'-B','-m','search.engine.'+module],check=True)
        subprocess.run([sys.executable,'-B','-m','search.report'],check=True)
        print('August research chain complete',flush=True)


if __name__=='__main__':main()
