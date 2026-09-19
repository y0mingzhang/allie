"""Own finite-transfer task queue. One model process per requested export."""
import fcntl,importlib,json,os,time,traceback
from pathlib import Path
from search.engine.service import ROOT,GLOBAL_STOP,atomic
from .oracle import ShipOracle


def main():
    queue=ROOT/'transfer-v1/queue';queue.mkdir(exist_ok=True)
    with (queue/'service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        oracle=None;model=None
        atomic(queue/'worker.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),time=time.time()))
        while not GLOBAL_STOP.exists() and not (ROOT/'STOP').exists():
            for p in sorted(queue.glob('*.request.json')):
                out=p.with_name(p.name.replace('.request.json','.result.json'));err=p.with_name(p.name.replace('.request.json','.error.json'))
                if out.exists() or err.exists():continue
                spec=json.loads(p.read_text());name=spec['model'];assert name in ('large','small')
                if oracle is not None and model!=name:
                    # SGLang's globals/graph pools need a fresh process for a new architecture.
                    atomic(queue/'restart.json',dict(next_model=name,time=time.time()))
                    return
                try:
                    if oracle is None:
                        model=name;oracle=ShipOracle(ROOT/'transfer-v1'/('ship-'+name+'-export'),capacity=350000)
                        atomic(queue/'ready.json',dict(model=name,pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),startup_seconds=oracle.startup_seconds,time=time.time()))
                    assert spec['module'].startswith('search.transfer.') and spec['module'].count('.')==2
                    mod=importlib.import_module(spec['module']);atomic(out,mod.run(oracle,spec))
                except Exception:
                    traceback.print_exc();atomic(err,dict(traceback=traceback.format_exc()))
                if oracle is not None:oracle.reset()
            time.sleep(1)


if __name__=='__main__':main()
