"""One resident GPU runner, coarse research tasks, no per-node RPC overhead.

Queue requests and outputs are private durable artifacts. Each request executes
serially; a process lock prevents accidental duplicate GPU runners. Failed tasks
are recorded explicitly. Interrupted tasks can resume immutable completed blocks.
"""
import fcntl
import hashlib
import importlib
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from .direct import DirectOracle

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
QUEUE=ROOT/'engine-queue'
GLOBAL_STOP=Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')


def atomic(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def inside(path):
    path=Path(path).resolve()
    assert path.is_relative_to(ROOT.resolve()),'Task paths must stay in private results'
    return path


def tree(oracle,spec):
    from .tree import batch
    source=inside(spec['input']);out=inside(spec['output']);out.mkdir(exist_ok=True)
    rows=json.loads(source.read_text())['positions'];widths=tuple(spec.get('widths',[4,2,2]))
    bs=spec.get('batch_size',1024);nr=spec.get('roots_per_batch',32)
    plan=spec|dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.rglob('*.py')},
        export=json.loads((ROOT/'serving-export/provenance.json').read_text()))
    path=out/'engine-plan.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Cannot mix different implementations in a cache'
    else:atomic(path,plan)
    start=time.monotonic();cost=[]
    for lo in range(0,len(rows),nr):
        if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():raise RuntimeError('STOP requested')
        dest=out/f'{lo:06d}.npz'
        if not dest.exists():
            block=rows[lo:lo+nr];oracle.reset()
            q,z,stats=batch(block,oracle,widths=widths,batch_size=bs)
            stats.update(new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
            tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,q=q,root=z,stats=json.dumps(stats),
                game=np.array([r['game'] for r in block]),ply=np.array([r['ply'] for r in block]))
            tmp.replace(dest)
        with np.load(dest) as f:cost.append(json.loads(str(f['stats'])))
        if lo%(nr*16)==0:print('tree progress',str(out),lo+len(rows[lo:lo+nr]),'/',len(rows),flush=True)
    return dict(positions=len(rows),elapsed_seconds=time.monotonic()-start,
        summed_block_seconds=sum(c['seconds'] for c in cost),
        new_tokens=sum(c['new_tokens'] for c in cost),blocks=len(cost))


def main():
    QUEUE.mkdir(exist_ok=True)
    with (QUEUE/'service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        oracle=DirectOracle(capacity=262144)
        atomic(QUEUE/'ready.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),
            startup_seconds=oracle.startup_seconds,ready_unix=time.time(),
            process_startup_seconds=time.time()-float(os.environ.get('ALLIE_SERVICE_STARTED',time.time()))))
        print('ready',oracle.startup_seconds,flush=True)
        while not (ROOT/'STOP').exists() and not GLOBAL_STOP.exists():
            for path in sorted(QUEUE.glob('*.request.json')):
                dest=path.with_name(path.name.replace('.request.json','.result.json'))
                error=path.with_name(path.name.replace('.request.json','.error.json'))
                if dest.exists() or error.exists():continue
                try:
                    spec=json.loads(path.read_text());kind=spec['kind']
                    if kind=='benchmark':
                        from .benchmark import run
                        value=run(oracle)
                    elif kind=='mcts_benchmark':
                        from .benchmark_mcts import run_benchmark
                        value=run_benchmark(oracle)
                    elif kind=='experiment':
                        module=spec['module']
                        assert module.startswith('search.engine.') and module.count('.')==2
                        handler=importlib.import_module(module)
                        value=handler.run(oracle,spec)
                    elif kind=='tree':value=tree(oracle,spec)
                    else:raise ValueError(f'Unknown task {kind}')
                    atomic(dest,value)
                except Exception:
                    traceback.print_exc();atomic(error,dict(traceback=traceback.format_exc()))
                oracle.reset()
            time.sleep(.2)


if __name__=='__main__':main()
