"""Larger fixed-budget MCTS studies using the tested native tree and cached runner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .adaptive_pilot import traverse
from .service import ROOT, inside, atomic, GLOBAL_STOP


def run(oracle, spec):
    source = ROOT/'dev.json'
    rows = json.loads(source.read_text())['positions']
    out = inside(spec['output'])
    out.mkdir(exist_ok=True)
    mean = int(spec['mean_sims'])
    block = int(spec.get('roots_per_batch', 48))
    assert 1 <= mean <= 16384
    # Includes maximum root prefill length; bounds both token slots and tree rows.
    assert block*(mean+1025) < min(oracle.capacity, oracle.runner.max_total_num_tokens)
    methods = spec.get('methods', ['released_fixed'])
    assert 'released_fixed' in methods and set(methods) <= {'released_fixed', 'fixed_repairs'}
    files = [Path(__file__), *(Path(__file__).with_name(n) for n in (
        'adaptive_pilot.py', 'mcts_native.hpp', 'board.cpp', 'direct.py',
        'policy.py', 'analyze_adaptive.py', 'adaptive_policy.py'))]
    plan = dict(spec=spec, dev_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                export=json.loads((ROOT/'serving-export/provenance.json').read_text()),
                stage='Fixed larger simulation budget on development only; no golden selection.')
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == plan
    else:
        atomic(out/'plan.json', plan)
    start = time.monotonic()
    costs = {method: [] for method in methods}
    for method in methods:
        for lo in range(0, len(rows), block):
            if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():
                raise RuntimeError('STOP requested')
            part = rows[lo:lo+block]
            dest = out/f'{method}-{lo:06d}.npz'
            if not dest.exists():
                oracle.reset()
                begin = time.monotonic()
                z = oracle([r['prefix'] for r in part])
                ns = np.full(len(part), mean, np.int32)
                cp = np.full(len(part), 1.25)
                result, stats = traverse(part, z, oracle, ns, cp,
                                         repairs=method == 'fixed_repairs')
                stats.update(end_to_end_seconds=time.monotonic()-begin,
                             new_tokens=oracle.new_tokens, forward_seconds=oracle.forward_seconds,
                             method=method, lo=lo)
                tmp = dest.with_suffix('.partial')
                with tmp.open('wb') as f:
                    np.savez_compressed(f, **result, root=z, simulations=ns,
                                        game=np.array([r['game'] for r in part]),
                                        ply=np.array([r['ply'] for r in part]), stats=json.dumps(stats))
                tmp.replace(dest)
            with np.load(dest) as f:
                costs[method].append(json.loads(str(f['stats'])))
            print('MCTS larger budget', mean, method, lo+len(part), '/', len(rows), flush=True)
    subprocess.run([sys.executable, '-B', '-m', 'search.engine.analyze_adaptive', str(out)],
                   check=True, env=dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'))
    return dict(stage='Development-only larger-budget MCTS', positions=len(rows), mean_sims=mean,
                elapsed_seconds=time.monotonic()-start,
                methods={m: dict(simulations=mean*len(rows),
                                 evaluated_leaves=sum(c['evaluated_leaves'] for c in costs[m]),
                                 seconds=sum(c['end_to_end_seconds'] for c in costs[m])) for m in methods})
