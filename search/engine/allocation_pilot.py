"""Allie budget allocation at exactly matched total simulations.

Reuse the already completed fixed controls, then isolate predicted-time routing,
its exploration coupling, and a shuffled-time negative control. No label enters
allocation. All repaired variants share the same native first-visit/depth fixes.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .adaptive_pilot import traverse
from .adaptive_policy import allocate
from .native_mcts import reference
from .service import ROOT, inside, atomic, GLOBAL_STOP


def run(oracle, spec):
    out = inside(spec['output'])
    out.mkdir(exist_ok=True)
    source = ROOT/'dev.json'
    rows = json.loads(source.read_text())['positions']
    old = ROOT/'mcts1000-pilot'
    oldplan = json.loads((old/'plan.json').read_text())
    mean, block = spec['mean_sims'], spec['roots_per_batch']
    assert mean == oldplan['spec']['mean_sims'] == 1000
    assert block == oldplan['spec']['roots_per_batch'] == 128
    assert oldplan['dev_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    for name, digest in oldplan['sources'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    assert oldplan['export'] == json.loads((ROOT/'serving-export/provenance.json').read_text())
    roots, reuse = [], {}
    for lo in range(0, len(rows), block):
        with np.load(old/f'released_fixed-{lo:06d}.npz') as f:
            assert list(f['game']) == [r['game'] for r in rows[lo:lo+block]]
            assert list(f['ply']) == [r['ply'] for r in rows[lo:lo+block]]
            roots.append(f['root'])
        for name in ('released_fixed', 'fixed_repairs'):
            path = old/f'{name}-{lo:06d}.npz'
            reuse[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    roots = np.concatenate(roots)
    seconds = reference.expected_seconds(roots)
    fixed = np.full(len(rows), mean, np.int32)
    timed = allocate(seconds, mean)
    shuffled = timed[np.random.default_rng(774193).permutation(len(rows))]
    budgets = dict(released_fixed=fixed, fixed_repairs=fixed, time_normalized=timed,
                   time_coupled=timed, time_shuffled=shuffled)
    assert set(spec['methods']) == set(budgets)
    capacity = min(oracle.capacity, oracle.runner.max_total_num_tokens)
    for ns in budgets.values():
        assert ns.sum() == mean*len(rows)
        for lo in range(0, len(rows), block):
            # Upper bound includes every root prefix and every simulation.
            assert ns[lo:lo+block].sum()+sum(len(r['prefix']) for r in rows[lo:lo+block]) < capacity
    plan = dict(spec=spec, dev_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reused_control_plan_sha256=hashlib.sha256((old/'plan.json').read_bytes()).hexdigest(),
                reused_control_sha256=reuse, frozen_dependencies=oldplan['sources'],
                note='Every arm has exactly 1000 simulations/position on average. Fixed controls reused byte for byte. Time budgets use only predictions; permutation retains exactly the same budget histogram.')
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == plan
    else:
        atomic(out/'plan.json', plan)
    for name, digest in reuse.items():
        dest = out/name
        if not dest.exists():
            os.link(old/name, dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    start = time.monotonic()
    costs = {name: [] for name in spec['methods']}
    for name in spec['methods']:
        ns = budgets[name]
        cp = 1.25*np.sqrt(mean/ns) if name == 'time_coupled' else np.full(len(rows), 1.25)
        for lo in range(0, len(rows), block):
            if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():
                raise RuntimeError('STOP requested')
            part = rows[lo:lo+block]
            dest = out/f'{name}-{lo:06d}.npz'
            if not dest.exists():
                begin = time.monotonic()
                oracle.reset()
                z = oracle([r['prefix'] for r in part])
                np.testing.assert_array_equal(z, roots[lo:lo+block])
                result, stats = traverse(part, z, oracle, ns[lo:lo+block], cp[lo:lo+block], repairs=True)
                stats.update(end_to_end_seconds=time.monotonic()-begin, new_tokens=oracle.new_tokens,
                             forward_seconds=oracle.forward_seconds, method=name, lo=lo)
                tmp = dest.with_suffix('.partial')
                with tmp.open('wb') as f:
                    np.savez_compressed(f, **result, root=z, simulations=ns[lo:lo+block],
                                        game=np.array([r['game'] for r in part]),
                                        ply=np.array([r['ply'] for r in part]), stats=json.dumps(stats))
                tmp.replace(dest)
            with np.load(dest) as f:
                costs[name].append(json.loads(str(f['stats'])))
            if name not in ('released_fixed', 'fixed_repairs'):
                print('Allie allocation', name, lo+len(part), '/', len(rows), flush=True)
    subprocess.run([sys.executable, '-B', '-m', 'search.engine.analyze_adaptive', str(out)],
                   check=True, env=dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'))
    return dict(stage='Matched-budget Allie adaptation; development only',
                elapsed_seconds=time.monotonic()-start, reused_controls=list(reuse),
                methods={name: dict(simulations=int(budgets[name].sum()),
                                    evaluated_leaves=sum(c['evaluated_leaves'] for c in costs[name]),
                                    scoring_seconds=sum(c['end_to_end_seconds'] for c in costs[name]))
                         for name in spec['methods']})
