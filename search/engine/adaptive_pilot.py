"""Development-only adaptive-MCTS ablation, using the resident cached runner.

Released baseline is retained. Revised variants hold tree exploration fixed,
normalize allocation to an exact simulation total, and calibrate output separately.
No played move, outcome or observed thinking time is read by tree construction.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from .adaptive_policy import allocate
from .native_mcts import reference
from .native_board import MOVES
import _allie_board_v2 as experimental
experimental.initialize(MOVES)
NativeMCTS = experimental.NativeMCTS
from .policy import solve
from .service import ROOT, inside, atomic, GLOBAL_STOP


def traverse(rows, z, oracle, ns, cp, repairs=False):
    start = time.monotonic()
    tree = NativeMCTS([r['prefix'] for r in rows], z, ns.tolist(), cp.tolist())
    tree.first_prior = tree.preserve_depth = repairs
    while not tree.done:
        prefixes = tree.select()
        if prefixes:
            tree.update(oracle(prefixes))
    summaries = tree.summaries()
    policy = solve(summaries, ns, cp)
    values = np.zeros_like(policy)
    visits = np.zeros_like(policy, dtype=np.int32)
    for i, (ids, counts, q, prior) in enumerate(summaries):
        values[i, ids] = q
        visits[i, ids] = counts
    stats = tree.stats()
    del tree
    stats['seconds'] = time.monotonic() - start
    return dict(policy=policy, values=values, visits=visits), stats


def run(oracle, spec):
    source = ROOT / 'dev.json'
    out = inside(spec['output']); out.mkdir(exist_ok=True)
    rows = json.loads(source.read_text())['positions']
    mean = int(spec.get('mean_sims', 64))
    block = int(spec.get('roots_per_batch', 256))
    methods = spec.get('methods', ['released_fixed', 'released_time', 'decoupled_time', 'shuffled_time', 'entropy', 'fixed_repairs', 'time_repairs', 'released_time_repairs'])
    assert set(methods) <= {'released_fixed', 'released_time', 'decoupled_time', 'shuffled_time', 'entropy', 'fixed_repairs', 'time_repairs', 'released_time_repairs'}
    assert 0 < mean <= 1024 and block * mean <= 180000
    files = [Path(__file__), Path(__file__).with_name('adaptive_policy.py'), Path(__file__).with_name('mcts_native.hpp'), Path(__file__).with_name('board.cpp'), Path(__file__).with_name('direct.py'), Path(__file__).with_name('policy.py')]
    plan = dict(spec=spec, dev_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                export=json.loads((ROOT/'serving-export/provenance.json').read_text()),
                stage='Development research. Corrected means decoupled regularization/allocation, not a claim of one uniquely correct MCTS objective.')
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == plan
    else:
        atomic(out/'plan.json', plan)
    start = time.monotonic()
    roots = []
    for lo in range(0, len(rows), block):
        oracle.reset(); roots.append(oracle([r['prefix'] for r in rows[lo:lo+block]]))
    roots = np.concatenate(roots)
    seconds = reference.expected_seconds(roots)
    legal = np.zeros((len(rows),1968), bool)
    for i,r in enumerate(rows):
        legal[i,np.array(r['legal'])-378] = True
    p = softmax(np.where(legal, roots[:,378:2346].astype(float), -np.inf), axis=1)
    entropy = -(p*np.log(np.maximum(p,1e-300))).sum(1)/np.log(legal.sum(1).clip(2))
    rng = np.random.default_rng(774193)
    allocations = dict(released_fixed=np.full(len(rows),mean,np.int32),
                       released_time=reference.budgets(roots,mean),
                       decoupled_time=allocate(seconds,mean),
                       shuffled_time=allocate(seconds[rng.permutation(len(rows))],mean),
                       entropy=allocate(entropy,mean))
    allocations['fixed_repairs'] = allocations['released_fixed']
    allocations['time_repairs'] = allocations['decoupled_time']
    allocations['released_time_repairs'] = allocations['released_time']
    costs = []
    for method in methods:
        ns = allocations[method]
        cp = 1.25*np.sqrt(mean/np.maximum(ns,1)) if method.startswith('released_time') else np.full(len(rows),1.25)
        for lo in range(0,len(rows),block):
            if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():
                raise RuntimeError('STOP requested')
            dest = out/f'{method}-{lo:06d}.npz'
            if not dest.exists():
                part = rows[lo:lo+block]; oracle.reset(); begin = time.monotonic()
                z = oracle([r['prefix'] for r in part])
                np.testing.assert_array_equal(z,roots[lo:lo+block])
                result,stats = traverse(part,z,oracle,ns[lo:lo+block],cp[lo:lo+block], repairs=method.endswith('_repairs'))
                stats.update(end_to_end_seconds=time.monotonic()-begin,new_tokens=oracle.new_tokens,
                             forward_seconds=oracle.forward_seconds,method=method,lo=lo)
                tmp = dest.with_suffix('.partial')
                with tmp.open('wb') as f:
                    np.savez_compressed(f,**result,root=z,simulations=ns[lo:lo+block],
                                        game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]),
                                        stats=json.dumps(stats))
                tmp.replace(dest)
            with np.load(dest) as f:
                costs.append(json.loads(str(f['stats'])))
            print('adaptive pilot',method,lo+len(rows[lo:lo+block]),'/',len(rows),flush=True)
    return dict(stage='Development tree caches; no golden scores, no calibration fit in GPU worker',
                positions=len(rows),mean_sims=mean,elapsed_seconds=time.monotonic()-start,
                methods={m:dict(simulations=int(allocations[m].sum()),
                    evaluated_leaves=sum(c['evaluated_leaves'] for c in costs if c['method']==m),
                    seconds=sum(c['end_to_end_seconds'] for c in costs if c['method']==m)) for m in methods})
