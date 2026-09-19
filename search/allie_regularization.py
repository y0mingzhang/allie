"""Separate Allie budget allocation from its output regularization schedule.

Grill et al. Eq.4/7: lambda_N = c_N sqrt(N)/(N + |A|), reverse KL.
Use the actual native tree's final c_N (including its logarithmic term).
Report literal scale1 and a dev-calibrated common multiplier. All calibration
comes from the fixed control, so time-dependent arms receive no separate tuning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from .engine.adaptive_policy import output

ROOT = Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    source = Path(args.directory).resolve()
    assert source.is_relative_to(ROOT.resolve())
    start = time.monotonic()
    plan = json.loads((source/'plan.json').read_text())
    rows = json.loads((ROOT/'dev.json').read_text())['positions']
    n = len(rows)
    bs = plan['spec']['roots_per_batch']
    mean = plan['spec']['mean_sims']
    ids = np.zeros((n, max(len(r['legal']) for r in rows)), int)
    legal = np.zeros_like(ids, bool)
    target = np.zeros(n, int)
    for i, row in enumerate(rows):
        ids[i, :len(row['legal'])] = np.array(row['legal'])-378
        legal[i, :len(row['legal'])] = True
        target[i] = row['legal'].index(row['target'])
    ar = np.arange(n)
    fit = np.array([r['fold'] == 0 for r in rows])
    expert = np.array([r['cell'] % 4 == 3 for r in rows])
    trees, hashes = {}, {}
    z = None
    for method in plan['spec']['methods']:
        chunks = []
        for lo in range(0, n, bs):
            path = source/f'{method}-{lo:06d}.npz'
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as f:
                assert list(f['game']) == [r['game'] for r in rows[lo:lo+bs]]
                assert list(f['ply']) == [r['ply'] for r in rows[lo:lo+bs]]
                chunks.append({k: f[k] for k in ('root', 'values', 'simulations')})
        data = {k: np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
        zz = data['root'][:, 378:2346].astype(float)[ar[:, None], ids]
        if z is None:
            z = zz
        else:
            np.testing.assert_array_equal(z, zz)
        ns = data['simulations']
        assert (ns > 0).all()
        cp = 1.25*np.sqrt(mean/ns) if method == 'time_coupled' else np.full(n, 1.25)
        c_n = cp+np.log((ns+19652+1)/19652)
        lam = c_n*np.sqrt(ns)/(ns+legal.sum(1))
        q = data['values'].astype(float)[ar[:, None], ids]
        trees[method] = dict(q=q, lam=lam)
    assert 'released_fixed' in trees
    reference_lambda = (1.25+np.log((mean+19652+1)/19652))*np.sqrt(mean)/(mean+32)
    # Use a fixed reference action count only to parameterize the shared scalar;
    # every actual root's lambda still uses that root's exact legal action count.
    reference = trees['released_fixed']
    scaled_q = reference['q']*(reference_lambda/reference['lam'])[:, None]
    best = (float('inf'), None, None)
    for alpha in (.8, .9, 1., 1.1):
        for beta in (.25, .5, 1., 2., 4., 8., 16., 32.):
            p = output(z[fit], scaled_q[fit], legal[fit], alpha, beta, 'reverse')
            ce = float(-np.log(p[np.arange(fit.sum()), target[fit]]).mean())
            if ce < best[0]:
                best = (ce, alpha, beta)
    records = {}
    for method, tree in trees.items():
        for label, alpha, q, beta in (
            ('literal_grill', 1., tree['q']/tree['lam'][:, None], 1.),
            ('calibrated_grill', best[1], tree['q']*(reference_lambda/tree['lam'])[:, None], best[2]),
        ):
            p = output(z, q, legal, alpha, beta, 'reverse')
            loss = -np.log(p[ar, target])
            correct = p.argmax(1) == target
            assert np.isfinite(loss).all() and np.allclose(p.sum(1), 1.)
            records[method+'_'+label] = dict(metrics={
                name: dict(ce=float(loss[m].mean()), expert_ce=float(loss[m & expert].mean()),
                           accuracy=float(correct[m].mean()), expert_accuracy=float(correct[m & expert].mean()))
                for name, m in [('fit', fit), ('confirmation', ~fit)]})
    report = dict(stage='Development-only reverse-KL schedule ablation; all arms reported',
                  formula='lambda_N = [cpuct + log((N+19653)/19652)] * sqrt(N)/(N+legal_count)',
                  source='https://proceedings.mlr.press/v119/grill20a/grill20a.pdf (Eq.4/7)',
                  reference_lambda=float(reference_lambda), calibration=dict(fit_ce=best[0], alpha=best[1], beta=best[2]),
                  calibration_reference='released_fixed fit games only; same coefficients for every allocation',
                  results=records, elapsed_seconds=time.monotonic()-start, training_equivalent_cm=None,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cache_sha256=hashes,
                  input_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),
                  cm_note='No golden-law conversion for blitz development metrics.')
    dest = source/'regularization-results.json'
    tmp = dest.with_suffix('.partial')
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    tmp.replace(dest)
    for name, record in records.items():
        print(name, record['metrics']['confirmation'], flush=True)


if __name__ == '__main__':
    main()
