"""Calibrate adaptive expectation on dev fold0, report every arm on fold1."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2] / 'results/search-v1'


def main(directory=None):
    if directory is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('directory')
        directory = parser.parse_args().directory
    out = Path(directory).resolve()
    assert out.is_relative_to(ROOT.resolve())
    plan = json.loads((out / 'plan.json').read_text())
    dev = ROOT / 'dev.json'
    assert hashlib.sha256(dev.read_bytes()).hexdigest() == plan['dev_sha256']
    rows = json.loads(dev.read_text())['positions']
    bs = plan['spec'].get('roots_per_batch', 64)
    n = len(rows)
    roots, values, costs = [], [], []
    for lo in range(0, n, bs):
        with np.load(out / f'{lo:06d}.npz') as block:
            part = rows[lo:lo+bs]
            assert list(block['game']) == [r['game'] for r in part]
            assert list(block['ply']) == [r['ply'] for r in part]
            roots.append(block['root'])
            values.append(block['q'])
            costs.append(json.loads(str(block['stats'])))
    z = np.concatenate(roots)[:, 378:2346].astype(float)
    q = np.concatenate(values, axis=1).astype(float)
    ids = np.zeros((n, max(len(r['legal']) for r in rows)), int)
    mask = np.zeros_like(ids, bool)
    y = np.zeros(n, int)
    for i, row in enumerate(rows):
        ids[i, :len(row['legal'])] = np.array(row['legal']) - 378
        mask[i, :len(row['legal'])] = True
        y[i] = row['legal'].index(row['target'])
    ar = np.arange(n)
    z = z[ar[:, None], ids]
    q = q[:, ar[:, None], ids]
    assert np.isfinite(q[:, mask]).all()
    q = np.nan_to_num(q)
    fit = np.array([r['fold'] == 0 for r in rows])
    expert = np.array([r['cell'] % 4 == 3 for r in rows])
    assert set(r['game'] for r in rows if r['fold'] == 0).isdisjoint(
        r['game'] for r in rows if r['fold'] == 1)

    def calibrate(x, bounds):
        xx, mm, yy = x[fit], mask[fit], y[fit]
        aa = np.arange(len(yy))
        def objective(w):
            logits = np.where(mm, xx @ w, -np.inf)
            normalizer = logsumexp(logits, axis=1)
            p = np.exp(logits - normalizer[:, None])
            return float((normalizer-logits[aa, yy]).mean()), (
                np.einsum('na,nak->k', p, xx)/len(yy) - xx[aa, yy].mean(0))
        result = minimize(objective, [1., *([0.]*(x.shape[-1]-1))], jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options=dict(ftol=1e-12, gtol=1e-8))
        assert result.success, result.message
        return result.x

    records, losses = {}, {}
    def score(name, features, weights):
        logits = np.where(mask, features @ weights, -np.inf)
        loss = logsumexp(logits, axis=1) - logits[ar, y]
        correct = logits.argmax(1) == y
        losses[name] = loss
        records[name] = dict(coefficients=list(map(float, weights)), metrics={
            label: dict(ce=float(loss[m].mean()), expert_ce=float(loss[m & expert].mean()),
                        accuracy=float(correct[m].mean()),
                        expert_accuracy=float(correct[m & expert].mean()))
            for label, m in [('fit', fit), ('confirmation', ~fit)]})

    score('legal', z[:, :, None], [1.])
    x = z[:, :, None]
    score('temperature', x, calibrate(x, [(.5, 2.)]))
    for i, name in enumerate(('one_ply', 'adaptive')):
        x = np.stack([z, q[i]], axis=-1)
        score(name, x, calibrate(x, [(.5, 2.), (0., 32.)]))
    # Same predeclared disagreement correction as the fixed-depth extension.
    x = np.stack([z, q[1], q[1]-q[0]], axis=-1)
    score('adaptive_plus_disagreement', x,
          calibrate(x, [(.5, 2.), (0., 32.), (-32., 32.)]))
    games = np.array([r['game'] for r in rows])[~fit]
    unique, ix = np.unique(games, return_inverse=True)
    weights = np.random.default_rng(196921).multinomial(
        len(unique), np.full(len(unique), 1/len(unique)), size=2000)
    for name, record in records.items():
        record['paired_delta_vs_legal'] = {}
        for label, population in [('overall', np.ones(n, bool)), ('expert', expert)]:
            selected = population[~fit]
            delta = (losses[name]-losses['legal'])[~fit]
            numer = np.bincount(ix, weights=delta*selected, minlength=len(unique))
            denom = np.bincount(ix, weights=selected, minlength=len(unique))
            draws = (weights @ numer)/(weights @ denom)
            record['paired_delta_vs_legal'][label] = dict(
                delta=float(delta[selected].mean()), ci95=np.quantile(draws, [.025, .975]).tolist())
    result = dict(stage='Development-only adaptive expectation; all preset arms reported',
                  priority=plan['spec']['priority'], positions=int((~fit).sum()),
                  expert_positions=int((~fit & expert).sum()), results=records,
                  cost=dict(seconds=sum(c['seconds'] for c in costs),
                            forward_seconds=sum(c['forward_seconds'] for c in costs),
                            new_tokens=sum(c['new_tokens'] for c in costs),
                            evaluated_leaves=sum(c['evaluated_leaves'] for c in costs)),
                  plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  training_equivalent_cm=None,
                  cm_note='Golden laws do not apply to blitz development losses.')
    tmp = out/'results.partial'
    tmp.write_text(json.dumps(result, indent=2)+'\n')
    tmp.replace(out/'results.json')
    for name, record in records.items():
        print(plan['spec']['priority'], name, record['metrics']['confirmation'], flush=True)


if __name__ == '__main__':
    main()
