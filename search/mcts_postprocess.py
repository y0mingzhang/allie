"""CPU-only MCTS output ablations on cached development predictions.

No new model evaluations. Fit on fold0; use game CV inside fold0 for selection;
report all arms on the reused fold1. Never read golden data or its laws.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    start = time.monotonic()
    source = ROOT/'mcts1000-pilot'
    out = ROOT/'mcts1000-postprocess'
    out.mkdir(exist_ok=True)
    rows = json.loads((ROOT/'dev.json').read_text())['positions']
    source_plan = json.loads((source/'plan.json').read_text())
    bs = source_plan['spec']['roots_per_batch']
    n = len(rows)
    chunks, hashes = [], {}
    for lo in range(0, n, bs):
        path = source/f'fixed_repairs-{lo:06d}.npz'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            assert list(f['game']) == [r['game'] for r in rows[lo:lo+bs]]
            assert list(f['ply']) == [r['ply'] for r in rows[lo:lo+bs]]
            chunks.append({k: f[k] for k in ('root', 'values', 'visits')})
    arrays = {k: np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
    width = max(len(r['legal']) for r in rows)
    ids = np.zeros((n, width), int)
    legal = np.zeros_like(ids, bool)
    target = np.zeros(n, int)
    elo = np.zeros(n)
    for i, row in enumerate(rows):
        ids[i, :len(row['legal'])] = np.array(row['legal'])-378
        legal[i, :len(row['legal'])] = True
        target[i] = row['legal'].index(row['target'])
        offset = 3 if (len(row['prefix'])-11) % 2 == 0 else 7
        elo[i] = sum(row['prefix'][offset+j]*10**(3-j) for j in range(4))
    ar = np.arange(n)
    z = arrays['root'][:, 378:2346].astype(float)[ar[:, None], ids]
    q = arrays['values'].astype(float)[ar[:, None], ids]
    visits = arrays['visits'].astype(float)[ar[:, None], ids]
    logp = np.where(legal, z, -np.inf)
    logp -= logsumexp(logp, axis=1, keepdims=True)
    prior = np.exp(logp)
    root_value = softmax(arrays['root'][:, 2413:2416].astype(float), axis=1) @ np.array([1., 0., -1.])
    visit_ratio = np.where(legal, np.log(np.maximum(visits+16*prior, 1e-300))
                           - np.log(visits.sum(1, keepdims=True)+16)
                           - np.where(legal, logp, 0.), 0.)
    f = np.clip((elo-1000)/1600, 0., 1.)[:, None]
    fit = np.array([r['fold'] == 0 for r in rows])
    cells = np.array([r['cell'] % 4 for r in rows])
    expert = cells == 3
    folds = np.array([int(hashlib.sha256(('mcts-output-cv:'+r['game']).encode()).hexdigest()[:8], 16) % 3 for r in rows])
    # Fixed shrinkage strengths and Elo interpolation knots; no golden fitting.
    arms = {
        'temperature': ([z], [(.5, 2.)], False),
        'q': ([z, q], [(.5, 2.), (0., 32.)], False),
        'visit16': ([z, visit_ratio], [(.5, 2.), (0., 2.)], False),
        'q_visit16': ([z, q, visit_ratio], [(.5, 2.), (0., 32.), (-2., 2.)], False),
        'shrink1': ([z, (visits*q+root_value[:, None])/(visits+1)], [(.5, 2.), (0., 32.)], False),
        'shrink16': ([z, (visits*q+16*root_value[:, None])/(visits+16)], [(.5, 2.), (0., 32.)], False),
        'elo_q': ([z, q*(1-f), q*f], [(.5, 2.), (0., 32.), (0., 32.)], False),
        'elo_q_balanced': ([z, q*(1-f), q*f], [(.5, 2.), (0., 32.), (0., 32.)], True),
    }
    plan = dict(source_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),
                cache_sha256=hashes, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                arms=list(arms), elo_knots=[1000, 2600], visit_pseudocount=16,
                selection='Three-way game CV inside fit fold; all confirmation arms reported. No golden access.')
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == plan
    else:
        (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')

    def optimize(x, mask, bounds, balanced):
        xx, ok, yy = x[mask], legal[mask], target[mask]
        weights = np.ones(len(yy))
        if balanced:
            counts = np.bincount(cells[mask], minlength=4)
            assert (counts > 0).all()
            weights = 1/counts[cells[mask]]
        weights /= weights.sum()
        aa = np.arange(len(yy))
        def objective(w):
            logits = np.where(ok, xx@w, -np.inf)
            norm = logsumexp(logits, axis=1)
            probs = np.exp(logits-norm[:, None])
            return float(weights@(norm-logits[aa, yy])), (
                np.einsum('n,na,nak->k', weights, probs, xx)
                - np.einsum('n,nk->k', weights, xx[aa, yy]))
        result = minimize(objective, [1., *([0.]*(x.shape[-1]-1))], jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options=dict(ftol=1e-12, gtol=1e-8, maxiter=300))
        assert result.success, result.message
        return result.x

    def loss(x, w):
        logits = np.where(legal, x@w, -np.inf)
        return logsumexp(logits, axis=1)-logits[ar, target], logits.argmax(1) == target

    results = {}
    for name, (parts, bounds, balanced) in arms.items():
        x = np.stack(parts, axis=-1)
        cv = np.full(n, np.nan)
        for fold in range(3):
            w = optimize(x, fit & (folds != fold), bounds, balanced)
            losses, _ = loss(x, w)
            selected = fit & (folds == fold)
            cv[selected] = losses[selected]
        w = optimize(x, fit, bounds, balanced)
        losses, correct = loss(x, w)
        results[name] = dict(coefficients=w.tolist(), cv_ce=float(cv[fit].mean()),
                            cv_expert_ce=float(cv[fit & expert].mean()),
                            cv_blitz_rating_macro=float(np.mean([cv[fit & (cells == c)].mean() for c in range(4)])),
                            metrics={label: dict(ce=float(losses[m].mean()),
                                                expert_ce=float(losses[m & expert].mean()),
                                                blitz_rating_macro=float(np.mean([losses[m & (cells == c)].mean() for c in range(4)])),
                                                accuracy=float(correct[m].mean()),
                                                expert_accuracy=float(correct[m & expert].mean()))
                                     for label, m in [('fit', fit), ('confirmation', ~fit)]})
    report = dict(stage='Cached development-only MCTS output ablations; no new GPU work or golden access',
                  results=results, positions=int((~fit).sum()), expert_positions=int((~fit & expert).sum()),
                  selected_by_fit_cv_macro=min(results, key=lambda k: results[k]['cv_blitz_rating_macro']),
                  selected_by_fit_cv_expert=min(results, key=lambda k: results[k]['cv_expert_ce']),
                  elapsed_seconds=time.monotonic()-start, training_equivalent_cm=None,
                  cm_note='Blitz development CE has no matching golden scaling law.')
    tmp = out/'results.partial'
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    tmp.replace(out/'results.json')
    for name, item in results.items():
        print(name, item['metrics']['confirmation'], flush=True)


if __name__ == '__main__':
    main()
