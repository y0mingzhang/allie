"""Does a nonlinear behavioral utility improve on expected win-minus-loss?"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from .balanced_eval import ROOT, atomic, digest
from .analyze_august import means
from .innovation import fit


def main():
    start = time.monotonic(); out = ROOT / 'aug-selection-v1'
    rows = json.loads((ROOT / 'aug-tune-v1/sample.json').read_text())['positions']
    n = len(rows); ar = np.arange(n)
    cells = np.array([r['cell'] for r in rows]); games = np.array([r['game'] for r in rows])
    fm = np.array([r['fold'] == 0 for r in rows])
    cv = np.array([int(hashlib.sha256(('cv:' + g).encode()).hexdigest(), 16) % 3 for g in games])
    k = max(len(r['legal']) for r in rows)
    mask = np.zeros((n, k), bool); ids = np.zeros((n, k), int); target = np.zeros(n, int)
    for i, r in enumerate(rows):
        mask[i, :len(r['legal'])] = True; ids[i, :len(r['legal'])] = np.array(r['legal']) - 378
        target[i] = r['legal'].index(r['target'])
    root = np.zeros((n, 2432)); q = np.zeros((n, k)); nodes = np.zeros(n)
    for lo in range(0, n, 512):
        with np.load(out / f'zero_cp25/{lo:06d}.npz') as f:
            hi = lo + len(f['game']); assert list(f['game']) == list(games[lo:hi])
            root[lo:hi] = f['z']; q[lo:hi, :f['q'].shape[-1]] = f['q'][-1]; nodes[lo:hi] = f['evaluated_nodes'][-1]
    logits = np.where(mask, root[:, 378:2346][ar[:, None], ids], 0.)
    p = softmax(np.where(mask, logits, -np.inf), axis=1)
    mu = (p * q).sum(1, keepdims=True)
    sd = np.sqrt((p * (q-mu)**2).sum(1, keepdims=True))
    ranks = np.zeros_like(q)
    for i in range(n):
        # Policy-weighted mid-CDF: equal Q values get equal ranks, independent
        # of move ordering. There is no target move in this computation.
        ranks[i] = (((q[i, :, None] > q[i, None, :]) + .5 * (q[i, :, None] == q[i, None, :])) * p[i, None, :]).sum(1)
    assert np.max(np.abs(q)) <= 1 + 1e-10
    menus = [('linear', q), ('logodds', np.arctanh(.95*q)),
             ('tanh', np.tanh(2*q)), ('cubic', q**3), ('rank', ranks),
             ('standardized05', (q-mu)/np.maximum(sd, .05)),
             ('standardized10', (q-mu)/np.maximum(sd, .10)),
             ('standardized20', (q-mu)/np.maximum(sd, .20))]
    records = {}; losses = {}
    for name, utility in menus:
        features = np.stack([logits, utility], 1)
        pred, params = fit(features, mask, target, cells, fm, 0.)
        loss = -pred[ar, target]; oof = np.full(n, np.nan); converged = []
        for fold in range(3):
            val = fm & (cv == fold); v, info = fit(features, mask, target, cells, fm & (cv != fold), 0.)
            oof[val] = -v[ar[val], target[val]]; converged.append(info['converged'])
        a = means(loss[~fm], cells[~fm]); b = means(oof[fm], cells[fm]); losses[name] = loss
        records[name] = dict(parameters=params, cv_converged=converged, training_equivalent_cm=None,
            mean_nodes=float(means(nodes, cells).mean()),
            confirmation=dict(macro_ce=float(a.mean()), expert_ce=float(a[3::4].mean()), cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()), expert_ce=float(b[3::4].mean())))
        print(name, records[name]['fit_game_cv'], a.mean(), a[3::4].mean(), flush=True)
    prior = json.loads((out / 'results.json').read_text())['results']['zero_cp25_1000_elo']['confirmation']
    for key in ('macro_ce', 'expert_ce'):
        np.testing.assert_allclose(records['linear']['confirmation'][key], prior[key], atol=2e-6, rtol=0)
    selected = {metric: min(records, key=lambda k: records[k]['fit_game_cv'][metric]) for metric in ('macro_ce', 'expert_ce')}
    _, ix = np.unique(games[~fm], return_inverse=True); g = ix.max() + 1
    count = np.zeros((g, 16)); np.add.at(count, (ix, cells[~fm]), 1)
    w = np.random.default_rng(8317).multinomial(g, np.full(g, 1/g), size=2000).astype(float)
    den = w @ count; assert (den > 0).all()
    for name in records:
        sums = np.zeros((g, 16)); np.add.at(sums, (ix, cells[~fm]), (losses[name] - losses['linear'])[~fm]); draws = w @ sums / den
        records[name]['confirmation_delta_ci95'] = dict(macro=np.quantile(draws.mean(1), [.025, .975]).tolist(), expert=np.quantile(draws[:, 3::4].mean(1), [.025, .975]).tolist())
    atomic(out / 'utilities.json', dict(results=records, fit_cv_selected=selected, analysis_seconds=time.monotonic()-start,
        source_sha256=digest(Path(__file__)), stage='August potentially training-seen; same cp2.5 tree/prior/node cost. Monotone fixed utility transforms or policy-weighted per-position standardization; all eight arms reported, selection inside fit-game CV. CM pending golden.'))
    print('SELECTED', selected, flush=True)


if __name__ == '__main__': main()
