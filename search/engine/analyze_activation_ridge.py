"""Fold-fitted activation-correction screen; immutable model and separate confirmation."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_august import means
from .analyze_header_adaptation import tilted


def centered_correlation(a, b, mask):
    count = mask.sum(1, keepdims=True)
    a = np.where(mask, a - (a * mask).sum(1, keepdims=True) / count, 0.)[mask]
    b = np.where(mask, b - (b * mask).sum(1, keepdims=True) / count, 0.)[mask]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else None


def main():
    start = time.monotonic()
    out = ROOT / 'aug-activation-ridge-v1'
    worker = json.loads((out / 'worker.json').read_text())
    source_plan = json.loads((out / 'plan.json').read_text())
    assert worker['plan_sha256'] == digest(out / 'plan.json')
    d = read('aug-tune-expanded-v1')
    rows, cells, games, fm, cv, mask, target = (
        d[k] for k in ('rows', 'cells', 'games', 'fit', 'cv', 'mask', 'target'))
    n, k = mask.shape
    ar = np.arange(n)
    hs, ss = source_plan['histories'], source_plan['steps']
    logits = np.zeros((len(hs), len(ss), n, k))
    old = np.zeros((len(hs), n, k))
    same = np.zeros_like(old)
    perturb = np.zeros((len(hs), n))
    counts = np.zeros(n, int)
    covered = np.zeros(n, bool)
    for path in sorted(out.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo, hi = int(path.stem), int(path.stem) + len(f['game'])
            assert not covered[lo:hi].any()
            covered[lo:hi] = True
            np.testing.assert_array_equal(f['game'], games[lo:hi])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in rows[lo:hi]])
            width = f['logits'].shape[-1]
            logits[:, :, lo:hi, :width] = f['logits']
            old[:, lo:hi, :width] = f['old']
            same[:, lo:hi, :width] = f['same']
            perturb[:, lo:hi] = f['perturb_rms']
            counts[lo:hi] = f['count']
    assert covered.all() and np.isfinite(logits).all()
    np.testing.assert_array_equal(logits[0, 0], logits[1, 0])
    assert np.array_equal(logits[:, :, counts == 0],
                          np.broadcast_to(logits[:, :1, counts == 0], logits[:, :, counts == 0].shape))
    surface = ROOT / 'aug-budget-surface-v2/policies.npz'
    with np.load(surface) as f:
        np.testing.assert_array_equal(f['games'], games)
        np.testing.assert_array_equal(f['target'], target)
        np.testing.assert_array_equal(f['mask'], mask)
        parents = f['policy'][-1].astype(float)
        nodes = f['nodes'][-1]
    features = {'parent': np.zeros((n, k))}
    correlations = {}
    for h, history in enumerate(hs):
        for s, step in enumerate(ss[1:], 1):
            feature = np.where(mask, logits[h, s] - logits[h, 0], 0.)
            name = f'activation_h{history}_step{step:g}'
            features[name] = feature
            correlations[name] = {
                'old_cosine_residual': centered_correlation(feature[fm], old[h, fm], mask[fm]),
                'same_kernel_residual': centered_correlation(feature[fm], same[h, fm], mask[fm]),
            }
        if h == 0:
            features['old_cosine_h8'] = old[h]
            features['same_kernel_h8'] = same[h]
    plan = dict(source_sha256=digest(Path(__file__)),
                worker_sha256=digest(out / 'worker.json'),
                surface_sha256=digest(surface), menus=list(features),
                inference='Frozen parent times exp(kappa * paired corrected-minus-zero head logits). No neural weights change. Only strictly past own targets enter activation correction.',
                fit='Nonnegative scalar kappa in [0,2], separately refit on full fit games and three training-game folds. Select method using out-of-fold macro/expert CE, never confirmation.',
                uncertainty='2000 paired whole-game bootstrap draws over separate August confirmation; all arms reported. August may have been training-seen. Golden CM not applied to this split.',
                cost='Each non-parent arm charges one additional root prefill; count tokens and head calculations separately from search nodes.')
    pp = out / 'analysis-plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        atomic(pp, plan)
    records, losses = {}, {}
    for name, ratio in features.items():
        oof = np.full(n, np.nan)
        params = []
        for vi, (fold, train) in enumerate([(None, fm), *[(f, fm & (cv != f)) for f in range(3)]]):
            p = parents[vi] / parents[vi].sum(1, keepdims=True)
            base = np.where(mask, np.log(np.maximum(p, 1e-300)), 0.)
            ct = np.bincount(cells[train], minlength=16)
            assert (ct > 0).all()
            w = 1 / ct[cells[train]]
            w /= w.sum()
            objective = lambda a: tilted(a, base[train], ratio[train], mask[train], target[train], w)
            kappa = 0.
            if name != 'parent':
                opt = minimize_scalar(objective, bounds=(0., 2.), method='bounded', options=dict(xatol=1e-7))
                assert opt.success
                kappa = min((0., float(opt.x), 2.), key=objective)
            prob = tilted(kappa, base, ratio, mask, target)
            assert np.isfinite(prob).all() and (prob[mask] > 0).all()
            ll = -np.log(prob[ar, target])
            params.append(dict(fold=fold, kappa=kappa))
            if fold is None:
                losses[name] = ll
            else:
                oof[fm & (cv == fold)] = ll[fm & (cv == fold)]
        ce, cc = means(losses[name][~fm], cells[~fm]), means(oof[fm], cells[fm])
        records[name] = dict(parameters=params, training_equivalent_cm=None,
                             mean_nodes=float(nodes.mean()) + int(name != 'parent'),
                             extra_full_prefix_queries=int(name != 'parent'),
                             mean_extra_prefill_tokens=float(np.mean([len(r['prefix']) for r in rows])) * int(name != 'parent'),
                             confirmation=dict(macro_ce=float(ce.mean()), expert_ce=float(ce[3::4].mean()), cells=ce.tolist()),
                             fit_game_cv=dict(macro_ce=float(cc.mean()), expert_ce=float(cc[3::4].mean())))
    selected = {m: min(records, key=lambda name: records[name]['fit_game_cv'][m]) for m in ('macro_ce', 'expert_ce')}
    _, ix = np.unique(games[~fm], return_inverse=True)
    ng = ix.max() + 1
    ct = np.zeros((ng, 16))
    np.add.at(ct, (ix, cells[~fm]), 1)
    draws = np.random.default_rng(41912).multinomial(ng, np.full(ng, 1 / ng), size=2000).astype(float)
    den = draws @ ct
    assert (den > 0).all()
    for name, rec in records.items():
        delta = losses[name] - losses['parent']
        point = means(delta[~fm], cells[~fm])
        sums = np.zeros((ng, 16))
        np.add.at(sums, (ix, cells[~fm]), delta[~fm])
        boot = draws @ sums / den
        rec['delta_vs_parent'] = dict(macro=float(point.mean()), expert=float(point[3::4].mean()),
                                      macro_ci95=np.quantile(boot.mean(1), [.025, .975]).tolist(),
                                      expert_ci95=np.quantile(boot[:, 3::4].mean(1), [.025, .975]).tolist())
    atomic(out / 'results.json', dict(results=records, fit_cv_selected=selected,
           correlations_fit_only=correlations, perturb_rms_quantiles=np.quantile(perturb, [0, .5, .9, .99, 1], axis=1).tolist(),
           analysis_seconds=time.monotonic()-start, analysis_plan_sha256=digest(pp), worker=worker))
    np.savez_compressed(out / 'scores.npz', names=list(losses), loss=np.stack(list(losses.values())), cells=cells, games=games, fit=fm)
    print(json.dumps(dict(selected=selected, results={k: records[k] for k in set(selected.values()) | {'parent'}}, correlations=correlations), indent=2), flush=True)


if __name__ == '__main__':
    main()
