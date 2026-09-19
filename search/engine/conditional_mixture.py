"""Condition latent search strength on causal state features, not just one beta."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_expanded import components
from .analyze_player_search import FACTORS
from .analyze_august import means


def objective(theta, x, pt, weights, ridge):
    # Gaussian prior on log strength, shifted by mu(x), unit log-width.
    # Its -mu^2/2 term cancels from the normalization.
    f = np.log(FACTORS)
    w = softmax(-.5*f[None, :]**2+(x@theta)[:, None]*f, axis=1)
    probability = np.sum(w*pt, axis=1)
    posterior = w*pt/probability[:, None]
    loss = -weights@np.log(probability)+.5*ridge*np.square(theta[1:]).sum()
    grad = x.T@(weights*np.sum((w-posterior)*f[None, :], axis=1))
    grad[1:] += ridge*theta[1:]
    return float(loss), grad


def test():
    rng = np.random.default_rng(641)
    x = np.c_[np.ones(31), rng.normal(size=(31, 4))]
    pt = rng.uniform(.001, .5, size=(31, 5))
    theta = rng.normal(0, .2, size=5)
    w = rng.uniform(.1, 1, size=31); w /= w.sum()
    _, grad = objective(theta, x, pt, w, .01)
    finite = []
    for j in range(5):
        a, b = theta.copy(), theta.copy(); a[j] += 1e-6; b[j] -= 1e-6
        finite.append((objective(a, x, pt, w, .01)[0]-objective(b, x, pt, w, .01)[0])/2e-6)
    np.testing.assert_allclose(grad, finite, atol=1e-9, rtol=1e-6)
    original = softmax(-.5*np.log(FACTORS)**2)
    got, _ = objective(np.zeros(5), x, pt, w, 0.)
    np.testing.assert_allclose(got, -w@np.log(pt@original), atol=1e-15)
    print('PASS mixture gradient and exact zero-gate fixed-mixture limit', flush=True)


def main():
    test(); start = time.monotonic()
    out, source = ROOT/'aug-conditional-mixture-v1', ROOT/'aug-expanded-search-v1'
    out.mkdir(exist_ok=True)
    d = read('aug-tune-expanded-v1')
    rows, cells, games, fm, cv, ids, mask, target = (d[k] for k in ('rows', 'cells', 'games', 'fit', 'cv', 'ids', 'mask', 'target'))
    n, ar = len(rows), np.arange(len(rows))
    stack_file = ROOT/'aug-expanded-stack-v1/results.json'
    stack = json.loads(stack_file.read_text())
    inventory = ROOT/'aug-tune-v1'
    manifest = json.loads((inventory/'manifest.json').read_text())
    for name in ('strat.npz', 'feats.npz'):
        assert digest(inventory/name) == manifest['files_sha256'][name]
    with np.load(inventory/'strat.npz') as f:
        tokens, labels = f['rows'], f['labels']
    with np.load(inventory/'feats.npz') as f:
        feats = f['feats']
    seconds = []
    for r in rows:
        rr, cc = r['row'], r['column']
        assert tokens[rr, cc] == r['target'] and labels[rr, cc] == r['cell']
        np.testing.assert_array_equal(tokens[rr, cc-len(r['prefix']):cc], r['prefix'])
        seconds.append(feats[rr, cc-1, 0])
    seconds = np.array(seconds)
    menus = [('fixed', [], None), ('intercept', [], .0)]
    for name, fields in [('state', [0, 1, 2, 3]), ('clock', [4, 5, 6]), ('all', list(range(7)))]:
        for ridge in (.001, .01):
            menus.append((name+str(ridge), fields, ridge))
    plan = dict(sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'), stack_sha256=digest(stack_file),
        source_sha256=digest(Path(__file__)), sidecar_sha256=manifest['files_sha256'], menus=menus,
        formula='pi=sum_j softmax_j(-log(f_j)^2/2 + mu(x)*log(f_j))*pi_j; f=[.25,.5,1,2,4], mu=intercept+standardized causal features. Frozen subtree_sigma10 component policies, no neural-weight updates.',
        fields=['prior_entropy', 'log_prior_weighted_Q_std', 'ply', 'predicted_log_think_time', 'log_pre_move_clock', 'clock_le15', 'clock_missing'],
        selection='Gate coefficients and feature normalizers refitted inside3 game folds; frozen subtree output coefficients independently refitted on same folds. All arms on disjoint August confirmation. Adds legitimate clock information only in clock/all arms; no actual target thinking time or target outcome.',
        cost='No new NN calls; output gate only. Small calibration on potentially training-seen August; golden CM pending.')
    plan = json.loads(json.dumps(plan))
    pp = out/'plan.json'
    if pp.exists(): assert json.loads(pp.read_text()) == plan
    else: atomic(pp, plan)
    root, q, nodes = np.zeros((n, 2432)), np.zeros(mask.shape), np.zeros(n)
    for lo in range(0, n, 1024):
        with np.load(source/f'{lo:06d}.npz') as f:
            hi = lo+len(f['game']); kk = f['ids'].shape[1]
            np.testing.assert_array_equal(f['game'], games[lo:hi])
            root[lo:hi] = f['z']; q[lo:hi, :kk] = f['q'][-1, 1]; nodes[lo:hi] = f['evaluated_nodes'][-1]
    z = np.where(mask, root[:, 378:2346][ar[:, None], ids], 0.)
    prior = softmax(np.where(mask, z, -np.inf), axis=1)
    entropy = -(prior*np.log(np.maximum(prior, 1e-300))).sum(1)
    qmean = (prior*q).sum(1)
    qstd = np.sqrt((prior*(q-qmean[:, None])**2).sum(1))
    tp = softmax(root[:, 2350:2413], axis=1)
    time_centers = np.r_[np.arange(16), 16*np.exp(np.arange(47)/7.06)]
    known = seconds >= 0
    features = np.column_stack([entropy, np.log(.01+qstd), [len(r['prefix'])-11 for r in rows],
        np.log1p(tp@time_centers), np.where(known, np.log1p(np.maximum(seconds, 0)), 0.),
        known & (seconds <= 15), ~known])
    versions = []
    for entry in stack['fold_parameters']['subtree']:
        versions.append(components(z, q, mask, entry['parameters'], cells%4))
    records, losses = {}, {}
    for name, fields, ridge in menus:
        parameters, oof = [], np.full(n, np.nan)
        for version, (fold, train) in enumerate([(None, fm), *[(f, fm & (cv!=f)) for f in range(3)]]):
            features_ = features[:, fields]
            mean = features_[train].mean(0)
            scale = np.maximum(features_[train].std(0), 1e-6)
            x = np.c_[np.ones(n), np.clip((features_-mean)/scale, -3, 3)]
            p = versions[version]
            pt = p[:, ar, target].T
            count = np.bincount(cells[train], minlength=16)
            w = 1/count[cells[train]]; w /= w.sum()
            theta = np.zeros(x.shape[1])
            if ridge is not None:
                opt = minimize(lambda t: objective(t, x[train], pt[train], w, ridge), theta, jac=True,
                    method='L-BFGS-B', bounds=[(-2, 2)]*len(theta), options=dict(ftol=1e-11, gtol=1e-7, maxiter=200))
                assert opt.success, opt
                theta = opt.x
            weights = softmax(-.5*np.log(FACTORS)[None, :]**2+(x@theta)[:, None]*np.log(FACTORS), axis=1)
            loss = -np.log(np.sum(weights*pt, axis=1))
            parameters.append(dict(fold=fold, theta=theta.tolist(), mean=mean.tolist(), scale=scale.tolist(), fields=fields, ridge=ridge))
            if fold is None: losses[name] = loss
            else: oof[fm & (cv==fold)] = loss[fm & (cv==fold)]
        ce, cc = means(losses[name][~fm], cells[~fm]), means(oof[fm], cells[fm])
        records[name] = dict(parameters=parameters, mean_nodes=float(nodes.mean()), training_equivalent_cm=None,
            confirmation=dict(macro_ce=float(ce.mean()), expert_ce=float(ce[3::4].mean()), cells=ce.tolist()),
            fit_game_cv=dict(macro_ce=float(cc.mean()), expert_ce=float(cc[3::4].mean())))
        print(name, records[name]['fit_game_cv'], ce.mean(), ce[3::4].mean(), flush=True)
    for field in ('confirmation', 'fit_game_cv'):
        for metric in ('macro_ce', 'expert_ce'):
            np.testing.assert_allclose(records['fixed'][field][metric], stack['results']['subtree_sigma10'][field][metric], atol=1e-12, rtol=0)
    selected = {m: min(records, key=lambda k: records[k]['fit_game_cv'][m]) for m in ('macro_ce', 'expert_ce')}
    _, ix = np.unique(games[~fm], return_inverse=True); ng = ix.max()+1
    count = np.zeros((ng, 16)); np.add.at(count, (ix, cells[~fm]), 1)
    w = np.random.default_rng(98318).multinomial(ng, np.full(ng, 1/ng), size=2000).astype(float)
    den = w@count; assert (den>0).all()
    for name, rec in records.items():
        delta = losses[name]-losses['fixed']; point = means(delta[~fm], cells[~fm])
        sums = np.zeros((ng, 16)); np.add.at(sums, (ix, cells[~fm]), delta[~fm]); draw = w@sums/den
        rec['delta_vs_fixed'] = dict(macro=float(point.mean()), expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1), [.025, .975]).tolist(), expert_ci95=np.quantile(draw[:, 3::4].mean(1), [.025, .975]).tolist())
    atomic(out/'results.json', dict(stage=plan['selection'], results=records, fit_cv_selected=selected, seconds=time.monotonic()-start, plan_sha256=digest(pp)))
    with (out/'scores.npz').open('wb') as f:
        np.savez_compressed(f, names=list(losses), loss=np.stack(list(losses.values())), cells=cells, games=games, fit=fm)
    print('SELECTED', selected, flush=True)


if __name__ == '__main__': main()
