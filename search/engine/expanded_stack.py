"""Stack fitted soft backups and latent search strengths, with nested game folds."""
import ast
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT, atomic
from .balanced_eval import digest
from .sample_data import read
from .diff_backup_native import load
from .fit_policy import fit
from .analyze_expanded import components, learn
from .analyze_player_search import FACTORS
from .analyze_august import means


def main():
    start = time.monotonic()
    out = ROOT / 'aug-expanded-stack-v1'
    out.mkdir(exist_ok=True)
    source = ROOT / 'aug-expanded-search-v1'
    fitted = ROOT / 'aug-expanded-backup-v1'
    log = ROOT / 'logs/expanded-backup-analysis.log'
    prior = json.loads((fitted / 'results.json').read_text())
    plan = dict(
        sample_sha256=digest(ROOT / 'aug-tune-expanded-v1/sample.json'),
        parent_search=digest(source / 'analysis-plan.json'),
        parent_backup=digest(fitted / 'plan.json'),
        backup_results=digest(fitted / 'results.json'),
        fold_log_sha256=digest(log),
        sources={p.name: digest(p) for p in [Path(__file__), *[Path(__file__).with_name(s) for s in
            ('analyze_expanded.py', 'sample_data.py', 'diff_backup.cpp', 'diff_backup_native.py', 'fit_policy.py')]]},
        variants='constant/subtree/joint fitted backup x single/sigma.5/sigma1/learned simplex',
        selection='Backup and output coefficients already fitted separately within each of three game folds; recover those exact coefficients from their hashed execution log. Mixture simplex is fitted on each training fold only. Select on out-of-fold fit CE. August confirmation is potentially training-seen; golden CM pending.',
        cost='All arms use the identical 1000-simulation trees, zero additional neural evaluations.')
    pp = out / 'plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        atomic(pp, plan)
    # The previous analysis saved full fits in JSON, but fold fits only in its log.
    # Parse data literals, bind that log's hash, and verify the full fit independently.
    recovered = {}
    for line in log.read_text().splitlines():
        if line.startswith('Fitted '):
            _, fold, name, data = line.split(' ', 3)
            key = (None if fold == 'None' else int(fold), name)
            assert key not in recovered
            recovered[key] = ast.literal_eval(data)
    assert len(recovered) == 8
    for name in ('fixed_output_2scalar', 'joint_output_10scalar'):
        assert recovered[None, name] == prior['results'][name]['parameters']

    d = read('aug-tune-expanded-v1')
    cells, games, fm, cv = (d[k] for k in ('cells', 'games', 'fit', 'cv'))
    ids, mask, target = (d[k] for k in ('ids', 'mask', 'target'))
    n = len(target)
    ar, group = np.arange(n), cells % 4
    z = np.zeros(mask.shape)
    q = np.zeros((2, *mask.shape))
    nodes = np.zeros(n)
    blocks = []
    module = load()
    for lo in range(0, n, 1024):
        with np.load(source / f'{lo:06d}.npz') as f:
            hi = lo + len(f['game'])
            kk = f['ids'].shape[1]
            np.testing.assert_array_equal(f['game'], games[lo:hi])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in d['rows'][lo:hi]])
            np.testing.assert_array_equal(f['ids'], ids[lo:hi, :kk])
            z[lo:hi] = f['z'][:, 378:2346][np.arange(hi-lo)[:, None], ids[lo:hi]]
            q[:, lo:hi, :kk] = f['q'][-1]
            nodes[lo:hi] = f['evaluated_nodes'][-1]
            payload = {key: f[key] for key in ('parent', 'move', 'depth', 'born', 'degree', 'prior', 'boot', 'mass', 'terminal', 'roots')}
            blocks.append((lo, hi, module.Backup(payload, 1000)))
    z = np.where(mask, z, 0.)
    records, losses, fold_parameters = {}, {}, {}
    for method in ('constant', 'subtree', 'joint'):
        full = {}
        oof = {name: np.full(n, np.nan) for name in ('single', 'sigma05', 'sigma10', 'learned')}
        fold_parameters[method] = []
        for fold, train in [(None, fm), *[(f, fm & (cv != f)) for f in range(3)]]:
            if method == 'joint':
                details = recovered[fold, 'joint_output_10scalar']
                a, b = np.log(details['tau0']), details['count_exponent']
                qq = np.concatenate([obj.reduce(a, b, ids[lo:hi].astype(np.int32))[0] for lo, hi, obj in blocks])
                params = {str(g): dict(alpha=details['alpha'][g], beta=details['beta'][g]) for g in range(4)}
            else:
                qq = q[0 if method == 'constant' else 1]
                details = dict(tau0=.1 if method == 'constant' else .2, count_exponent=0 if method == 'constant' else -.5)
                params = {}
                for g in range(4):
                    tr = train & (group == g)
                    counts = np.bincount(cells[tr], minlength=16)
                    params[str(g)] = fit(z[tr], qq[tr], mask[tr], target[tr], 'forward', 1/counts[cells[tr]])
                    assert params[str(g)]['converged']
            p = components(z, qq, mask, params, group)
            weights = dict(single=np.array([0., 0., 1., 0., 0.]),
                sigma05=softmax(-.5*(np.log(FACTORS)/.5)**2),
                sigma10=softmax(-.5*np.log(FACTORS)**2), learned=learn(p, target, cells, train))
            fold_parameters[method].append(dict(fold=fold, backup=details, parameters=params, weights={k: w.tolist() for k, w in weights.items()}))
            for mixture, w in weights.items():
                probs = np.einsum('a,ank->nk', w, p)
                loss = -np.log(probs[ar, target])
                if fold is None:
                    full[mixture] = loss
                else:
                    val = fm & (cv == fold)
                    oof[mixture][val] = loss[val]
        for mixture, loss in full.items():
            name = f'{method}_{mixture}'
            ce, cc = means(loss[~fm], cells[~fm]), means(oof[mixture][fm], cells[fm])
            f = fold_parameters[method][0]
            records[name] = dict(backup=f['backup'], parameters=f['parameters'], weights=f['weights'][mixture],
                training_equivalent_cm=None, mean_nodes=float(nodes.mean()),
                fit_game_cv=dict(macro_ce=float(cc.mean()), expert_ce=float(cc[3::4].mean())),
                confirmation=dict(macro_ce=float(ce.mean()), expert_ce=float(ce[3::4].mean()), cells=ce.tolist()))
            losses[name] = loss
            if method == 'joint' and mixture == 'single':
                for field in ('confirmation', 'fit_game_cv'):
                    for metric in ('macro_ce', 'expert_ce'):
                        np.testing.assert_allclose(records[name][field][metric], prior['results']['joint_output_10scalar'][field][metric], atol=1e-12, rtol=0)
        print('Stack completed', method, flush=True)
    old = json.loads((source / 'results.json').read_text())
    for method in ('constant', 'subtree'):
        for mixture in ('single', 'sigma05', 'sigma10', 'learned'):
            for field in ('confirmation', 'fit_game_cv'):
                for metric in ('macro_ce', 'expert_ce'):
                    np.testing.assert_allclose(records[f'{method}_{mixture}'][field][metric],
                        old['results'][f'{method}1000_{mixture}'][field][metric], atol=1e-12, rtol=0)
    _, ix = np.unique(games[~fm], return_inverse=True)
    ng = ix.max()+1
    count = np.zeros((ng, 16))
    np.add.at(count, (ix, cells[~fm]), 1)
    w = np.random.default_rng(98318).multinomial(ng, np.full(ng, 1/ng), size=2000).astype(float)
    den = w @ count
    assert (den > 0).all()
    for name, rec in records.items():
        for ref in ('constant_single', 'subtree_sigma05', 'subtree_sigma10'):
            delta = losses[name]-losses[ref]
            point = means(delta[~fm], cells[~fm])
            sums = np.zeros((ng, 16))
            np.add.at(sums, (ix, cells[~fm]), delta[~fm])
            draw = w @ sums / den
            rec['delta_vs_'+ref] = dict(macro=float(point.mean()), expert=float(point[3::4].mean()),
                macro_ci95=np.quantile(draw.mean(1), [.025, .975]).tolist(), expert_ci95=np.quantile(draw[:, 3::4].mean(1), [.025, .975]).tolist())
    selected = {metric: min(records, key=lambda name: records[name]['fit_game_cv'][metric]) for metric in ('macro_ce', 'expert_ce')}
    atomic(out/'results.json', dict(stage=plan['selection'], results=records, fit_cv_selected=selected,
        fold_parameters=fold_parameters, seconds=time.monotonic()-start, plan_sha256=digest(pp)))
    with (out/'scores.npz').open('wb') as f:
        np.savez_compressed(f, names=list(losses), loss=np.stack(list(losses.values())), cells=cells, games=games, fit=fm)
    for name, r in records.items():
        print(name, r['confirmation']['macro_ce'], r['confirmation']['expert_ce'], r['fit_game_cv'], flush=True)
    print('SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
