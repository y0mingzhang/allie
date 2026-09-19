"""Frozen causal search-strength gates on the exact existing golden search trees."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT, atomic
from .balanced_eval import digest
from .analyze_expanded import components
from .analyze_player_search import FACTORS
from .golden_metrics import summarize

OUT = ROOT/'golden-conditional-mixture-v1'
PARENT = ROOT/'golden-expanded-stack-v2'
GOLD = Path('/data/group_data/dei-group/yimingz3/allie/strat-eval-v1')


def freeze():
    OUT.mkdir(exist_ok=True)
    dev = ROOT/'aug-conditional-mixture-v1/results.json'
    report = json.loads(dev.read_text())
    assert report['fit_cv_selected'] == dict(macro_ce='all0.001', expert_ce='state0.01')
    golden_manifest = json.loads((GOLD/'manifest.json').read_text())
    plan = dict(methods={name: report['results'][name]['parameters'][0] for name in ('state0.01', 'all0.001')},
        dev_sha256=digest(dev), parent_plan_sha256=digest(PARENT/'plan.json'), parent_results_sha256=digest(PARENT/'results.json'),
        sample_sha256=digest(ROOT/'golden-balanced-v1/sample.json'), laws_sha256=digest(ROOT/'training-cm-laws.json'),
        input_blocks={p.name: digest(p) for p in sorted(PARENT.glob('[0-9]*.npz'))},
        golden_sidecars=dict(strat=golden_manifest['sha256'], feats=golden_manifest['feats_sha256']),
        sources={p.name: digest(p) for p in [Path(__file__), *[Path(__file__).with_name(s) for s in
            ('conditional_mixture.py', 'analyze_expanded.py', 'analyze_player_search.py', 'golden_metrics.py')]]},
        selection='Both expanded-August fit-gameCV winners, plus exact fixed-mixture control. No golden fitting or selection. State uses only model predictions and prefix length; all additionally uses actual pre-move clock.',
        missing_clock='If clock is missing, the all arm uses the unchanged fixed mixture (mu0), because August fit contained no missing clocks. State arm never reads actual clock.',
        semantics='Cached live trees from golden-expanded-stack-v2, unchanged970 mean NN nodes. This output-only gate has no effect on expansion and needs no additional model queries. Includes stored full-prefix neural work; CPU analysis timing is not total standalone search latency. Reused golden requires fresh-game confirmation. Conditional anchored-law training equivalence; no multiplicity/law-error correction.')
    pp = OUT/'plan.json'
    if pp.exists(): assert json.loads(pp.read_text()) == plan
    else: atomic(pp, plan)
    return plan


def main():
    plan = freeze(); start = time.monotonic()
    rows = json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    assert digest(GOLD/'strat.npz') == plan['golden_sidecars']['strat']
    assert digest(GOLD/'feats.npz') == plan['golden_sidecars']['feats']
    with np.load(GOLD/'strat.npz') as f: tokens, labels = f['rows'], f['labels']
    with np.load(GOLD/'feats.npz') as f: feats = f['feats']
    seconds = []
    for r in rows:
        rr, cc = r['row'], r['column']
        assert tokens[rr, cc] == r['target'] and labels[rr, cc] == r['cell']
        np.testing.assert_array_equal(tokens[rr, cc-len(r['prefix']):cc], r['prefix'])
        seconds.append(feats[rr, cc-1, 0])
    seconds = np.array(seconds)
    parent_plan = json.loads((PARENT/'plan.json').read_text())
    parent_results = json.loads((PARENT/'results.json').read_text())
    cfg = parent_plan['methods']['subtree_sigma10']
    scores = {k: [] for k in ('port_raw', 'new_legal', 'fixed', *plan['methods'])}
    all_nodes, policies, offsets = [], {k: [] for k in plan['methods']}, []
    for path in sorted(PARENT.glob('[0-9]*.npz')):
        assert digest(path) == plan['input_blocks'][path.name]
        with np.load(path) as f:
            lo = int(path.stem); part = rows[lo:lo+len(f['game'])]
            hi = lo+len(part); n, ar = len(part), np.arange(len(part))
            np.testing.assert_array_equal(f['game'], [r['game'] for r in part])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in part])
            root, ids, mask = f['z'], f['ids'], f['mask']
            q = f['subtree_sigma10_q']
            z = np.where(mask, root[:, 378:2346][ar[:, None], ids].astype(float), 0.)
            p0 = softmax(np.where(mask, z, -np.inf), axis=1)
            qm = (p0*q).sum(1)
            ent = -(p0*np.log(np.maximum(p0, 1e-300))).sum(1)
            std = np.sqrt((p0*(q-qm[:, None])**2).sum(1))
            time_centers = np.r_[np.arange(16), 16*np.exp(np.arange(47)/7.06)]
            pt = softmax(root[:, 2350:2413].astype(float), axis=1)
            sec = seconds[lo:hi]; known = sec >= 0
            feature = np.column_stack([ent, np.log(.01+std), [len(r['prefix'])-11 for r in part],
                np.log1p(pt@time_centers), np.where(known, np.log1p(np.maximum(sec, 0)), 0.), known & (sec <= 15), ~known])
            groups = np.array([r['cell']%4 for r in part])
            comp = components(z, q, mask, cfg['parameters'], groups)
            baseline = np.einsum('a,ank->nk', np.array(cfg['weights']), comp)
            np.testing.assert_allclose(baseline, f['subtree_sigma10_policy'], atol=2e-15, rtol=0)
            target = np.array([r['legal'].index(r['target']) for r in part])
            scores['port_raw'].append(f['port_raw']); scores['new_legal'].append(f['new_legal'])
            scores['fixed'].append(f['subtree_sigma10']); all_nodes.append(f['nodes'])
            for name, params in plan['methods'].items():
                x = np.c_[np.ones(n), np.clip((feature[:, params['fields']]-params['mean'])/params['scale'], -3, 3)]
                mu = x@np.array(params['theta'])
                if name.startswith('all'): mu[~known] = 0.
                weights = softmax(-.5*np.log(FACTORS)[None, :]**2+mu[:, None]*np.log(FACTORS), axis=1)
                p = np.einsum('na,ank->nk', weights, comp)
                np.testing.assert_allclose(p.sum(1), 1., atol=1e-14)
                chosen = np.where(p == p.max(1)[:, None], ids, 1968).min(1)
                scores[name].append(np.stack([-np.log(p[ar, target]), chosen == ids[ar, target], p.max(1)], axis=1))
                policies[name].append(p[mask])
            offsets.extend(mask.sum(1).tolist())
    scores = {k: np.concatenate(v) for k, v in scores.items()}
    nodes = np.concatenate(all_nodes)
    costs = {k: nodes if k not in ('port_raw', 'new_legal') else np.zeros(len(rows)) for k in scores}
    result = summarize(rows, scores, costs, references=('new_legal', 'fixed'))
    for metric in ('macro', 'expert_macro', 'mean_nodes'):
        np.testing.assert_allclose(result['fixed'][metric], parent_results['methods']['subtree_sigma10'][metric], atol=1e-12, rtol=0)
    report = dict(methods=result, positions=len(rows), plan_sha256=digest(OUT/'plan.json'),
        analysis_seconds=time.monotonic()-start, missing_clocks=int((seconds<0).sum()),
        time_trouble_positions=int(((seconds>=0)&(seconds<=15)).sum()), caveat=plan['semantics'])
    atomic(OUT/'results.json', report)
    with (OUT/'scores.npz').open('wb') as f:
        np.savez_compressed(f, names=list(scores), scores=np.stack(list(scores.values())), nodes=nodes,
            game=[r['game'] for r in rows], ply=[r['ply'] for r in rows],
            policy_offsets=np.r_[0, np.cumsum(offsets)], **{k+'_policy': np.concatenate(v) for k, v in policies.items()})
    for name in ('fixed', *plan['methods']):
        print(name, {k: result[name][k] for k in ('macro', 'expert_macro', 'macro_training_eq_cm', 'expert_macro_training_eq_cm', 'mean_nodes')}, flush=True)


if __name__ == '__main__': main()
