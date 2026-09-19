"""Preregistered past-activation correction, no fitting on golden."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .service import ROOT, atomic
from .balanced_eval import digest
from .golden_metrics import summarize


def plan():
    out = ROOT / 'golden-activation-v1'
    out.mkdir(exist_ok=True)
    dev = ROOT / 'aug-activation-adaptation-v1/results.json'
    d = json.loads(dev.read_text())
    names = list(dict.fromkeys([*d['fit_cv_selected'].values(), 'old_cosine_h8']))
    frozen = {}
    for name in names:
        r = d['results'][name]
        assert r['parameters'][0]['fold'] is None
        frozen[name] = r['parameters'][0]['kappa']
    p = dict(parameters=frozen, cv_selections=d['fit_cv_selected'],
             dev_sha256=digest(dev), sample_sha256=digest(ROOT / 'golden-balanced-v1/sample.json'),
             parent_scores_sha256=digest(ROOT / 'golden-temperature-stack-v1/scores.npz'),
             parent_results_sha256=digest(ROOT / 'golden-temperature-stack-v1/results.json'),
             laws_sha256=digest(ROOT / 'training-cm-laws.json'),
             sources={f: digest(Path(__file__).with_name(f)) for f in
                      ('golden_activation.py', 'activation_collect_v2.py', 'features.py', 'header_adaptation.py', 'golden_metrics.py')},
             selection='Both August fit-CV choices plus matched-history old residual control are frozen and all reported. No winner chosen on golden.',
             cost='One extra full-prefix query per position for every candidate, plus past/current head projections. Search nodes remain unchanged. Report added prefill tokens separately.',
             caveat='Reused golden is exploratory; final success requires fresh disjoint games. Model immutable. Past gradient uses only earlier own targets; root target never enters inference correction.')
    path = out / 'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == p
    else:
        atomic(path, p)
    return p


def run(oracle, spec):
    from .activation_collect_v2 import run as collect
    p = plan()
    out = ROOT / 'golden-activation-v1'
    worker = collect(oracle, dict(sample='golden-balanced-v1/sample.json', output='golden-activation-v1/features'))
    start = time.monotonic()
    rows = json.loads((ROOT / 'golden-balanced-v1/sample.json').read_text())['positions']
    parent = ROOT / 'golden-temperature-stack-v1'
    with np.load(parent / 'scores.npz') as f:
        np.testing.assert_array_equal(f['game'], [r['game'] for r in rows])
        np.testing.assert_array_equal(f['ply'], [r['ply'] for r in rows])
        names = list(f['names'])
        scores = {k: f['scores'][names.index(k)] for k in ('port_raw', 'new_legal')}
        scores['parent'] = f['scores'][names.index('temperature')]
        values, offsets, nodes = f['policy'], f['policy_offsets'], f['nodes']
    blocks = {k: [] for k in p['parameters']}
    policies = {k: [] for k in p['parameters']}
    for path in sorted((out / 'features').glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo, hi = int(path.stem), int(path.stem) + len(f['game'])
            part = rows[lo:hi]
            np.testing.assert_array_equal(f['game'], [r['game'] for r in part])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in part])
            logits, old = f['logits'], f['old']
        n, k = logits.shape[-2:]
        mask = np.arange(k)[None, :] < np.array([len(r['legal']) for r in part])[:, None]
        base = np.zeros((n, k)); base[mask] = values[offsets[lo]:offsets[hi]]
        np.testing.assert_allclose(base.sum(1), 1., atol=1e-14)
        target = np.array([r['legal'].index(r['target']) for r in part])
        np.testing.assert_allclose(-np.log(base[np.arange(n), target]), scores['parent'][lo:hi, 0], atol=1e-14)
        for name, kappa in p['parameters'].items():
            if name == 'old_cosine_h8':
                ratio = old[0]
            else:
                history = int(name.split('_')[1][1:]); step = float(name.split('step')[1])
                h, s = [8, 32].index(history), [0., 1., 4., 16., 64.].index(step)
                ratio = logits[h, s] - logits[h, 0]
            z = np.where(mask, np.log(np.maximum(base, 1e-300)) + kappa * ratio, -np.inf)
            prob = np.exp(z - logsumexp(z, axis=1, keepdims=True))
            assert np.isfinite(prob).all() and (prob[mask] > 0).all()
            move_ids = np.full((n, k), 100000, int)
            for i, r in enumerate(part): move_ids[i, :len(r['legal'])] = r['legal']
            chosen = np.where(prob == prob.max(1)[:, None], move_ids, 100000).min(1)
            blocks[name].append(np.stack([-np.log(prob[np.arange(n), target]), chosen == np.array([r['target'] for r in part]), prob.max(1)], 1))
            policies[name].append(prob[mask])
    for name in blocks: scores[name] = np.concatenate(blocks[name])
    costs = {k: nodes + 1 if k in blocks else nodes if k == 'parent' else np.zeros(len(rows)) for k in scores}
    result = summarize(rows, scores, costs, references=('new_legal', 'parent', 'old_cosine_h8'))
    reference = json.loads((parent / 'results.json').read_text())['methods']['temperature']
    for k in ('macro', 'expert_macro', 'macro_training_eq_cm', 'expert_macro_training_eq_cm'):
        np.testing.assert_allclose(result['parent'][k], reference[k], atol=1e-12, rtol=0)
    atomic(out / 'results.json', dict(methods=result, positions=len(rows), plan_sha256=digest(out / 'plan.json'),
           worker=worker, analysis_seconds=time.monotonic()-start, extra_prefill_tokens=sum(b['prefill_tokens'] for b in worker['blocks']),
           caveat=p['caveat']))
    np.savez_compressed(out / 'scores.npz', names=list(scores), scores=np.stack(list(scores.values())), nodes=nodes,
                        policy_names=list(policies), policies=np.stack([np.concatenate(v) for v in policies.values()]),
                        policy_offsets=offsets, game=[r['game'] for r in rows], ply=[r['ply'] for r in rows])
    return {name: {k: r[k] for k in ('macro', 'expert_macro', 'macro_training_eq_cm', 'expert_macro_training_eq_cm', 'mean_nodes')} for name, r in result.items()}


if __name__ == '__main__':
    plan()
