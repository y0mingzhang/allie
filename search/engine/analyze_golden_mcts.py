"""Same per-cell paired estimator and game bootstrap as the first golden check."""
import json
import time
from pathlib import Path
import numpy as np
from .analyze_balanced import bootstrap_deltas, cellmean
from .balanced_eval import atomic, digest, ROOT
from ..training_cm import multiplier

OUT = ROOT/'golden-mcts1000-v1'


def main():
    start = time.monotonic()
    plan = json.loads((OUT/'plan.json').read_text())
    assert digest(ROOT/'golden-balanced-v1/sample.json') == plan['sample_sha256']
    rows = json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    names = ['port_raw', 'legal', 'calibrated_direct', 'mcts_forward', 'mcts_reverse', 'mcts_elo']
    scores = {k:[] for k in names}
    costs = []
    bs = plan['roots_per_block']
    for lo in range(0, len(rows), bs):
        with np.load(OUT/f'{lo:06d}.npz') as f:
            assert list(f['game']) == [r['game'] for r in rows[lo:lo+bs]]
            assert list(f['ply']) == [r['ply'] for r in rows[lo:lo+bs]]
            for name in names:
                scores[name].append(f[name])
            costs.append(json.loads(str(f['stats'])))
    scores = {k:np.concatenate(v) for k, v in scores.items()}
    # Prior comparison is paired on exactly the same positions and game bootstrap.
    old = []
    for lo in range(0, len(rows), 32):
        with np.load(ROOT/f'golden-balanced-v1/tree-{lo:06d}.npz') as f:
            assert list(f['game']) == [r['game'] for r in rows[lo:lo+32]]
            assert list(f['ply']) == [r['ply'] for r in rows[lo:lo+32]]
            old.append(f['calibrated_four_ply'])
    names.append('previous_four_ply')
    scores['previous_four_ply'] = np.concatenate(old)
    cells = np.array([r['cell'] for r in rows])
    games = np.array([r['game'] for r in rows])
    expert = np.arange(3, 16, 4)
    baseline = json.loads((ROOT/'golden-baseline/results.json').read_text())['methods']
    full = np.array(list(baseline['canonical_raw']['cells'].values()))
    cellnames = list(baseline['canonical_raw']['cells'])
    ref = np.array([r['canonical_raw_nll'] for r in rows])
    refcorrect = np.array([r['canonical_raw_correct'] for r in rows])
    loss = np.stack([scores[k][:, 0] for k in names], axis=1)
    correct = np.stack([scores[k][:, 1] for k in names], axis=1)
    assert np.isfinite(loss).all()
    delta = loss-ref[:, None]
    accdelta = correct-refcorrect[:, None]
    boot = bootstrap_deltas(np.concatenate([delta, accdelta], axis=1), cells, games)
    draws = full[None, :, None]+boot[:, :, :len(names)]
    points = full[:, None]+np.stack([cellmean(delta[:, i], cells) for i in range(len(names))], axis=1)
    laws = json.loads((ROOT/'training-cm-laws.json').read_text())
    ci = lambda x:np.quantile(x, [.025, .975]).tolist()
    results = {}
    for i, name in enumerate(names):
        legal_i = names.index('legal')
        ece = []
        for cell in range(16):
            values = scores[name][cells == cell]
            bins = np.minimum((values[:, 2]*15).astype(int), 14)
            gap = 0.
            for bucket in range(15):
                selected = values[bins == bucket]
                if len(selected):
                    gap += len(selected)*abs(float(selected[:, 1].mean()-selected[:, 2].mean()))
            ece.append(gap/len(values))
        item = dict(sample_macro_ece=float(np.mean(ece)), sample_expert_ece=float(np.array(ece)[expert].mean()),
                    cells={c:dict(ce=float(points[j, i]), ci95=ci(draws[:, j, i]),
                                  delta_vs_legal=float(points[j, i]-points[j, legal_i]),
                                  delta_vs_legal_ci95=ci(draws[:, j, i]-draws[:, j, legal_i]),
                                  sample_ece=ece[j]) for j, c in enumerate(cellnames)})
        for metric, idx in [('macro', np.arange(16)), ('expert_macro', expert)]:
            point = float(points[idx, i].mean())
            samples = draws[:, idx, i].mean(1)
            law = laws['metrics'][metric]['law']
            alternative = laws['metrics'][metric]['alternative_shared_floor']
            raw = baseline['official_raw'][metric]
            cm = lambda loss:multiplier(law, laws['budget_nd'], raw, float(loss))
            lo, hi = ci(samples)
            item[metric] = point
            item[metric+'_ci95'] = [lo, hi]
            item[metric+'_training_eq_cm'] = cm(point)
            item[metric+'_cm_ci95'] = [cm(hi), cm(lo)]
            item[metric+'_cm_shared_floor_sensitivity'] = multiplier(alternative, laws['budget_nd'], raw, point)
            item[metric+'_accuracy'] = baseline['canonical_raw'][metric+'_accuracy']+float(cellmean(accdelta[:, i], cells)[idx].mean())
            item[metric+'_accuracy_ci95'] = ci(baseline['canonical_raw'][metric+'_accuracy']+boot[:, idx, len(names)+i].mean(1))
            for reference in ('calibrated_direct', 'previous_four_ply'):
                k = names.index(reference)
                item[metric+'_delta_vs_'+reference] = float((points[idx, i]-points[idx, k]).mean())
                item[metric+'_delta_vs_'+reference+'_ci95'] = ci((draws[:, idx, i]-draws[:, idx, k]).mean(1))
                item[metric+'_cm_vs_'+reference] = cm(point)/cm(points[idx, k].mean())
        results[name] = item
    old_reference = json.loads((ROOT/'golden-balanced-v1/results.json').read_text())['methods']['calibrated_four_ply']
    for key in ('macro', 'expert_macro', 'macro_ci95', 'expert_macro_ci95', 'macro_training_eq_cm', 'expert_macro_training_eq_cm'):
        np.testing.assert_allclose(results['previous_four_ply'][key], old_reference[key], atol=1e-12, rtol=0)
    report = dict(stage='Frozen MCTS balanced golden confirmation; reused sample, no selection on golden',
                  methods=results, positions=len(rows), games=len(set(games)),
                  estimator='Full canonical cell CE + sampled paired difference; equal-cell macro, whole-game bootstrap.',
                  caveat='CM is conditional on the transferred training-law shape; intervals exclude law-fit uncertainty.',
                  plan_sha256=digest(OUT/'plan.json'), analysis_sha256=digest(Path(__file__)),
                  scoring_seconds=sum(c['seconds'] for c in costs), evaluated_leaves=sum(c['evaluated_leaves'] for c in costs),
                  analysis_seconds=time.monotonic()-start)
    atomic(OUT/'results.json', report)
    for name, item in results.items():
        print(name, *(item[k] for k in ('macro', 'expert_macro', 'macro_training_eq_cm', 'expert_macro_training_eq_cm')), flush=True)


if __name__ == '__main__':
    main()
