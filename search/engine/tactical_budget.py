"""Cached cost-substitution test: can tactical work replace ordinary tree nodes?"""
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


def run(oracle, spec):
    start = time.monotonic()
    out = ROOT / 'aug-tactical-budget-v1'
    out.mkdir(exist_ok=True)
    d = read('aug-tune-expanded-v1')
    cells, games, fm, cv, mask, target = (d[k] for k in ('cells', 'games', 'fit', 'cv', 'mask', 'target'))
    n, k = mask.shape
    ar = np.arange(n)
    source = ROOT / 'aug-tactical-v1'
    surface = ROOT / 'aug-budget-surface-v2/policies.npz'
    with np.load(surface) as f:
        np.testing.assert_array_equal(f['games'], games)
        np.testing.assert_array_equal(f['ids'], d['ids'])
        policies, nodes, budgets = f['policy'].astype(float), f['nodes'], f['budgets']
    q = np.zeros((4, n, k)); initial = np.zeros((n, k))
    extra = np.zeros(n); first = np.zeros(n); seen = np.zeros(n, bool)
    for path in sorted(source.glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo = int(path.stem); hi = lo + len(f['game'])
            assert not seen[lo:hi].any()
            np.testing.assert_array_equal(f['game'], games[lo:hi])
            np.testing.assert_array_equal(f['ply'], [r['ply'] for r in d['rows'][lo:hi]])
            q[:, lo:hi] = f['q']; initial[lo:hi] = f['initial']
            extra[lo:hi] = f['nodes'] + 1
            first[lo:hi] = f['initial_nodes'] + 1
            seen[lo:hi] = True
    assert seen.all()
    plan = dict(source_sha256=digest(Path(__file__)), surface_sha256=digest(surface),
        tactical_plan_sha256=digest(source/'plan.json'), tactical_worker_sha256=digest(source/'worker.json'),
        budgets=budgets.tolist(), arms=['base', 'oneply', 'tactical_soft05', 'tactical_max_elo'],
        fitting='Tactical variants selected in the preceding screen. Refit only their scalar output corrections within the same game folds for each lower base budget. Fit-CV selects candidate; report all arms on reused, game-disjoint August development.',
        cost='Existing ordinary search nodes + all tactical queries + the additional tactical root prefill. No deduplication credit. Baseline root prefill common to both sides and omitted from both. Compare against the next larger ordinary-search budget; these are measured unequal costs, not a claim of exact matched cost. No new NN evaluations.',
        population='All sixteen cells equally weighted; no golden access or law inversion.')
    pp = out / 'plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        atomic(pp, plan)
    records, losses = {}, {}
    for bi, budget in enumerate(budgets):
        arms = [('base', np.zeros_like(initial), False, np.zeros(n))]
        if bi < len(budgets)-1:
            arms += [('oneply', initial, False, first),
                     ('tactical_soft05', q[2]-initial, False, extra),
                     ('tactical_max_elo', q[3]-initial, True, extra)]
        for arm, feature, by_elo, cost in arms:
            name = f'{budget}_{arm}'
            group = cells % 4 if by_elo else np.zeros(n, int)
            oof = np.full(n, np.nan); params = []
            for vi, (fold, train) in enumerate([(None, fm), *[(f, fm & (cv != f)) for f in range(3)]]):
                p = policies[bi, vi] / policies[bi, vi].sum(1, keepdims=True)
                base = np.where(mask, np.log(np.maximum(p, 1e-300)), 0.)
                pred = np.zeros_like(p); coefficients = {}
                for g in np.unique(group):
                    take = train & (group == g); allg = group == g
                    ct = np.bincount(cells[take], minlength=16)
                    w = 1 / ct[cells[take]]; w /= w.sum()
                    objective = lambda a: tilted(a, base[take], feature[take], mask[take], target[take], w)
                    alpha = 0.
                    if arm != 'base':
                        opt = minimize_scalar(objective, bounds=(0,20), method='bounded')
                        assert opt.success
                        alpha = min((0., float(opt.x), 20.), key=objective)
                    pred[allg] = tilted(alpha, base[allg], feature[allg], mask[allg], target[allg])
                    coefficients[str(int(g))] = alpha
                assert np.isfinite(pred).all() and (pred[mask] > 0).all()
                ll = -np.log(pred[ar,target]); params.append(dict(fold=fold, coefficients=coefficients))
                if fold is None:
                    losses[name] = ll
                else:
                    oof[fm & (cv == fold)] = ll[fm & (cv == fold)]
            a, b = means(losses[name][~fm],cells[~fm]), means(oof[fm],cells[fm])
            counts = means(nodes[bi]+cost,cells)
            records[name] = dict(parameters=params, training_equivalent_cm=None,
                mean_nodes=float(counts.mean()), expert_mean_nodes=float(counts[3::4].mean()),
                confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
                fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())),
                next_base=f'{budgets[min(bi+1,len(budgets)-1)]}_base')
    _,ix=np.unique(games[~fm],return_inverse=True); ng=ix.max()+1
    ct=np.zeros((ng,16)); np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(41912).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float)
    den=draws@ct; assert (den>0).all()
    for name, rec in records.items():
        reference=rec['next_base']; delta=losses[name]-losses[reference]
        point=means(delta[~fm],cells[~fm]); sums=np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm]); boot=draws@sums/den
        rec['delta_vs_next_base']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),
            expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist(),
            mean_nodes_delta=rec['mean_nodes']-records[reference]['mean_nodes'])
    result=dict(results=records,analysis_seconds=time.monotonic()-start,plan_sha256=digest(pp))
    atomic(out/'results.json',result)
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print(json.dumps(result,indent=2),flush=True)
    return dict(study=out.name,analysis_seconds=result['analysis_seconds'])
