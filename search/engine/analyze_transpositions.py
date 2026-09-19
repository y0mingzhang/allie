"""Prefix-only history smoothing: direct diagnostics and incremental search gains."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from .balanced_eval import ROOT, atomic, digest
from .analyze_august import means
from .innovation import fit


def main():
    start = time.monotonic()
    out = ROOT / 'aug-transpositions-v1'
    rows = json.loads((ROOT / 'aug-tune-v1/sample.json').read_text())['positions']
    n = len(rows); ar = np.arange(n)
    cells = np.array([r['cell'] for r in rows]); games = np.array([r['game'] for r in rows])
    fm = np.array([r['fold'] == 0 for r in rows])
    cv = np.array([int(hashlib.sha256(('cv:' + g).encode()).hexdigest(), 16) % 3 for g in games])
    k = max(len(r['legal']) for r in rows)
    mask = np.zeros((n, k), bool); ids = np.zeros((n, k), int); target = np.zeros(n, int)
    for i, r in enumerate(rows):
        mask[i, :len(r['legal'])] = True
        ids[i, :len(r['legal'])] = np.array(r['legal']) - 378
        target[i] = r['legal'].index(r['target'])
    z = np.zeros((n, 5, k)); count = np.zeros(n, int)
    root = np.zeros((n, 2432)); q = np.zeros((n, k)); nodes = np.zeros(n)
    for lo in range(0, n, 512):
        with np.load(out / f'{lo:06d}.npz') as f:
            hi = lo + len(f['game']); assert list(f['game']) == list(games[lo:hi])
            z[lo:hi, :, :f['logits'].shape[-1]] = f['logits']; count[lo:hi] = f['counts']
        with np.load(ROOT / f'aug-selection-v1/zero_cp25/{lo:06d}.npz') as f:
            assert list(f['game']) == list(games[lo:hi])
            root[lo:hi] = f['z']; q[lo:hi, :f['q'].shape[-1]] = f['q'][-1]
            nodes[lo:hi] = f['evaluated_nodes'][-1]
    assert ((1 <= count) & (count <= 5)).all()
    # All missing variants retain the original policy, never omit a position.
    for j in range(1, 5): z[count <= j, j] = z[count <= j, 0]
    raw = np.where(mask[:, None, :], z, -np.inf)
    lp = raw - logsumexp(raw, axis=2, keepdims=True)
    available = np.arange(1, 5)[None, :] < count[:, None]
    weights = available / np.maximum(count[:, None] - 1, 1)
    avg_p = np.einsum('nv,nvk->nk', weights, np.exp(lp[:, 1:]))
    avg_z = np.einsum('nv,nvk->nk', weights, z[:, 1:])
    avg_p[count == 1] = np.exp(lp[count == 1, 0]); avg_z[count == 1] = z[count == 1, 0]
    arithmetic = np.log(np.maximum(.5 * np.exp(lp[:, 0]) + .5 * avg_p, 1e-300))
    geometric_z = np.where(mask, .5 * z[:, 0] + .5 * avg_z, -np.inf)
    geometric = geometric_z - logsumexp(geometric_z, axis=1, keepdims=True)
    assert np.allclose(np.exp(arithmetic)[mask].sum(), n)
    np.testing.assert_allclose(arithmetic[count == 1][mask[count == 1]], lp[count == 1, 0][mask[count == 1]], atol=1e-12)
    np.testing.assert_allclose(geometric[count == 1][mask[count == 1]], lp[count == 1, 0][mask[count == 1]], atol=1e-12)
    direct = {}
    for name, v in [(f'variant{j}', lp[:, j]) for j in range(5)] + [('arithmetic', arithmetic), ('geometric', geometric)]:
        a = means(-v[ar, target][~fm], cells[~fm])
        direct[name] = dict(macro_ce=float(a.mean()), expert_ce=float(a[3::4].mean()), cells=a.tolist())
    logits = np.where(mask, root[:, 378:2346][ar[:, None], ids], 0.)
    # Delta isolates history effects: no credit for new batches changing the
    # original prefix's BF16 logits. The calibrated baseline stays bit-identical.
    da = np.zeros((n, k)); dg = np.zeros((n, k))
    da[mask] = arithmetic[mask] - lp[:, 0][mask]
    dg[mask] = geometric[mask] - lp[:, 0][mask]
    menus = [('direct', [logits, q * 0]), ('arithmetic_direct', [logits + da, q * 0]),
             ('geometric_direct', [logits + dg, q * 0]), ('search', [logits, q]),
             ('arithmetic_search', [logits + da, q]), ('geometric_search', [logits + dg, q])]
    records = {}; losses = {}
    for name, features in menus:
        f = np.stack(features, 1)
        p, params = fit(f, mask, target, cells, fm, 0.)
        loss = -p[ar, target]; oof = np.full(n, np.nan); converged = []
        for fold in range(3):
            val = fm & (cv == fold)
            v, info = fit(f, mask, target, cells, fm & (cv != fold), 0.)
            oof[val] = -v[ar[val], target[val]]; converged.append(info['converged'])
        a = means(loss[~fm], cells[~fm]); b = means(oof[fm], cells[fm]); losses[name] = loss
        extra = count if name not in ('direct', 'search') else np.zeros(n)
        basecost = nodes if name.endswith('search') else np.zeros(n)
        records[name] = dict(parameters=params, cv_converged=converged, training_equivalent_cm=None,
            mean_search_nodes=float(means(basecost, cells).mean()),
            mean_additional_full_prefix_queries=float(means(extra, cells).mean()),
            mean_logical_queries=float(means(basecost + extra, cells).mean()),
            confirmation=dict(macro_ce=float(a.mean()), expert_ce=float(a[3::4].mean()), cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()), expert_ce=float(b[3::4].mean())))
        print(name, records[name]['fit_game_cv'], records[name]['confirmation']['macro_ce'], records[name]['confirmation']['expert_ce'], flush=True)
    prior = json.loads((ROOT / 'aug-selection-v1/results.json').read_text())['results']['zero_cp25_1000_elo']['confirmation']
    for key in ('macro_ce', 'expert_ce'):
        np.testing.assert_allclose(records['search']['confirmation'][key], prior[key], atol=2e-6, rtol=0)
    selected = {metric: min(records, key=lambda k: records[k]['fit_game_cv'][metric]) for metric in ('macro_ce', 'expert_ce')}
    _, ix = np.unique(games[~fm], return_inverse=True); g = ix.max() + 1
    count_game = np.zeros((g, 16)); np.add.at(count_game, (ix, cells[~fm]), 1)
    w = np.random.default_rng(8317).multinomial(g, np.full(g, 1/g), size=2000).astype(float)
    den = w @ count_game; assert (den > 0).all()
    for name in records:
        reference = 'search' if name.endswith('search') else 'direct'
        sums = np.zeros((g, 16)); np.add.at(sums, (ix, cells[~fm]), (losses[name] - losses[reference])[~fm])
        draws = w @ sums / den
        records[name]['paired_reference'] = reference
        records[name]['confirmation_delta_ci95'] = dict(macro=np.quantile(draws.mean(1), [.025, .975]).tolist(), expert=np.quantile(draws[:, 3::4].mean(1), [.025, .975]).tolist())
    coverage = json.loads((out / 'variants.json').read_text()); coverage.pop('variants')
    worker = json.loads((out / 'worker.json').read_text())
    atomic(out / 'results.json', dict(results=records, direct_legal_diagnostics=direct, fit_cv_selected=selected,
        coverage=coverage, measured_cost=worker, analysis_seconds=time.monotonic()-start, source_sha256=digest(Path(__file__)),
        stage='August potentially training-seen, game-disjoint fit/confirmation. Half original and half available-variant mean per position. All missing variants fall back. Prefix-only generation, unchanged absolute positions, no clock inputs. Delta correction preserves cached root logits; no batch drift credited. Full-prefix queries/tokens cost more than cached leaf nodes and are reported separately. CM pending golden.'))
    with (out / 'scores.npz').open('wb') as f:
        np.savez_compressed(f, names=list(losses), loss=np.stack(list(losses.values())), cells=cells, games=games, fit=fm)
    print('SELECTED', selected, flush=True)


if __name__ == '__main__': main()
