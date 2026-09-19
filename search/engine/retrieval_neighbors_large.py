"""Larger June datastore: matched neighbors with and without shared players."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .features import last
from .retrieval_cache import read as cached_bank


def sharing(query, bank):
    """Known IDs only; either participant matching is conservatively excluded."""
    result = torch.zeros((len(query), len(bank)), dtype=torch.bool, device=query.device)
    for a in range(2):
        for b in range(2):
            result |= (query[:, a, None] != 0) & (query[:, a, None] == bank[None, :, b])
    return result


def test():
    q = torch.tensor([[1, 2], [0, 3], [0, 0]])
    b = torch.tensor([[2, 9], [7, 3], [0, 0], [4, 5]])
    np.testing.assert_array_equal(sharing(q, b).numpy(), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    # Signed reinterpretation of64-bit identities preserves equality.
    high = np.array([[2**63+3, 0], [2**63+4, 2]], dtype=np.uint64).view(np.int64)
    np.testing.assert_array_equal(sharing(torch.from_numpy(high), torch.from_numpy(high)).numpy(), np.eye(2, dtype=bool))
    print('PASS participant exclusions, unknown handling and unsigned ID representation', flush=True)


def run(oracle, spec):
    test()
    start = time.monotonic()
    bank, side = ROOT/'retrieval-features-large-v1', ROOT/'retrieval-players-large-v1'
    out = ROOT/'aug-retrieval-large-v1'
    out.mkdir(exist_ok=True)
    complete = json.loads((bank/'results.json').read_text())
    sample = ROOT/'aug-tune-expanded-v1/sample.json'
    rows = json.loads(sample.read_text())['positions']
    sm = json.loads((side/'manifest.json').read_text())
    assert sm['sample_sha256'] == digest(sample)
    assert sm['players_sha256'] == digest(side/'players.npz')
    plan = dict(bank_plan_sha256=digest(bank/'plan.json'), bank_results_sha256=digest(bank/'results.json'),
        participant_manifest_sha256=digest(side/'manifest.json'), sample_sha256=digest(sample),
        sources={p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('features.py'), Path(__file__).with_name('direct.py'), Path(__file__).with_name('retrieval_cache.py')]},
        kernels=[dict(k=k, temperature=t) for k in (32, 128, 512) for t in (.03, .1)],
        max_neighbors=512, filters=['all', 'different_players'], query_batch_size=128, root_prefill_batch=512,
        metric='cosine FP32, TF32 disabled; same format x mover-Elo cell, legal labels',
        fit='Only expanded August fit games; each base and global interpolation fitted separately within3-fold gameCV. Disjoint confirmation. Direct baseline and fixed subtree_sigma10 search baseline. Lambda[0,.95]. No golden selection.',
        attribution='Additional June human-game memory, not pure search. Same-player ablation excludes any shared participant. Conditional on this small checkpoint; report all cells, bytes, build/query cost and extra root prefill.')
    pp = out/'plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        atomic(pp, plan)
    root_path = out/'roots.npz'
    if not root_path.exists():
        zs, hs = [], []
        tokens, forward = 0, 0.
        begin = time.monotonic()
        for lo in range(0, len(rows), 512):
            oracle.reset()
            z, h = last(oracle, [r['prefix'] for r in rows[lo:lo+512]])
            zs.append(z); hs.append(h)
            tokens += oracle.new_tokens; forward += oracle.forward_seconds
        tmp = root_path.with_suffix('.partial')
        with tmp.open('wb') as f:
            np.savez(f, z=np.concatenate(zs), hidden=np.concatenate(hs), game=[r['game'] for r in rows],
                ply=[r['ply'] for r in rows], stats=json.dumps(dict(seconds=time.monotonic()-begin, input_tokens=tokens, forward_seconds=forward)))
        tmp.replace(root_path)
    data, cache_stats = cached_bank(bank)
    with np.load(root_path) as f:
        h, root_stats = f['hidden'], json.loads(str(f['stats']))
        np.testing.assert_array_equal(f['game'], [r['game'] for r in rows])
    with np.load(side/'players.npz') as f:
        bank_players, query_players = f['bank_players'].view(np.int64), f['query_players'].view(np.int64)
        np.testing.assert_array_equal(f['query_game'], [r['game'] for r in rows])
        np.testing.assert_array_equal(f['query_ply'], [r['ply'] for r in rows])
        np.testing.assert_array_equal(f['query_site'], [r['site'] for r in rows])
        with np.load(ROOT/'retrieval-bank-large-v1/bank.npz') as b:
            np.testing.assert_array_equal(f['bank_game'], b['game'])
            assert not set(b['game']) & {r['site'] for r in rows}
    assert np.isfinite(data['hidden']).all() and np.isfinite(h).all()
    n, maxk = len(rows), plan['max_neighbors']
    scores = np.full((2, n, maxk), -np.inf, np.float32)
    indices = np.full((2, n, maxk), -1, np.int64)
    same_counts = np.zeros(n, np.int32)
    cell = np.array([r['cell'] for r in rows])
    legal = np.zeros((n, 1968), bool)
    for i, r in enumerate(rows):
        legal[i, np.array(r['legal'])-378] = True
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    begin = time.monotonic()
    norm = lambda x: x.float()/x.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
    try:
        for c in range(16):
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():
                raise RuntimeError('STOP')
            ix, qi_all = np.flatnonzero(data['cell']==c), np.flatnonzero(cell==c)
            keys = norm(torch.from_numpy(data['hidden'][ix]).cuda())
            values = torch.from_numpy(data['target'][ix].astype(np.int64)).cuda()
            participants = torch.from_numpy(bank_players[data['game_ix'][ix]]).cuda()
            for lo in range(0, len(qi_all), 128):
                qi = qi_all[lo:lo+128]
                sim = norm(torch.from_numpy(h[qi]).cuda()) @ keys.T
                sim.masked_fill_(~torch.from_numpy(legal[qi]).cuda()[:, values], -torch.inf)
                shared = sharing(torch.from_numpy(query_players[qi]).cuda(), participants)
                for mode in range(2):
                    if mode:
                        sim.masked_fill_(shared, -torch.inf)
                    ss, ii = sim.topk(min(maxk, len(ix)), dim=1)
                    if mode == 0:
                        same_counts[qi] = (shared.gather(1, ii) & torch.isfinite(ss)).sum(1).cpu().numpy()
                    scores[mode, qi, :ss.shape[1]] = ss.cpu().numpy()
                    indices[mode, qi, :ss.shape[1]] = ix[ii.cpu().numpy()]
                    if mode:
                        assert not (shared.gather(1, ii) & torch.isfinite(ss)).any().item()
                del sim, shared, ss, ii
            del keys, values, participants
            print('Large neighbors cell', c, 'bank', len(ix), 'queries', len(qi_all), flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    query_seconds = time.monotonic()-begin
    valid = np.isfinite(scores)
    labels = data['target'][np.maximum(indices, 0)]
    labels[~valid] = -1
    for mode in range(2):
        assert np.all(legal[np.arange(n)[:, None], np.maximum(labels[mode], 0)][valid[mode]])
    dest = out/'neighbors.npz'
    tmp = dest.with_suffix('.partial')
    with tmp.open('wb') as f:
        np.savez(f, similarity=scores, index=indices, label=labels, same_player_neighbors=same_counts,
            game_ix=data['game_ix'][np.maximum(indices, 0)], game=[r['game'] for r in rows], ply=[r['ply'] for r in rows])
    tmp.replace(dest)
    report = dict(positions=n, bank_positions=len(data['hidden']), bank_feature_bytes=complete['bytes'],
        root=root_stats, query_seconds=query_seconds, invocation_seconds=time.monotonic()-start, cache=cache_stats,
        bank_build=complete, empty_neighbors=(~valid.any(2)).sum(1).tolist(),
        same_player_fraction_top512=float(same_counts.sum()/max(valid[0].sum(), 1)),
        same_player_fraction_by_cell=[float(same_counts[cell==c].sum()/max(valid[0, cell==c].sum(), 1)) for c in range(16)],
        plan_sha256=digest(pp), root_sha256=digest(root_path), neighbors_sha256=digest(dest))
    atomic(out/'worker.json', report)
    print('Large retrieval worker', query_seconds, report['same_player_fraction_top512'], flush=True)
    return dict(positions=n, seconds=report['invocation_seconds'], query_seconds=query_seconds)


if __name__ == '__main__':
    test()
