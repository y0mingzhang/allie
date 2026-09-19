"""Same-cell hidden-neighbor pilot; labels used only from the June datastore."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .features import last

def neighbors(keys,values,query,legal,k=128):
    """Cosine neighbors restricted to known-legal moves. No query target argument."""
    assert keys.ndim==query.ndim==2 and keys.shape[1]==query.shape[1]
    norm=lambda x:x.float()/x.float().norm(dim=1,keepdim=True).clamp_min(1e-12)
    similarity=norm(query)@norm(keys).T
    allowed=legal[:,values.long()]
    similarity=similarity.masked_fill(~allowed,-torch.inf)
    score,index=similarity.topk(min(k,len(keys)),dim=1)
    return score,index

def test():
    rng=np.random.default_rng(9838)
    keys=rng.normal(size=(38,7)).astype(np.float32);q=rng.normal(size=(9,7)).astype(np.float32)
    labels=rng.integers(0,13,size=38);legal=rng.random((9,13))>.3
    sim,ix=neighbors(torch.from_numpy(keys),torch.from_numpy(labels),torch.from_numpy(q),torch.from_numpy(legal),8)
    ref=q/np.linalg.norm(q,axis=1,keepdims=True)@(keys/np.linalg.norm(keys,axis=1,keepdims=True)).T
    ref[~legal[:,labels]]=-np.inf
    order=np.argsort(-ref,axis=1)[:,:8]
    np.testing.assert_array_equal(ix.numpy(),order)
    np.testing.assert_allclose(sim.numpy(),np.take_along_axis(ref,order,1),atol=2e-7)
    print('PASS cosine legal-neighbor top-k independent numpy reference',flush=True)

def run(oracle,spec):
    test();start=time.monotonic()
    bank=ROOT/'retrieval-features-v2';out=ROOT/'aug-retrieval-v1';out.mkdir(exist_ok=True)
    complete=json.loads((bank/'results.json').read_text())
    sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    plan=dict(bank_plan_sha256=digest(bank/'plan.json'),bank_results_sha256=digest(bank/'results.json'),
        sample_sha256=digest(sample),sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('features.py'),Path(__file__).with_name('direct.py')]},
        kernels=[dict(k=k,temperature=t) for k in (8,32,128) for t in (.01,.03,.1)],
        query_batch_size=128,root_prefill_batch=512,metric='cosine FP32, TF32 disabled',
        neighbor_filter='Same format x mover-Elo cell, and bank move legal at query. Missing neighbors get zero weight; no legal neighbor falls back to baseline.',
        fit='August fold0 only, three-fold game CV; lambda in [0,.95], global per kernel and base. Both fold0 CV winners reported on fold1. No golden yet.',
        caveat='Retrieval augmentation adds June human-game memory. Separate from pure search and disclose build/query cost, bytes and examples.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    root_path=out/'roots.npz'
    if not root_path.exists():
        zs=[];hs=[];tokens=0;forward=0.;begin=time.monotonic()
        for lo in range(0,len(rows),512):
            oracle.reset();z,h=last(oracle,[r['prefix'] for r in rows[lo:lo+512]])
            zs.append(z);hs.append(h);tokens+=oracle.new_tokens;forward+=oracle.forward_seconds
        temp=root_path.with_suffix('.partial')
        with temp.open('wb') as f:np.savez(f,z=np.concatenate(zs),hidden=np.concatenate(hs),game=[r['game'] for r in rows],ply=[r['ply'] for r in rows],stats=json.dumps(dict(seconds=time.monotonic()-begin,input_tokens=tokens,forward_seconds=forward)))
        temp.replace(root_path)
    chunks=[]
    for name,sha in complete['shards'].items():
        path=bank/name;assert digest(path)==sha
        with np.load(path) as z:chunks.append({k:z[k] for k in ('hidden','target','cell','game_ix','position')})
    data={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
    del chunks
    with np.load(root_path) as z:h=z['hidden'];root_stats=json.loads(str(z['stats']))
    with np.load(ROOT/'retrieval-bank-v2/bank.npz') as z:bank_games=z['game']
    assert not set(bank_games).intersection(r['game'] for r in rows)
    assert np.isfinite(data['hidden']).all() and np.isfinite(h).all()
    old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    scores=np.full((len(rows),128),-np.inf,np.float32);indices=np.full((len(rows),128),-1,np.int64)
    cell=np.array([r['cell'] for r in rows]);legal=np.zeros((len(rows),1968),bool)
    for i,r in enumerate(rows):legal[i,np.asarray(r['legal'])-378]=True
    begin=time.monotonic()
    try:
        for c in range(16):
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            ix=np.flatnonzero(data['cell']==c);query_ix=np.flatnonzero(cell==c)
            keys=torch.from_numpy(data['hidden'][ix]).cuda();values=torch.from_numpy(data['target'][ix].astype(np.int64)).cuda()
            for lo in range(0,len(query_ix),128):
                qi=query_ix[lo:lo+128]
                ss,ii=neighbors(keys,values,torch.from_numpy(h[qi]).cuda(),torch.from_numpy(legal[qi]).cuda())
                scores[qi,:ss.shape[1]]=ss.cpu().numpy();indices[qi,:ss.shape[1]]=ix[ii.cpu().numpy()]
            del keys,values
    finally:torch.backends.cuda.matmul.allow_tf32=old
    query_seconds=time.monotonic()-begin
    valid=np.isfinite(scores);assert np.all(indices[valid]>=0)
    labels=data['target'][np.maximum(indices,0)];labels[~valid]=-1
    assert np.all(legal[np.arange(len(rows))[:,None],np.maximum(labels,0)][valid])
    path=out/'neighbors.npz';temp=path.with_suffix('.partial')
    with temp.open('wb') as f:np.savez(f,similarity=scores,index=indices,label=labels,game_ix=data['game_ix'][np.maximum(indices,0)],game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    temp.replace(path)
    result=dict(positions=len(rows),bank_positions=len(data['hidden']),bank_feature_bytes=complete['bytes'],
        root=root_stats,query_seconds=query_seconds,invocation_seconds=time.monotonic()-start,
        empty_neighbors=int((~valid.any(1)).sum()),plan_sha256=digest(pp),root_sha256=digest(root_path),neighbors_sha256=digest(path))
    atomic(out/'worker.json',result);print('Retrieval neighbors',result,flush=True);return result
