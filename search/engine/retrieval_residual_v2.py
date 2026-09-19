"""Cache neighbor-model expectations for retrieval residual corrections."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .service import ROOT,atomic,GLOBAL_STOP
from .balanced_eval import digest

@torch.no_grad()
def run(oracle,spec):
    start=time.monotonic();src=ROOT/'aug-retrieval-v1';out=ROOT/'aug-retrieval-residual-v2';out.mkdir(exist_ok=True)
    plan=dict(parent_plan_sha256=digest(src/'plan.json'),parent_neighbors_sha256=digest(src/'neighbors.npz'),
        source_sha256=digest(Path(__file__)),kernels=[dict(k=k,temperature=.1) for k in (32,128)],
        methods=['linear_residual','log_ratio'],ratio_pseudocount=1.,linear_floor_fraction=.01,
        fit='Global strength per arm and base, on August fold0 with three-way game CV. Fold1 confirmation, no golden selection.',
        semantics='Additional June corpus memory. Neighbor model distribution is recomputed from stored causal hidden state and the frozen BF16 head, restricted to known query-legal moves. Compare empirical neighbor frequencies with this expected distribution. No query labels enter this GPU task.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions'];n=len(rows);k=max(len(r['legal']) for r in rows)
    ids=np.zeros((n,k),np.int64);mask=np.zeros((n,k),bool)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal']);mask[i,:len(r['legal'])]=True
    with np.load(src/'neighbors.npz') as z:indices=z['index'];similarity=z['similarity']
    complete=json.loads((ROOT/'retrieval-features-v2/results.json').read_text())
    hidden=[]
    for name,sha in complete['shards'].items():
        p=ROOT/'retrieval-features-v2'/name;assert digest(p)==sha
        with np.load(p) as z:hidden.append(z['hidden'])
    hidden=np.concatenate(hidden)
    head=oracle.runner.model.model.math.lm_head.weight
    expectations=np.zeros((len(plan['kernels']),n,k),np.float32);seconds=0.
    # Check that the head projection preserves captured root logits up to the
    # ordinary GEMM-shape rounding; report the error, do not silently call exact.
    with np.load(src/'roots.npz') as z:rh=z['hidden'][:32];rz=z['z'][:32]
    z=(23*torch.sigmoid((torch.nn.functional.linear(torch.from_numpy(rh).cuda().to(head.dtype),head)+5)/7.5)).float().cpu().numpy()
    head_drift=dict(max_abs=float(np.max(np.abs(z-rz))),mean_abs=float(np.mean(np.abs(z-rz))))
    for lo in range(0,n,128):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        hi=min(n,lo+128);begin=time.monotonic()
        h=torch.from_numpy(hidden[np.maximum(indices[lo:hi],0)]).cuda().to(head.dtype)
        # [roots,neighbors,width] x [roots,width,legal_actions].
        weights=head[torch.from_numpy(ids[lo:hi]).cuda()]
        logits=23*torch.sigmoid((torch.bmm(h,weights.transpose(1,2))+5)/7.5)
        logits=logits.float().masked_fill(~torch.from_numpy(mask[lo:hi]).cuda()[:,None,:],-torch.inf)
        p=torch.softmax(logits,dim=-1)
        for j,kernel in enumerate(plan['kernels']):
            size=kernel['k'];sim=similarity[lo:hi,:size];valid=np.isfinite(sim)
            maxima=np.where(valid,sim,-np.inf).max(1);maxima=np.where(np.isfinite(maxima),maxima,0.)
            w=np.exp(np.where(valid,(sim-maxima[:,None])/kernel['temperature'],-np.inf));den=w.sum(1)
            w/=np.maximum(den[:,None],1e-30)
            expectations[j,lo:hi]=(p[:,:size]*torch.from_numpy(w).cuda()[:,:,None]).sum(1).cpu().numpy()
        seconds+=time.monotonic()-begin
    path=out/'expected.npz';temp=path.with_suffix('.partial')
    with temp.open('wb') as f:np.savez(f,expected=expectations,ids=ids,mask=mask,game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    temp.replace(path)
    result=dict(positions=n,head_projections=int(indices.size),head_seconds=seconds,invocation_seconds=time.monotonic()-start,
        head_drift_vs_root_capture=head_drift,plan_sha256=digest(pp),expected_sha256=digest(path))
    atomic(out/'worker.json',result);print('Retrieval residual cache',result,flush=True);return result
