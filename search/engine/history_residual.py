"""Read-only history residuals from the fixed model; no parameter updates."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .features import full

VARIANTS=['same_uniform','same_hidden','same_hidden_last8','both_hidden']
def correction(tokens,hidden,logits,legal):
    # tokens is PREFIX ONLY. No requested next move or future appears here.
    n=len(tokens);out=np.zeros((len(VARIANTS),len(legal)),float)
    if n<=11:return out
    history=np.asarray(tokens[11:])-378;past=np.arange(len(history))
    own=past%2==(len(history)%2)
    h=hidden[10:n-1].astype(float);root=hidden[-1].astype(float)
    cosine=h@root/(np.maximum(np.linalg.norm(h,axis=1)*np.linalg.norm(root),1e-30))
    prob=softmax(logits[10:n-1,378:2346].astype(float),axis=1)
    residual=(history[:,None]==np.asarray(legal)[None,:]).astype(float)-prob[:,legal]
    for j,name in enumerate(VARIANTS):
        keep=np.ones(len(history),bool) if name=='both_hidden' else own.copy()
        if name=='same_hidden_last8':
            selected=np.flatnonzero(keep);keep[:]=False;keep[selected[-8:]]=True
        if not keep.any():continue
        w=np.ones(keep.sum()) if name=='same_uniform' else np.exp((cosine[keep]-cosine[keep].max())/.1)
        w/=w.sum();out[j]=w@residual[keep]
    return out

def test():
    rng=np.random.default_rng(881);h=rng.normal(size=(16,5));z=np.zeros((16,2432))
    tokens=[0]*11+[378,379,380,381,382];legal=np.array([0,1,2,3,4])
    got=correction(tokens,h,z,legal)
    # At ply5 the mover matches past plies1 and3.
    expected=np.array([0,.5,0,.5,0])-1/1968
    np.testing.assert_allclose(got[0],expected)
    np.testing.assert_array_equal(correction(tokens[:11],h[:11],z[:11],legal),np.zeros((4,5)))
    # Changing future/non-input logits cannot affect history residuals.
    zz=z.copy();zz[-1]=rng.normal(size=2432)
    np.testing.assert_array_equal(correction(tokens,h,zz,legal),got)
    print('PASS history-only target shift, same-player parity, no-history fallback, root-logit exclusion')

@torch.no_grad()
def run(oracle,spec):
    test();start=time.monotonic();out=ROOT/'aug-history-residual-v1';out.mkdir(exist_ok=True)
    sample=ROOT/'aug-tune-v1/sample.json';rows=json.loads(sample.read_text())['positions'];n=len(rows)
    plan=dict(variants=VARIANTS,sample_sha256=digest(sample),sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('features.py'),Path(__file__).with_name('direct.py')]},
        method='Weighted past-move one-hot minus model probability, using causal hidden states and same-player histories or both-player control. Cosine kernel temperature.1, optional last8 own moves. Root policy receives a calibrated residual logit feature. Fixed global checkpoint; no neural parameter updates and no external datastore.',
        fit='Per-arm global nonnegative strength0..100 fit on August fold0 with game CV, against matched direct/search controls. No query target argument in correction().')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
    c=np.zeros((len(VARIANTS),n,k),np.float32);root_z=np.zeros((n,2432),np.float32)
    group=[];size=0;groups=[]
    for i,r in enumerate(rows):
        if group and size+len(r['prefix'])>4096:groups.append(group);group=[];size=0
        group.append(i);size+=len(r['prefix'])
    if group:groups.append(group)
    head=oracle.runner.model.model.math.lm_head.weight;tokens=0;forward=0.;begin=time.monotonic()
    for group in groups:
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        prefixes=[rows[i]['prefix'] for i in group]
        # FULL requires distinct prefixes. Identical prefixes are handled singly
        # so zero-work deduplication cannot scramble per-token alignment.
        sets=[group] if len(set(map(tuple,prefixes)))==len(prefixes) else [[i] for i in group]
        for subset in sets:
            prefixes=[rows[i]['prefix'] for i in subset];oracle.reset();z,h=full(oracle,prefixes)
            hp=torch.from_numpy(h).cuda().to(head.dtype)
            all_z=(23*torch.sigmoid((torch.nn.functional.linear(hp,head)+5)/7.5)).float().cpu().numpy()
            at=0
            for i,rz,p in zip(subset,z,prefixes):
                length=len(p);m=int(mask[i].sum())
                c[:,i,:m]=correction(p,h[at:at+length],all_z[at:at+length],ids[i,:m]);root_z[i]=rz;at+=length
            tokens+=oracle.new_tokens;forward+=oracle.forward_seconds
    path=out/'features.npz';tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,correction=c,z=root_z,ids=ids,mask=mask,game=[r['game'] for r in rows],ply=[r['ply'] for r in rows])
    tmp.replace(path)
    result=dict(positions=n,seconds=time.monotonic()-begin,invocation_seconds=time.monotonic()-start,new_tokens=tokens,forward_seconds=forward,
        plan_sha256=digest(pp),features_sha256=digest(path),cost='No additional search nodes; full-prefix inference plus past hidden-head projections. Prefill tokens and wall time counted.')
    atomic(out/'worker.json',result);print('History residual',result,flush=True);return result
