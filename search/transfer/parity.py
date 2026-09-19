"""GPU full-prefix/cached parity, causal context and repeatability before scoring."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import log_softmax,softmax
from search.engine.service import ROOT,atomic
from search.engine.balanced_eval import digest
from .model import DenseBackend
from .handles import ShipHandles
from .board import encode


def comparison(a,b,targets):
    la,lb=(log_softmax(x[:,378:2346].astype(float),axis=1) for x in (a,b))
    pa,pb=softmax(a[:,2413:2416].astype(float),axis=1),softmax(b[:,2413:2416].astype(float),axis=1)
    return dict(max_logit_gap=float(abs(a-b).max()),mean_move_kl=float(np.mean(np.sum(np.exp(la)*(la-lb),1))),
        target_ce_gap=float(np.mean((la-lb)[np.arange(len(targets)),targets])),
        max_wdl_value_gap=float(abs((pa[:,0]-pa[:,2])-(pb[:,0]-pb[:,2])).max()))


def run(oracle,spec):
    start=time.monotonic();size=spec['model'];tag=spec.get('audit_tag','');assert not tag or tag.replace('-','').isalnum()
    out=ROOT/'transfer-v1'/('parity-'+size+('-'+tag if tag else ''));out.mkdir(exist_ok=True)
    assert (ROOT/'transfer-v1'/('math-'+size+'.json')).exists(),'CPU frozen-source math gate not complete'
    sample=json.loads((ROOT/'aug-tune-expanded-v1/sample.json').read_text())['positions']
    rows=[r for cell in range(16) for r in [r for r in sample if r['cell']==cell and r['fold']==0][:8]]
    with np.load(ROOT/'aug-tune-v1/feats.npz') as f:side=f['feats']
    prefixes=[r['prefix'] for r in rows]
    features=[side[r['row'],r['column']-len(r['prefix']):r['column']].copy() for r in rows]
    targets=np.array([r['target']-378 for r in rows]);n=len(rows)
    oracle.reset();bridge=ShipHandles(oracle,prefixes,features);root=bridge.root_logits.copy()
    handles=np.array([[n+i,i,r['legal'][0],len(r['prefix'])+1] for i,r in enumerate(rows)],np.int32)
    child=bridge(handles);child_features=[np.vstack((f,bridge.feats[n+i])) for i,f in enumerate(features)]
    child_prefixes=[p+[r['legal'][0]] for p,r in zip(prefixes,rows)]
    for i,p in enumerate(child_prefixes):np.testing.assert_array_equal(bridge.boards[n+i],encode(np.array(p)[None])[0,-1])
    oracle.reset();fresh=oracle.prefill(child_prefixes,child_features)
    cached_check=comparison(fresh,child,targets)
    oracle.reset();repeat=oracle.prefill(prefixes,features);np.testing.assert_array_equal(repeat,root)
    # Same frozen parameters, independent eager dense attention, same side inputs.
    math=oracle.runner.model.model.math;reference=[]
    with torch.inference_mode():
        for p,f in zip(prefixes,features):
            ids=torch.tensor(p,device='cuda');pos=torch.arange(len(p),device='cuda')
            b=torch.tensor(encode(np.array(p)[None])[0],device='cuda')
            reference.append(math(ids,pos,DenseBackend(),torch.tensor(f,device='cuda'),b)[-1].float().cpu().numpy())
    dense_check=comparison(np.asarray(reference),root,targets)
    result=dict(model=size,positions=n,cached_vs_fresh=cached_check,dense_vs_served=dense_check,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        repeat_bit_exact=True,board_transition_exact=True,seconds=time.monotonic()-start,
        export_sha256=digest(ROOT/'transfer-v1'/('ship-'+size+'-export')/'provenance.json'),
        clock_rule='Predicted next-move elapsed seconds rounded/clipped to remaining clock; increment afterward; third cf3 channel tracks same mover previous own think time. First moves do not tick. Missing clocks remain absent.',
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    atomic(out/'diagnostic.json',result)
    assert dense_check['mean_move_kl']<.002 and abs(dense_check['target_ce_gap'])<.005,dense_check
    assert cached_check['mean_move_kl']<.002 and abs(cached_check['target_ce_gap'])<.005,cached_check
    atomic(out/'results.json',result);print('PARITY PASS',size,dense_check,cached_check,flush=True)
    return result
