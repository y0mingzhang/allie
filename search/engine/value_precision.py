"""Diagnose critic rounding on fixed descendant prefixes; never changes model weights."""
import json, time, types
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .handles import HandleOracle

OUT=ROOT/'critic-precision-audit-v1'


def run(oracle,spec):
    OUT.mkdir(exist_ok=True)
    source=ROOT/'aug-deep-order-audit-v1/fixed1000-repeat/trace-paths.json'
    paths=sorted(map(tuple,json.loads(source.read_text())),key=lambda p:(len(p),p))
    root=paths[0];assert all(p[:len(root)]==root for p in paths)
    lookup={p:i for i,p in enumerate(paths)}
    assert len(lookup)==len(paths) and all(p[:-1] in lookup for p in paths[1:])
    plan=dict(source_sha256=digest(Path(__file__)),paths_sha256=digest(source),modes=['bf16','fp32_critic'],batch_sizes=[1,64,256],
        purpose='Software/numerics diagnosis only on fixed outlier descendants. No search decisions, target moves or quality selection. Compare same histories and causal KV extension under different batches.',
        intervention='Keep transformer and all move/time logits unchanged. Recompute only W/D/L rows2413:2416 from final hidden states using FP32 linear+softcap, without changing any weight. Scope patch to each test and restore unconditionally.',
        cost='Every fixed-prefix descendant queried once per mode/batch-size. This diagnostic is not a new deployed search result and has no CM.')
    pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    processor=oracle.runner.model.logits_processor;original=processor._compute_lm_head
    def precise(self,hidden_states,lm_head,embedding_bias=None):
        result=original(hidden_states,lm_head,embedding_bias).clone()
        wdl=torch.nn.functional.linear(hidden_states.float(),lm_head.weight[2413:2416].float())
        result[...,2413:2416]=23*torch.sigmoid((wdl+5)/7.5)
        return result
    saved={};stats={};started=time.monotonic()
    try:
        for mode in plan['modes']:
            processor._compute_lm_head=original if mode=='bf16' else types.MethodType(precise,processor)
            for bs in plan['batch_sizes']:
                name=f'{mode}-{bs}';dest=OUT/f'{name}.npz'
                if dest.exists():
                    with np.load(dest) as f:saved[name]=f['logits'];stats[name]=json.loads(str(f['stats']))
                    continue
                if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
                tick=time.monotonic();oracle.reset();bridge=HandleOracle(oracle,[root]);z=np.zeros((len(paths),2432));z[0]=bridge.root_logits[0]
                for depth in sorted(set(map(len,paths[1:]))):
                    level=[i for i,p in enumerate(paths) if len(p)==depth]
                    for lo in range(0,len(level),bs):
                        ix=level[lo:lo+bs]
                        h=np.array([[i,lookup[paths[i][:-1]],paths[i][-1],len(paths[i])] for i in ix],dtype=np.int32)
                        z[ix]=bridge(h)
                assert bridge.queries==len(paths)-1
                s=dict(seconds=time.monotonic()-tick,nn_queries=bridge.queries,prefill_tokens=len(root),forward_seconds=oracle.forward_seconds)
                saved[name]=z;stats[name]=s;temp=dest.with_suffix('.partial')
                with temp.open('wb') as f:np.savez_compressed(f,logits=z,stats=json.dumps(s))
                temp.replace(dest);print('critic-precision',name,s,flush=True)
    finally:processor._compute_lm_head=original
    result={}
    for mode in plan['modes']:
        base=saved[f'{mode}-1'];bp=softmax(base[:,2413:2416],axis=1);bv=bp[:,0]-bp[:,2]
        for bs in plan['batch_sizes']:
            name=f'{mode}-{bs}';z=saved[name];p=softmax(z[:,2413:2416],axis=1);v=p[:,0]-p[:,2]
            result[name]=dict(**stats[name],max_all_logit_diff=float(np.max(abs(z-base))),value_abs_delta_quantiles=np.quantile(abs(v-bv),[0,.5,.9,.99,1]).tolist())
    # The precision switch has no effect on hidden states/KV or other head rows.
    untouched=np.ones(2432,bool);untouched[2413:2416]=False
    for bs in plan['batch_sizes']:
        np.testing.assert_array_equal(saved[f'bf16-{bs}'][:,untouched],saved[f'fp32_critic-{bs}'][:,untouched])
    output=dict(results=result,seconds=time.monotonic()-started,noncritic_logits_unchanged=True,plan_sha256=digest(pp))
    atomic(OUT/'results.json',output);return output
