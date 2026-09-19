"""Comparable inference matmul FLOPs, including prefill; no embedding lookups counted."""
import json
import numpy as np
from search.engine.service import ROOT,atomic
from .collect import OUT,inventory,BUDGETS


def compute(size):
    c=json.loads((OUT/f'ship-{size}-export/config.json').read_text())['allie']
    w,l,h,v=c['width'],c['layers'],c['mlp_hidden'],c['vocab_size'];heads=w//c['head_dim']
    assert c['ws_short']*128>1025 and not c['key_offset']
    board_macs=64*(13*32*9+2*32*32*9+32*8)+32*32+544*w
    head=2*w*v;gate=2*((l+2*min(5,l//2))*heads*16+64)
    body=8*l*w*w+6*l*w*h+2*board_macs+2*64*w+gate
    rows,_,_=inventory('gold');n=len(rows);length=np.array([len(r['prefix']) for r in rows]);base=float(np.mean(body*length+4*w*l*length*(length+1)/2+head))
    point={k:dict(prefill_flops=base,nn_decode_flops=0.,total_flops=base,exact_matmul_count=True) for k in ['legal','port_raw','calibrated_direct']}
    for mode in ['fixed','adaptive']:
        folder=OUT/f'{size}-gold-{mode}-predicted';nn=np.zeros((len(BUDGETS),n));prefix=np.zeros((len(BUDGETS),n));upper=np.zeros_like(prefix);exact=0.
        for lo in range(0,n,128):
            with np.load(folder/f'{lo:06d}.npz') as f:
                count=f['nodes'];nb,nbatch=count.shape;st=json.loads(str(f['stats']));ll=length[lo:lo+nbatch]
                nn[:nb,lo:lo+nbatch]=count;prefix[:nb,lo:lo+nbatch]=count*(ll[None,:]+1)
                upper[:nb,lo:lo+nbatch]=count*(ll[None,:]+st['max_depth']);exact+=st['useful_prefix_tokens']
        for j,key in enumerate([str(b) for b in BUDGETS] if mode=='fixed' else ['adaptive']):
            cnt=nn[j].sum();lo=(cnt*(body+head)+4*w*l*prefix[j].sum())/n;hi=(cnt*(body+head)+4*w*l*upper[j].sum())/n
            is_exact=(mode=='adaptive' or key=='1000')
            true=(cnt*(body+head)+4*w*l*exact)/n if is_exact else None
            for name in ([f'frozen_{key}',f'refit_{key}'] if mode=='fixed' else ['adaptive_frozen','adaptive_refit']):
                point[name]=dict(prefill_flops=base,nn_decode_flops=true,total_flops=base+true if is_exact else None,
                    total_flops_bounds=[base+lo,base+hi],exact_matmul_count=is_exact)
        if mode=='fixed':point['frozen_projected']=point['refit_projected']=point['frozen_1000'].copy()
    folder=OUT/f'{size}-gold-baselines'
    for mode in ['shallow','released','repaired']:
        total={};counts={}
        for lo in range(0,n,128):
            with np.load(folder/f'{mode}-{lo:06d}.npz') as f:
                st=json.loads(str(f['stats']));ll=length[lo:lo+len(f['game'])]
                if mode=='shallow':
                    by=np.diff(f['nodes'],axis=0,prepend=np.zeros((1,len(ll))))
                    for name,depth in [('two_ply',2),('four_ply',4)]:
                        count=by[:depth];p=(count*(ll[None,:]+np.arange(1,depth+1)[:,None])).sum()
                        total[name]=total.get(name,0.)+count.sum()*(body+head)+4*w*l*p
                else:
                    name='allie_'+mode
                    total[name]=total.get(name,0.)+f['nodes'].sum()*(body+head)+4*w*l*st['useful_prefix_tokens']
        for name,flops in total.items():point[name]=dict(prefill_flops=base,nn_decode_flops=flops/n,total_flops=base+flops/n,exact_matmul_count=True)
    report=dict(model=size,body_flops_per_token=body,head_flops_per_token=head,context_free_decode_flops=body+head,
        formula='Forward attention/MLP/gate/head/clock-projection/board matmuls. Decoder attention adds4*width*layers*context_length; windows exceed all prefixes. Prefill uses causal pairs and computes head only at root. Embedding lookups, norms/nonlinearities/softmax, KV traffic, CPU rules/search excluded.',
        note='Lower/upper bounds at intermediate budgets use actual NN counts and measured per-block max depth; final/adaptive/shallow attention uses actual summed query lengths. FLOPs are analytical arithmetic, not measured wall time. Allie can overcount attention in rare depth-limit cache hits.',methods=point)
    atomic(OUT/f'cost-{size}.json',report);return report


if __name__=='__main__':
    for s in ['small','large']:print(s,compute(s)['context_free_decode_flops'])
