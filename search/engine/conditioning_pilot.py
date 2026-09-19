"""Cheap counterfactual-strength policy queries; moves/history stay identical."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .service import ROOT,atomic,GLOBAL_STOP

VARIANTS=[('base',0,0),('both_m400',-400,-400),('both_m200',-200,-200),
          ('both_p200',200,200),('both_p400',400,400),('both_p800',800,800),
          ('self_p400',400,0),('opponent_p400',0,400)]


def transform(prefix,self_delta,opponent_delta):
    p=list(prefix);assert len(p)>=11 and all(0<=t<=9 for t in p[3:11])
    mover_white=(len(p)-11)%2==0
    for start,delta in [(3,self_delta if mover_white else opponent_delta),(7,opponent_delta if mover_white else self_delta)]:
        old=int(''.join(map(str,p[start:start+4])))
        # Baseline is byte-identical even outside the counterfactual clamp range.
        new=old if delta==0 else int(np.clip(old+delta,400,3200))
        p[start:start+4]=list(map(int,f'{new:04d}'))
    assert p[:3]==prefix[:3] and p[11:]==prefix[11:]
    return p


def run(oracle,spec):
    out=ROOT/'aug-conditioning-v1';out.mkdir(exist_ok=True)
    path=ROOT/'aug-tune-v1/sample.json';rows=json.loads(path.read_text())['positions']
    plan=dict(variants=VARIANTS,sample_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage='August development only; counterfactual ratings selected without next-move targets. Root-only queries incur full-prefix work, reported separately from cached child expansions.')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==json.loads(json.dumps(plan))
    else:atomic(out/'plan.json',plan)
    begin=time.monotonic();cost=[]
    for name,a,b in VARIANTS:
        dest=out/f'{name}.npz'
        if dest.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        start=time.monotonic();parts=[];tokens=0;forward=0.;k=max(len(r['legal']) for r in rows)
        for lo in range(0,len(rows),512):
            part=rows[lo:lo+512];oracle.reset()
            z=oracle([transform(r['prefix'],a,b) for r in part]);q=np.zeros((len(part),k),np.float32)
            for i,r in enumerate(part):q[i,:len(r['legal'])]=z[i,r['legal']]
            parts.append(q);tokens+=oracle.new_tokens;forward+=oracle.forward_seconds
        stats=dict(name=name,seconds=time.monotonic()-start,new_tokens=tokens,forward_seconds=forward,root_queries=len(rows))
        tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,logits=np.concatenate(parts),stats=json.dumps(stats))
        tmp.replace(dest);cost.append(stats);print('Counterfactual ratings',stats,flush=True)
    result=dict(seconds=time.monotonic()-begin,variants=len(VARIANTS),positions=len(rows),cost=cost)
    atomic(out/'worker.json',result);return result
