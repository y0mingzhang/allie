"""Anytime human-policy expectation, with policy-sensitive frontier allocation.

This is a research alternative to win-maximizing PUCT. Unexpanded probability mass
retains its parent's critic value. Priority uses model predictions only; WDL variance
is a heuristic, not a calibrated estimate of epistemic error. All root moves survive.
"""
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from .native_board import from_prefix

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
STOP=Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')


@dataclass
class Node:
    root:int
    action:int
    prefix:list
    board:object
    depth:int
    mass:float
    value:float
    children:tuple


def batch(rows,oracle,*,budget=1000,width=2,depth=8,priority='mass',wave=8,batch_size=1024):
    assert priority in ('mass','policy','variance') and budget>0 and width>0 and depth>=1
    start=time.monotonic();n=len(rows);boards=[from_prefix(r['prefix']) for r in rows]
    white=[b.white for b in boards];root=oracle([r['prefix'] for r in rows]);q=np.full((n,1968),np.nan)
    heap=[[] for _ in rows];used=np.zeros(n,int);rootweight=np.zeros_like(q);serial=0;maxdepth=0;calls=1
    def frontier(node,z,variance):
        nonlocal serial
        if node.depth>=depth or len(node.prefix)>=1025 or node.mass<1e-12:return
        legal=node.board.legal();probs=softmax(z[legal].astype(float));chosen=np.argsort(probs)[::-1][:width]
        node.children=tuple((legal[j],float(probs[j])) for j in chosen)
        weight=node.mass
        if priority!='mass':weight*=rootweight[node.root,node.action]
        if priority=='variance':weight*=np.sqrt(max(variance,1e-4))
        heapq.heappush(heap[node.root],(-weight,serial,node));serial+=1
    def predict(pending,initial=False):
        nonlocal calls,maxdepth
        for lo in range(0,len(pending),batch_size):
            part=pending[lo:lo+batch_size];zs=oracle([x.prefix for x in part]);calls+=1
            p=softmax(zs[:,2413:2416].astype(float),axis=1);values=p@np.array([1.,.5,0.]);variance=p@np.array([1.,.25,0.])-values**2
            for node,z,value,var in zip(part,zs,values,variance):
                if node.depth%2:value=1-value
                if initial:q[node.root,node.action]=value
                else:q[node.root,node.action]+=node.mass*(value-node.value)
                node.value=float(value);used[node.root]+=1;maxdepth=max(maxdepth,node.depth)
                frontier(node,z,float(var))
    pending=[]
    for i,(r,b) in enumerate(zip(rows,boards)):
        legal=b.legal();assert budget>=len(legal),'Budget must cover every legal root move'
        p=softmax(root[i,legal].astype(float));rootweight[i,np.array(legal)-378]=np.sqrt(p*(1-p))
        for token in legal:
            board=b.child(token);terminal=board.outcome();a=token-378
            if terminal>=0:q[i,a]=terminal if white[i] else 1-terminal
            else:pending.append(Node(i,a,r['prefix']+[token],board,1,1.,0.,()))
    predict(pending,initial=True);initial_q=q.copy()
    while any(heap[i] and used[i]<budget for i in range(n)):
        pending=[];reserved=np.zeros(n,int)
        for i in range(n):
            for _ in range(wave):
                if not heap[i] or used[i]+reserved[i]>=budget:break
                _,_,parent=heapq.heappop(heap[i])
                for token,p in parent.children:
                    board=parent.board.child(token);terminal=board.outcome();mass=parent.mass*p
                    if terminal>=0:
                        value=terminal if white[i] else 1-terminal
                        q[i,parent.action]+=mass*(value-parent.value)
                    elif used[i]+reserved[i]<budget:
                        pending.append(Node(i,parent.action,parent.prefix+[token],board,parent.depth+1,mass,parent.value,()))
                        reserved[i]+=1
        predict(pending)
    for i,r in enumerate(rows):assert np.isfinite(q[i,np.array(r['legal'])-378]).all()
    assert np.nanmin(q)>=-1e-9 and np.nanmax(q)<=1+1e-9 and (used<=budget).all()
    return np.stack([initial_q,q]).astype(np.float32),root,dict(seconds=time.monotonic()-start,
        evaluated_leaves=int(used.sum()),per_root_leaves=used.tolist(),requests=calls,max_depth=maxdepth,
        priority=priority,root_budget=budget,width=width,depth_limit=depth,wave=wave)


def run(oracle,spec):
    source=ROOT/'dev.json';rows=json.loads(source.read_text())['positions'];out=Path(spec['output']).resolve()
    assert out.is_relative_to(ROOT.resolve());out.mkdir(exist_ok=True)
    files=[Path(__file__),Path(__file__).with_name('direct.py'),Path(__file__).with_name('native_board.py'),
           Path(__file__).with_name('analyze_expectation.py')]
    plan=dict(spec=spec,dev_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              export=json.loads((ROOT/'serving-export/provenance.json').read_text()),
              note='Dev-only, prediction-only priorities; unexpanded probability retains parent critic. No golden fitting.')
    p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2)+'\n')
    bs=spec.get('roots_per_batch',64);cost=[];start=time.monotonic()
    for lo in range(0,len(rows),bs):
        if (ROOT/'STOP').exists() or STOP.exists():raise RuntimeError('STOP requested')
        dest=out/f'{lo:06d}.npz';part=rows[lo:lo+bs]
        if not dest.exists():
            oracle.reset();q,z,stats=batch(part,oracle,budget=spec.get('budget',1000),width=spec.get('width',2),
                                         depth=spec.get('depth',8),priority=spec.get('priority','mass'))
            stats.update(new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
            tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,q=q,root=z,stats=json.dumps(stats),game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]))
            tmp.replace(dest)
        with np.load(dest) as f:cost.append(json.loads(str(f['stats'])))
        print('adaptive expectation',spec['priority'],lo+len(part),'/',len(rows),flush=True)
    from .analyze_expectation import main as analyze
    analyze(str(out))
    return dict(stage='Development research cache only; no golden scores',positions=len(rows),
                elapsed_seconds=time.monotonic()-start,scoring_seconds=sum(v['seconds'] for v in cost),
                evaluated_leaves=sum(v['evaluated_leaves'] for v in cost))
