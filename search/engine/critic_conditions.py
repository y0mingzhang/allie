"""Counterfactual rating context for values only; real human-policy prior retained."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .service import ROOT,atomic,GLOBAL_STOP
from .balanced_eval import digest
from .native_board import from_prefix
from .handles import HandleOracle

VARIANTS=['actual','equal1800','equal2400','plus400']


def transform(prefix,name):
    p=list(prefix);assert all(0<=t<=9 for t in p[3:11])
    if name=='actual':return p
    for a in (3,7):
        old=int(''.join(map(str,p[a:a+4])))
        value=int(name[5:]) if name.startswith('equal') else min(3200,max(400,old+400))
        p[a:a+4]=list(map(int,f'{value:04d}'))
    assert p[:3]==prefix[:3] and p[11:]==prefix[11:]
    return p


def run(oracle,spec):
    out=ROOT/'aug-critic-conditions-v1';out.mkdir(exist_ok=True)
    sample=ROOT/'aug-tune-expanded-v1/sample.json';rows=json.loads(sample.read_text())['positions']
    plan=dict(variants=VARIANTS,sample_sha256=digest(sample),roots_per_block=512,
        sources={p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(s) for s in ('handles.py','native_board.py','direct.py')]]},
        hypothesis='Actual-rating policy stays unchanged. Check whether equal-strength or shifted-strength VALUE-head queries offer complementary position-quality information. One-ply screen before any expensive conditioned deep search.',
        semantics='Headers alone change in hypothetical queries, all move histories and TC unchanged. No target or future outcomes supplied. Terminal child values from rules. Store WDL from ROOT MOVER perspective. Extra full-prefix query plus each nonterminal legal child is explicitly charged.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    start=time.monotonic();statistics=[]
    for lo in range(0,len(rows),512):
        part=rows[lo:lo+512];n=len(part);k=max(len(r['legal']) for r in part)
        owners=[];handles=[];nodes=np.zeros(n,int);terminal=np.zeros((n,k),bool)
        exact=np.zeros((n,k,3),float)
        for i,r in enumerate(part):
            board=from_prefix(r['prefix']);assert sorted(board.legal())==sorted(r['legal'])
            for j,move in enumerate(r['legal']):
                outcome=board.child(move).outcome()
                if outcome>=0:
                    terminal[i,j]=True;exact[i,j,1 if outcome==.5 else 0]=1.
                else:
                    handles.append([n+len(handles),i,move,len(r['prefix'])+1]);owners.append((i,j));nodes[i]+=1
        handles=np.array(handles,dtype=np.int32)
        for name in VARIANTS:
            folder=out/name;folder.mkdir(exist_ok=True);path=folder/f'{lo:06d}.npz'
            if path.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            begin=time.monotonic();oracle.reset()
            prefixes=[transform(r['prefix'],name) for r in part]
            bridge=HandleOracle(oracle,prefixes);root=bridge.root_logits
            wdl=exact.copy()
            if len(handles):
                z=bridge(handles);probs=softmax(z[:,2413:2416].astype(float),axis=1)[:,[2,1,0]]
                oi,oj=np.array(owners).T;wdl[oi,oj]=probs
            assert bridge.queries==int(nodes.sum())
            stats=dict(seconds=time.monotonic()-begin,forward_seconds=oracle.forward_seconds,new_tokens=oracle.new_tokens,
                root_queries=n,full_prefix_tokens=sum(map(len,prefixes)),nonroot_nn_requests=int(nodes.sum()))
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,wdl=wdl,root=root,nodes=nodes,terminal=terminal,game=[r['game'] for r in part],ply=[r['ply'] for r in part],stats=json.dumps(stats))
            tmp.replace(path);statistics.append(stats)
        print('critic conditions',lo+n,'/',len(rows),flush=True)
    result=dict(positions=len(rows),variants=VARIANTS,seconds=time.monotonic()-start,blocks=statistics)
    atomic(out/'worker.json',result);return result
