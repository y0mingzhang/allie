"""Exact golden baseline and reusable canonical logits, before search evaluation.

No choices are fitted here: official raw, canonical raw, and legal normalization.
Uses unchanged strat-eval-v1 arrays and equal-cell metrics.
"""
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from client import Oracle
import golden_data
import training_cm

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1';OUT=ROOT/'golden-baseline'
OFFICIAL=Path('/data/group_data/dei-group/yimingz3/allie/results/lm-eval/r2-3e16-control-t20-w20-pf052h-s42/strat-v1.json')


def task(index,docs):
    dest=OUT/f'{index:05d}.npz'
    if dest.exists():return index
    positions=[];indices=[];offset=0;start=time.monotonic()
    for doc in docs:
        for p in golden_data.positions(doc):positions.append(p);indices.append(offset+p['root_index'])
        offset+=len(doc['prefix'])
    oracle=Oracle();rootfile=OUT/f'{index:05d}-roots.npz'
    if rootfile.exists():
        with np.load(rootfile) as z:
            root=z['logits'];assert np.array_equal(z['game'],[p['game'] for p in positions])
            assert np.array_equal(z['ply'],[p['ply'] for p in positions])
    else:
        pred=oracle([d['prefix'] for d in docs],all_tokens=True,compact=True);assert len(pred)==offset
        root=pred[np.asarray(indices,int)]
        tmp=rootfile.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez(f,logits=root,game=np.array([p['game'] for p in positions]),ply=np.array([p['ply'] for p in positions]))
        tmp.replace(rootfile)
    move=root[:,378:2346].astype(np.float64);target=np.array([p['target']-378 for p in positions]);ar=np.arange(len(positions))
    mask=np.zeros_like(move,dtype=bool)
    for i,p in enumerate(positions):mask[i,np.array(p['legal'])-378]=True
    loss=[];correct=[];confidence=[]
    for x in (move,np.where(mask,move,-np.inf)):
        norm=logsumexp(x,axis=1);loss.append(norm-x[ar,target]);correct.append(x.argmax(1)==target);confidence.append(np.exp(x.max(1)-norm))
    tmp=dest.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,nll=np.stack(loss,1),correct=np.stack(correct,1),confidence=np.stack(confidence,1),
        cell=np.array([p['cell'] for p in positions]),game=np.array([p['game'] for p in positions]),ply=np.array([p['ply'] for p in positions]),seconds=time.monotonic()-start)
    tmp.replace(dest);return index


def main():
    OUT.mkdir(exist_ok=True);rows,labels,manifest=golden_data.load();official=json.loads(OFFICIAL.read_text());oracle=Oracle()
    assert oracle.ready['checkpoint_sha256']==official['model_sha256'] and official['strat_sha256']==manifest['sha256']
    plan=dict(checkpoint=oracle.ready['checkpoint_sha256'],strat_sha256=manifest['sha256'],games_per_block=32,
        official_sha256=hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(),
        source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ('golden_baseline.py','golden_data.py')},
        protocol='Existing golden arrays. Canonical BF16 document coordinates. No fitted parameters, no search selection.')
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2)+'\n')
    docs=list(golden_data.games(rows,labels));total=(len(docs)+31)//32;next_block=0;done=0;start=time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        pending=set()
        while next_block<total or pending:
            while next_block<total and len(pending)<8:
                assert not (ROOT/'STOP').exists() and not Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP').exists()
                i=next_block;next_block+=1;pending.add(pool.submit(task,i,docs[i*32:(i+1)*32]))
            finished,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for f in finished:f.result();done+=1
            if done%20==0:print('baseline blocks',done,'/',total,'seconds',time.monotonic()-start,flush=True)
    pieces=[]
    for i in range(total):
        with np.load(OUT/f'{i:05d}.npz') as z:pieces.append({k:z[k] for k in ('nll','correct','confidence','cell')})
    d={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};methods={}
    for j,name in enumerate(('canonical_raw','legal')):
        metrics=golden_data.aggregate(d['nll'][:,j],d['cell'],manifest)
        a=golden_data.aggregate(d['correct'][:,j],d['cell'],manifest)
        methods[name]=metrics|dict(macro_accuracy=a['macro'],expert_macro_accuracy=a['expert_macro'])
    methods['official_raw']={k:official[k] for k in ('macro','expert_macro','cells','counts')}
    report=dict(stage='Exact golden baselines; no search-method scores or parameter fitting',methods=training_cm.annotate(methods),
        elapsed_seconds=time.monotonic()-start,positions=len(d['cell']),plan=plan)
    tmp=OUT/'results.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'results.json')
    print(json.dumps({k:{m:v[m] for m in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm')} for k,v in report['methods'].items()},indent=2),flush=True)


if __name__=='__main__':main()
