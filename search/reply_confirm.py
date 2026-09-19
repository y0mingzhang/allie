"""Expand frozen one-ply/all-moves and two-ply human-reply choices on dev fold1."""
import concurrent.futures as cf
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp

from client import Oracle
import reply_pilot
import mcts_confirm

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
OUT=ROOT/'reply-confirmation'


def task(index,rows):
    dest=OUT/f'{index:05d}.npz'
    if dest.exists():return index
    chosen=json.loads((ROOT/'reply-pilot/selected.json').read_text())
    params=[chosen[k]['parameters'] for k in ('best_one_ply','selected')]
    with np.load(ROOT/f'confirmation/{index:05d}-root.logits.npz') as z:root=z['logits'].astype(np.float64)
    with np.load(ROOT/f'confirmation/scores-{index:05d}.npz') as z:
        assert np.array_equal(z['game'],[r['game'] for r in rows]) and np.array_equal(z['ply'],[r['ply'] for r in rows])
    q,stats=reply_pilot.batch(rows,Oracle());loss=[];correct=[];confidence=[]
    rv=softmax(root[:,2413:2416],axis=1)@np.array([1.,.5,0.]);move=root[:,378:2346]
    legal=np.isfinite(q['q1']);target=np.array([r['target']-378 for r in rows]);ar=np.arange(len(rows))
    for p in params:
        assert p['k']==1968 and p['gate']=='none' and not p['expert_only']
        val=(1-p['reply_mix'])*q['q1']+p['reply_mix']*q['q2']
        logits=np.where(legal,move/p['temperature']+p['beta']*(val-rv[:,None]),-np.inf)
        norm=logsumexp(logits,axis=1)
        loss.append(norm-logits[ar,target]);correct.append(logits.argmax(1)==target)
        confidence.append(np.exp(logits.max(1)-norm))
    tmp=dest.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,nll=np.stack(loss,1),correct=np.stack(correct,1),confidence=np.stack(confidence,1),
        q1=q['q1'],q2=q['q2'],
        expert=np.array([r['elo']>=2400 for r in rows]),game=np.array([r['game'] for r in rows]),ply=np.array([r['ply'] for r in rows]),
        stats=json.dumps(stats))
    tmp.replace(dest);gc.collect();return index


def main():
    OUT.mkdir(exist_ok=True);chosen=ROOT/'reply-pilot/selected.json'
    manifest=dict(selection_sha256=hashlib.sha256(chosen.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection=json.loads(chosen.read_text()),scope='all existing dev/dev_expert fold1; parameters frozen before this confirmation',
        generator_sha256=hashlib.sha256(Path(reply_pilot.__file__).read_bytes()).hexdigest(),
        checkpoint=Oracle().ready['checkpoint_sha256'])
    plan=OUT/'plan.json'
    if plan.exists():assert json.loads(plan.read_text())==manifest
    else:plan.write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.monotonic();done=0;iterator=iter(mcts_confirm.batches())
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        pending=set()
        for _ in range(8):
            item=next(iterator,None)
            if item is not None:pending.add(pool.submit(task,*item))
        while pending:
            completed,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for f in completed:
                index=f.result();done+=1
                if done%10==0:print('blocks',done,'last',index,'seconds',time.monotonic()-started,flush=True)
                item=next(iterator,None)
                if item is not None:pending.add(pool.submit(task,*item))
    parts=[];base=[]
    for path in sorted(OUT.glob('[0-9]*.npz')):
        with np.load(path) as z:parts.append({k:z[k] for k in ('nll','correct','confidence','expert','game','ply')})
        with np.load(ROOT/f'confirmation/scores-{int(path.stem):05d}.npz') as z:base.append(z['nll'][:,2])
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};baseline=np.concatenate(base);e=d['expert']
    _,inv=np.unique(d['game'],return_inverse=True);ng=inv.max()+1;counts=np.bincount(inv);ecount=np.bincount(inv,weights=e)
    w=np.random.default_rng(78731).poisson(1,(2000,ng));summary={}
    for j,name in enumerate(('oneply_all','reply2_all')):
        delta=d['nll'][:,j]-baseline
        boot=np.stack([w@np.bincount(inv,weights=delta)/(w@counts),w@np.bincount(inv,weights=delta*e)/(w@ecount)],1)
        summary[name]=dict(ce=float(d['nll'][:,j].mean()),expert_ce=float(d['nll'][e,j].mean()),
            accuracy=float(d['correct'][:,j].mean()),expert_accuracy=float(d['correct'][e,j].mean()),
            paired_delta_ce_vs_legal_95pct=np.quantile(boot,[.025,.975],axis=0).tolist())
    report=dict(stage='expanded unchanged development confirmation, not golden',positions=len(e),expert_positions=int(e.sum()),
        games=int(ng),methods=summary,this_invocation_seconds=time.monotonic()-started)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
