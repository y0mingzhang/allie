"""Frozen Allie baselines on all existing fold1 development positions, four CPU workers."""
import concurrent.futures as cf
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np

import confirm
from allie_mcts import run, REFERENCE
from client import Oracle

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
OUT=ROOT/'mcts-confirmation'
METHODS=dict(fixed52=dict(n_sims=52),adaptive50=dict(adaptive=True,mean_n_sims=50))


def task(index,rows):
    dest=OUT/f'{index:05d}.npz'
    if dest.exists():return index,'cached'
    with np.load(ROOT/f'confirmation/{index:05d}-root.logits.npz') as z:root=z['logits']
    with np.load(ROOT/f'confirmation/scores-{index:05d}.npz') as z:
        assert np.array_equal(z['game'],[p['game'] for p in rows])
        assert np.array_equal(z['ply'],[p['ply'] for p in rows])
    target=np.array([p['target']-378 for p in rows]);ar=np.arange(len(rows));loss=[];correct=[];confidence=[];sims=[];stats={}
    oracle=Oracle()
    for name,params in METHODS.items():
        out,st=run(rows,root,oracle,**params);p=out['policy']
        loss.append(-np.log(p[ar,target].astype(np.float64)));correct.append(p.argmax(1)==target)
        confidence.append(p.max(1));sims.append(out['simulations']);stats[name]=st
        del out;gc.collect()
    tmp=dest.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez(f,nll=np.stack(loss,1),correct=np.stack(correct,1),confidence=np.stack(confidence,1),
        simulations=np.stack(sims,1),expert=np.array([p['elo']>=2400 for p in rows]),
        game=np.array([p['game'] for p in rows]),ply=np.array([p['ply'] for p in rows]),stats=json.dumps(stats))
    tmp.replace(dest)
    return index,stats


def batches():
    rows=[];index=0
    for row in confirm.positions():
        rows.append(row)
        if len(rows)==128:
            yield index,rows;index+=1;rows=[]
    if rows:yield index,rows


def summarize():
    parts=[];old=[];stats={name:dict(seconds=0.,simulations=0,evaluated_leaves=0,requests=0,useful_prefix_tokens=0) for name in METHODS}
    for path in sorted(OUT.glob('[0-9]*.npz')):
        with np.load(path) as z:
            parts.append({k:z[k] for k in ('nll','correct','confidence','expert','game','ply')})
            for name,st in json.loads(str(z['stats'])).items():
                for k in stats[name]:stats[name][k]+=st[k]
        with np.load(ROOT/f'confirmation/scores-{int(path.stem):05d}.npz') as z:old.append(z['nll'][:,2])
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};baseline=np.concatenate(old)
    expert=data['expert'];names=list(METHODS);_,inv=np.unique(data['game'],return_inverse=True);ng=inv.max()+1
    report=dict(stage='Unchanged methods on existing development confirmation; not golden',positions=len(expert),
        expert_positions=int(expert.sum()),games=int(ng),methods={},cost=stats)
    # Aggregate by game before bootstrap: more efficient than per-move weights.
    counts=np.bincount(inv);ecounts=np.bincount(inv,weights=expert)
    rng=np.random.default_rng(738311);weights=rng.poisson(1,(2000,ng))
    for j,name in enumerate(names):
        delta=data['nll'][:,j]-baseline
        sums=np.bincount(inv,weights=delta);esums=np.bincount(inv,weights=delta*expert)
        boot=np.stack([weights@sums/(weights@counts),weights@esums/(weights@ecounts)],1)
        report['methods'][name]=dict(ce=float(data['nll'][:,j].mean()),expert_ce=float(data['nll'][expert,j].mean()),
            accuracy=float(data['correct'][:,j].mean()),expert_accuracy=float(data['correct'][expert,j].mean()),
            paired_delta_ce_vs_legal_95pct=np.quantile(boot,[.025,.975],axis=0).tolist())
    return report


def main():
    OUT.mkdir(exist_ok=True)
    manifest=dict(reference=REFERENCE,methods=METHODS,scope='existing prepared dev/dev_expert fold1, all positions',
        base_confirmation_plan_sha256=hashlib.sha256((ROOT/'confirmation/plan.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256((Path(__file__).parent/'allie_mcts.py').read_bytes()).hexdigest(),
        checkpoint=Oracle().ready['checkpoint_sha256'])
    plan=OUT/'plan.json'
    if plan.exists():assert json.loads(plan.read_text())==manifest
    else:plan.write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.monotonic();done=0;iterator=iter(batches())
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        pending=set()
        for _ in range(8):
            item=next(iterator,None)
            if item is not None:pending.add(pool.submit(task,*item))
        while pending:
            completed,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for f in completed:
                index,st=f.result();done+=1
                if done%10==0:print('blocks',done,'last',index,'seconds',time.monotonic()-started,flush=True)
                item=next(iterator,None)
                if item is not None:pending.add(pool.submit(task,*item))
    result=summarize();result['this_invocation_seconds']=time.monotonic()-started
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
