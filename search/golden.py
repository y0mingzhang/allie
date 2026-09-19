"""Frozen inference methods on the exact existing strat-eval-v1 arrays.

No new evaluation sample. Expert search is a declared inference allocation rule;
all other cells retain the legal policy. Freeze only after development confirmation.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import allie_mcts
from client import Oracle
import golden_data
import reply_pilot
import training_cm

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
OUT=ROOT/'golden-v1'
OFFICIAL=Path('/data/group_data/dei-group/yimingz3/allie/results/lm-eval/r2-3e16-control-t20-w20-pf052h-s42/strat-v1.json')
NAMES=['canonical_raw','legal','allie_adaptive_expert','reply_fixed_expert','reply_calibrated_expert']
FILES=['golden.py','golden_data.py','allie_mcts.py','reply_pilot.py','training_cm.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    assert not OUT.exists(),'A frozen golden study is never overwritten'
    baseline=json.loads((ROOT/'golden-baseline/results.json').read_text())
    assert baseline['positions']==1553058
    # Each compared family must first finish the expanded unchanged dev check.
    for name in ('mcts-confirmation','reply-confirmation'):
        result=json.loads((ROOT/name/'results.json').read_text())
        assert result['positions']==64366 and result['expert_positions']==11396
    calibrated=json.loads((ROOT/'reply-confirmation/calibrated-results.json').read_text())
    assert calibrated['positions']==64366 and calibrated['expert_positions']==11396
    chosen=json.loads((ROOT/'reply-pilot/selected.json').read_text())['selected']['parameters']
    assert chosen['reply_mix']==1 and chosen['k']==1968 and chosen['gate']=='none'
    cal=json.loads((ROOT/'reply-pilot/calibrated-selection.json').read_text())['parameters']['all']
    manifest=json.loads((golden_data.DATA/'manifest.json').read_text())
    official=json.loads(OFFICIAL.read_text());oracle=Oracle()
    assert oracle.ready['checkpoint_sha256']==official['model_sha256']
    assert official['strat_sha256']==manifest['sha256']
    plan=dict(methods=NAMES,scope='Search only mover >=2400; other 12 cells use the identical legal policy',
        selection='Methods and parameters frozen from development only. Report all methods; no golden-based retuning.',
        inference_coordinate_variant='Per-document canonical saved BF16 rotary coordinates, aligned128 packing',
        official_raw_path=str(OFFICIAL),official_raw_sha256=sha(OFFICIAL),
        checkpoint=official['model_sha256'],strat_sha256=manifest['sha256'],
        reply=dict(alpha=1/chosen['temperature'],beta=chosen['beta']),
        reply_calibrated=dict(alpha=cal['alpha'],beta=cal['beta']),
        adaptive=dict(mean_n_sims=50,reference=allie_mcts.REFERENCE),
        source_sha256={f:sha(Path(__file__).parent/f) for f in FILES},
        cm_law_sha256=sha(ROOT/'training-cm-laws.json'),
        baseline_plan_sha256=sha(ROOT/'golden-baseline/plan.json'),
        selection_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'reply-pilot/selected.json',ROOT/'reply-pilot/calibrated-selection.json',
            ROOT/'mcts-confirmation/results.json',ROOT/'reply-confirmation/results.json',
            ROOT/'reply-confirmation/calibrated-results.json']},
        games_per_block=32,notes=['Golden set has previously been used by the training/data research tracks.',
            'This is the first search-track comparison on it, not a globally untouched final test.',
            'Training-equivalent CM is conditional on a transferred fitted law, not measured additional training.'])
    OUT.mkdir()
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('Frozen golden protocol; no model scoring performed',flush=True)


def validate():
    plan=json.loads((OUT/'plan.json').read_text())
    for f,h in plan['source_sha256'].items():assert sha(Path(__file__).parent/f)==h,f
    assert sha(OFFICIAL)==plan['official_raw_sha256']
    assert sha(ROOT/'training-cm-laws.json')==plan['cm_law_sha256']
    assert sha(ROOT/'golden-baseline/plan.json')==plan['baseline_plan_sha256']
    for p,h in plan['selection_sha256'].items():assert sha(ROOT/p)==h,p
    assert Oracle().ready['checkpoint_sha256']==plan['checkpoint']
    return plan


def task(index,docs,plan):
    dest=OUT/f'{index:05d}.npz'
    if dest.exists():return index
    started=time.monotonic();oracle=Oracle();positions=[];indices=[];offset=0
    for doc in docs:
        for p in golden_data.positions(doc):
            positions.append(p);indices.append(offset+p['root_index'])
        offset+=len(doc['prefix'])
    # Causality was verified against individually truncated prefixes on dev.
    # Taking all logits in one causal pass avoids recomputing each game's past.
    root_path=OUT/f'{index:05d}-roots.npz'
    if not root_path.exists():
        shared=ROOT/f'golden-baseline/{index:05d}-roots.npz'
        if shared.exists():root_path=shared
    if root_path.exists():
        with np.load(root_path) as z:
            root=z['logits']
            assert np.array_equal(z['game'],[p['game'] for p in positions])
            assert np.array_equal(z['ply'],[p['ply'] for p in positions])
    else:
        all_logits=oracle([d['prefix'] for d in docs],all_tokens=True,compact=True)
        assert len(all_logits)==offset
        root=all_logits[np.array(indices,dtype=int)]
        tmp=root_path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez(f,logits=root,game=np.array([p['game'] for p in positions]),
                                      ply=np.array([p['ply'] for p in positions]))
        tmp.replace(root_path)
    n=len(positions)
    if n==0:raise AssertionError('Unexpected empty golden block')
    move=root[:,378:2346].astype(np.float64);target=np.array([p['target']-378 for p in positions]);ar=np.arange(n)
    cell=np.array([p['cell'] for p in positions]);expert=cell%4==3;which=np.flatnonzero(expert)
    legal=np.zeros_like(move,dtype=bool)
    for i,p in enumerate(positions):legal[i,np.array(p['legal'])-378]=True
    raw_lp=move-logsumexp(move,axis=1,keepdims=True)
    legal_logits=np.where(legal,move,-np.inf);legal_lp=legal_logits-logsumexp(legal_logits,axis=1,keepdims=True)
    def measures(lp):
        return -lp[ar,target],lp.argmax(1)==target,np.exp(lp.max(1))
    raw=measures(raw_lp);base=measures(legal_lp)
    loss=[raw[0],base[0]];correct=[raw[1],base[1]];confidence=[raw[2],base[2]];stats={}
    policies=[]
    if len(which):
        erows=[positions[i] for i in which]
        pred,stats['allie']=allie_mcts.run(erows,root[which],oracle,adaptive=True,mean_n_sims=50)
        policies.append(np.log(pred['policy'].astype(np.float64),where=pred['policy']>0,
                               out=np.full(pred['policy'].shape,-np.inf)))
        q,stats['reply']=reply_pilot.batch(erows,oracle)
        for key in ('reply','reply_calibrated'):
            par=plan[key];x=np.where(legal[which],par['alpha']*move[which]+par['beta']*q['q2'],-np.inf)
            policies.append(x-logsumexp(x,axis=1,keepdims=True))
    else:policies=[None]*3
    for ep in policies:
        lp=legal_lp.copy()
        if len(which):lp[which]=ep
        a,b,c=measures(lp);loss.append(a);correct.append(b);confidence.append(c)
    assert np.isfinite(loss).all()
    tmp=dest.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,nll=np.stack(loss,1),correct=np.stack(correct,1),
        confidence=np.stack(confidence,1),cell=cell,game=np.array([p['game'] for p in positions]),
        ply=np.array([p['ply'] for p in positions]),stats=json.dumps(stats),seconds=time.monotonic()-started)
    tmp.replace(dest)
    return index


def summarize(manifest,expected_blocks):
    paths=sorted(OUT.glob('[0-9][0-9][0-9][0-9][0-9].npz'))
    assert len(paths)==expected_blocks,'Do not summarize a partial golden evaluation'
    parts=[];cost=[]
    for p in paths:
        with np.load(p) as z:
            parts.append({k:z[k] for k in ('nll','correct','confidence','cell','game')})
            cost.append(json.loads(str(z['stats'])))
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};cell=data['cell']
    assert np.array_equal(np.bincount(cell,minlength=16),manifest['scored_moves'])
    methods={}
    for j,name in enumerate(NAMES):
        metrics=golden_data.aggregate(data['nll'][:,j],cell,manifest)
        accuracy=golden_data.aggregate(data['correct'][:,j],cell,manifest)
        metrics.update(macro_accuracy=accuracy['macro'],expert_macro_accuracy=accuracy['expert_macro'])
        # Equal-cell weighted calibration, not the original population mix.
        weight=1/(16*np.bincount(cell,minlength=16)[cell]);bins=np.minimum((data['confidence'][:,j]*15).astype(int),14)
        metrics['macro_ece_15bins']=float(sum(abs(np.sum(weight[bins==b]*(data['correct'][bins==b,j]-data['confidence'][bins==b,j]))) for b in range(15)))
        methods[name]=metrics
    official=json.loads(OFFICIAL.read_text())
    methods['official_raw']={k:official[k] for k in ('macro','expert_macro','cells','counts')}
    methods=training_cm.annotate(methods)
    # Paired bootstrap by complete game, preserving macro aggregation each draw.
    from scipy.sparse import coo_matrix
    _,inv=np.unique(data['game'],return_inverse=True);ng=int(inv.max())+1
    counts=coo_matrix((np.ones(len(cell)),(inv,cell)),shape=(ng,16)).tocsr()
    sums=[coo_matrix((data['nll'][:,j]-data['nll'][:,1],(inv,cell)),shape=(ng,16)).tocsr() for j in range(len(NAMES))]
    rng=np.random.default_rng(441093);draws=[[] for _ in NAMES]
    for _ in range(16):
        w=rng.poisson(1,(64,ng));den=w@counts
        assert (den>0).all()
        for j,s in enumerate(sums):
            means=(w@s)/den;draws[j].append(np.stack([means.mean(1),means[:,3::4].mean(1)],axis=1))
    for j,name in enumerate(NAMES):
        boot=np.concatenate(draws[j]);methods[name]['paired_ce_delta_vs_legal_95pct']=np.quantile(boot,[.025,.975],axis=0).tolist()
    report=dict(stage='Exact golden evaluation of frozen expert-only search policies',positions=len(cell),games=ng,
        methods=methods,scope='Nonexpert cells use legal policy unchanged for every search method',
        uncertainty='Game bootstrap describes eval sampling, not uncertainty in the transferred scaling law',
        costs=cost,plan_sha256=sha(OUT/'plan.json'))
    tmp=OUT/'results.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'results.json')
    print(json.dumps({k:{m:v[m] for m in ('macro','expert_macro','macro_training_eq_cm','expert_macro_training_eq_cm')} for k,v in methods.items()},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.freeze:freeze();return
    plan=validate();rows,labels,manifest=golden_data.load();assert manifest['sha256']==plan['strat_sha256']
    docs=list(golden_data.games(rows,labels));batch=plan['games_per_block'];total=(len(docs)+batch-1)//batch
    started=time.monotonic();done=0;next_block=0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=set()
        while next_block<total or pending:
            while next_block<total and len(pending)<2*args.workers:
                assert not (ROOT/'STOP').exists() and not Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP').exists()
                i=next_block;next_block+=1
                pending.add(pool.submit(task,i,docs[i*batch:(i+1)*batch],plan))
            finished,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for f in finished:
                f.result();done+=1
            if done%10==0:print('golden blocks',done,'of',total,'seconds',time.monotonic()-started,flush=True)
    summarize(manifest,total)


if __name__=='__main__':main()
