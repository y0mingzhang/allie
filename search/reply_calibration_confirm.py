"""Evaluate previously frozen calibration using expanded reply prediction caches."""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    summary=json.loads((ROOT/'reply-confirmation/results.json').read_text())
    params=json.loads((ROOT/'reply-pilot/calibrated-selection.json').read_text())['parameters']['all']
    from mcts_confirm import batches
    correct=[];base=[];expert=[];games=[];values=[]
    for i,rows in batches():
        path=ROOT/f'reply-confirmation/{i:05d}.npz'
        with np.load(path) as z:
            q=z['q2'];expert.append(z['expert']);games.append(z['game'])
            assert np.array_equal(z['game'],[r['game'] for r in rows])
            assert np.array_equal(z['ply'],[r['ply'] for r in rows])
        with np.load(ROOT/f'confirmation/{i:05d}-root.logits.npz') as z:root=z['logits'][:,378:2346].astype(np.float64)
        with np.load(ROOT/f'confirmation/scores-{i:05d}.npz') as z:base.append(z['nll'][:,2])
        legal=np.isfinite(q);x=np.where(legal,params['alpha']*root+params['beta']*q,-np.inf)
        y=np.array([r['target']-378 for r in rows]);ar=np.arange(len(rows))
        values.append(logsumexp(x,axis=1)-x[ar,y]);correct.append(x.argmax(1)==y)
    loss=np.concatenate(values);acc=np.concatenate(correct);e=np.concatenate(expert);game=np.concatenate(games);base=np.concatenate(base)
    assert len(loss)==summary['positions']
    _,inv=np.unique(game,return_inverse=True);ng=inv.max()+1;count=np.bincount(inv);ecount=np.bincount(inv,weights=e)
    w=np.random.default_rng(81131).poisson(1,(2000,ng));delta=loss-base
    boot=np.stack([w@np.bincount(inv,weights=delta)/(w@count),w@np.bincount(inv,weights=delta*e)/(w@ecount)],1)
    report=dict(stage='Expanded development check of coefficients frozen before this check; not golden',parameters=params,
        positions=len(loss),expert_positions=int(e.sum()),ce=float(loss.mean()),expert_ce=float(loss[e].mean()),
        accuracy=float(acc.mean()),expert_accuracy=float(acc[e].mean()),
        paired_delta_ce_vs_legal_95pct=np.quantile(boot,[.025,.975],axis=0).tolist())
    path=ROOT/'reply-confirmation/calibrated-results.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
