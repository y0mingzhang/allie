"""Expanded check of the preselected state-scaled utility, using cached q2."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp,softmax

from mcts_confirm import batches

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    selection=ROOT/'reply-pilot/utility-selection.json';data=json.loads(selection.read_text())
    name=data['selected'];assert name=='uncertainty_scaled';params=data['results'][name]
    manifest=dict(selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'reply-confirmation/utility-plan.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    summary=json.loads((ROOT/'reply-confirmation/results.json').read_text())
    loss=[];correct=[];base=[];expert=[];games=[]
    for i,rows in batches():
        with np.load(ROOT/f'reply-confirmation/{i:05d}.npz') as z:
            q=z['q2'].astype(float);expert.append(z['expert']);games.append(z['game'])
            assert np.array_equal(z['game'],[r['game'] for r in rows])
            assert np.array_equal(z['ply'],[r['ply'] for r in rows])
        with np.load(ROOT/f'confirmation/{i:05d}-root.logits.npz') as z:root=z['logits'].astype(float)
        with np.load(ROOT/f'confirmation/scores-{i:05d}.npz') as z:base.append(z['nll'][:,2])
        rv=softmax(root[:,2413:2416],axis=1)@np.array([1.,.5,0.])
        feature=q/np.maximum(.2,np.sqrt(rv*(1-rv)))[:,None]
        z=np.where(np.isfinite(q),params['alpha']*root[:,378:2346]+params['beta']*feature,-np.inf)
        y=np.array([r['target']-378 for r in rows]);ar=np.arange(len(rows))
        loss.append(logsumexp(z,axis=1)-z[ar,y]);correct.append(z.argmax(1)==y)
    loss=np.concatenate(loss);acc=np.concatenate(correct);e=np.concatenate(expert);game=np.concatenate(games);base=np.concatenate(base)
    assert len(loss)==summary['positions']
    _,inv=np.unique(game,return_inverse=True);ng=inv.max()+1;count=np.bincount(inv);ecount=np.bincount(inv,weights=e)
    w=np.random.default_rng(81131).poisson(1,(2000,ng));delta=loss-base
    boot=np.stack([w@np.bincount(inv,weights=delta)/(w@count),w@np.bincount(inv,weights=delta*e)/(w@ecount)],1)
    report=dict(stage='Expanded dev; utility family and coefficients fixed on pilot fit games',method=name,
        alpha=params['alpha'],beta=params['beta'],positions=len(loss),expert_positions=int(e.sum()),
        ce=float(loss.mean()),expert_ce=float(loss[e].mean()),accuracy=float(acc.mean()),expert_accuracy=float(acc[e].mean()),
        paired_delta_ce_vs_legal_95pct=np.quantile(boot,[.025,.975],axis=0).tolist(),plan=manifest)
    path=ROOT/'reply-confirmation/utility-results.json';tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
