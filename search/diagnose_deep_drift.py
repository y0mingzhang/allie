"""Locate cached/live numerical drift before using the search comparison."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .lapse_screen import ROOT,atomic,digest,means


def main():
    old=ROOT/'aug-deep-frontier-v1';new=ROOT/'aug-deep-frontier-live-v1'
    rows=json.loads((ROOT/'aug-deep-scale-v1/sample.json').read_text())['positions'];n=len(rows)
    with np.load(old/'scores.npz') as f:old_loss=dict(zip(f['names'],f['loss']));cells=f['cells'];games=f['games'];fm=f['fit']
    with np.load(new/'scores.npz') as f:new_loss=dict(zip(f['names'],f['loss']));np.testing.assert_array_equal(f['games'],games)
    k=max(len(r['legal']) for r in rows);mask=np.zeros((n,k),bool);ids=np.zeros((n,k),int);target=[]
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target.append(r['legal'].index(r['target']))
    target=np.array(target);ar=np.arange(n);zold=np.zeros((n,2432));qold=np.zeros((n,k))
    for path in sorted(old.glob('[0-9]*.npz')):
        with np.load(path) as f:lo=int(path.stem);hi=lo+len(f['game']);zold[lo:hi]=f['root'];qold[lo:hi]=f['q'][3]
    znew=np.zeros_like(zold);qnew=np.zeros_like(qold)
    for path in sorted((new/'fixed1000').glob('[0-9]*.npz')):
        with np.load(path) as f:lo=int(path.stem);hi=lo+len(f['game']);znew[lo:hi]=f['root'];qnew[lo:hi]=f['q']
    pold=softmax(np.where(mask,zold[:,378:2346][ar[:,None],ids],-np.inf),axis=1)
    pnew=softmax(np.where(mask,znew[:,378:2346][ar[:,None],ids],-np.inf),axis=1)
    root_ce=np.log(pold[ar,target])-np.log(pnew[ar,target]);delta=new_loss['fixed1000']-old_loss['fixed1000']
    root_policy=np.abs(pnew-pold).max(1);dq=np.abs(qnew-qold);qchange=np.where(mask,dq,0).max(1)
    take=(~fm)&(cells%4==3);order=np.flatnonzero(take);order=order[np.argsort(np.abs(delta[order]))[::-1]]
    top=[dict(index=int(i),game=games[i],ply=rows[i]['ply'],cell=int(cells[i]),delta_ce=float(delta[i]),root_delta_ce=float(root_ce[i]),root_max_policy_delta=float(root_policy[i]),max_q_delta=float(qchange[i])) for i in order[:20]]
    result=dict(scope='Diagnostic of already-scored reused development only, not a new tuned method. Ranking by targets is for error localization only.',
        source_sha256=digest(Path(__file__)),comparison='fixed1000 cached64-root batches versus live256-root batches on a new node of the sameGPUtype; multiple execution factors changed. A fixed-size order audit must isolate ordering.',
        raw_root_delta=dict(macro=float(means(root_ce[~fm],cells[~fm]).mean()),expert=float(means(root_ce[~fm],cells[~fm])[3::4].mean())),
        root_policy_delta_quantiles=np.quantile(root_policy,[.5,.9,.99,1]).tolist(),max_q_delta_quantiles=np.quantile(qchange,[.5,.9,.99,1]).tolist(),
        top_expert_moves=top,expert_positions=int(take.sum()),expert_game_count=len(set(games[take])),
        expert_signed_delta_sum=float(delta[take].sum()),expert_top5_signed_sum=float(delta[order[:5]].sum()),expert_top20_signed_sum=float(delta[order[:20]].sum()))
    atomic(new/'drift-diagnostic.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
