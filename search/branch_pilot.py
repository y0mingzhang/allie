"""Future-legality and opponent-response features from the same frozen model.

Conditioning on the next move being legal gives q(a) proportional to
p(a)*sum_{b legal after a} p(b|history,a). This is a structural observation,
not an observed future human move. Also cache response entropy for an ablation.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax

from allie_mcts import MOVE_ID,clone_board
from client import Oracle
from reply_pilot import board_for,outcome_value

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
OUT=ROOT/'branch-pilot'


def batch(rows,oracle):
    # log future legal mass, entropy of legal reply distribution, child WDL value
    features=np.full((len(rows),1968,3),np.nan,np.float32);pending=[];prefixes=[];boards=[]
    for i,r in enumerate(rows):
        board=board_for(r['prefix'])
        for move in board.legal_moves:
            t=MOVE_ID[move.uci()];child=clone_board(board);child.push(move)
            terminal=outcome_value(child,board.turn)
            if terminal is not None:features[i,t-378]=[0.,0.,terminal]
            else:pending.append((i,t));prefixes.append(r['prefix']+[t]);boards.append(child)
    started=time.monotonic();tokens=0
    for lo in range(0,len(prefixes),256):
        group=prefixes[lo:lo+256];pred=oracle(group);tokens+=sum(map(len,group))
        for (i,t),board,logits in zip(pending[lo:lo+256],boards[lo:lo+256],pred):
            legal=np.array([MOVE_ID[m.uci()] for m in board.legal_moves]);x=logits.astype(np.float64)
            lz=logsumexp(x[legal]);logmass=lz-logsumexp(x[378:2346]);lp=x[legal]-lz
            entropy=-np.sum(np.exp(lp)*lp);value=1-softmax(x[2413:2416])@np.array([1.,.5,0.])
            features[i,t-378]=[logmass,entropy,value]
    for i,r in enumerate(rows):assert np.isfinite(features[i,np.array(r['legal'])-378]).all()
    return features,dict(seconds=time.monotonic()-started,leaves=len(prefixes),useful_prefix_tokens=tokens)


def main():
    OUT.mkdir(exist_ok=True);data=ROOT/'dev.json';rows=json.loads(data.read_text())['positions'];n=len(rows);oracle=Oracle()
    plan=dict(data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),checkpoint=oracle.ready['checkpoint_sha256'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=['log_future_legal_mass','reply_entropy','root_side_child_value'],
        scope='Existing pilot fit fold0 only; root target unavailable to feature inference')
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2)+'\n')
    arrays=[]
    for lo in range(0,n,64):
        dest=OUT/f'{lo:05d}.npz'
        if not dest.exists():
            feats,stats=batch(rows[lo:lo+64],oracle);tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,features=feats,stats=json.dumps(stats))
            tmp.replace(dest);print(lo+len(feats),stats,flush=True)
        with np.load(dest) as z:arrays.append(z['features'])
    features=np.concatenate(arrays).astype(np.float64);legal=np.isfinite(features[:,:,0]);features=np.nan_to_num(features)
    move=np.concatenate([np.load(ROOT/f'cache-{i:05d}.npz')['root'][:,378:2346] for i in range(0,n,128)]).astype(np.float64)
    q2=np.concatenate([np.load(ROOT/f'reply-pilot/{i:05d}.npz')['q2'] for i in range(0,n,64)])
    q2=np.nan_to_num(q2).astype(np.float64)
    target=np.array([r['target']-378 for r in rows]);ar=np.arange(n);fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
    result={}
    def score(x):
        x=np.where(legal,x,-np.inf);loss=logsumexp(x,axis=1)-x[ar,target];correct=x.argmax(1)==target
        return {name:dict(ce=float(loss[m].mean()),expert_ce=float(loss[m&expert].mean()),
                         accuracy=float(correct[m].mean()),expert_accuracy=float(correct[m&expert].mean())) for name,m in [('fit',fit),('confirmation',~fit)]}
    result['legal']=score(move)
    result['exact_future_legality']=score(move+features[:,:,0])
    # Fit a small, predeclared set of stackable features using the all-player
    # fit CE. Bounded coefficients limit a near-zero illegal-mass feature's leverage.
    base=np.stack([move,q2],axis=-1)
    for name,extra in [('legality',features[:,:,0:1]),('reply_entropy',features[:,:,1:2]),('both',features[:,:,:2])]:
        x=np.concatenate([base,extra],axis=-1);xf=x[fit];mask=legal[fit];y=target[fit];a=np.arange(len(y))
        bounds=[(.5,2.),(0.,32.)]+([(0.,20.)] if name=='legality' else [(-1.,1.)] if name=='reply_entropy' else [(0.,20.),(-1.,1.)])
        def objective(w):
            z=np.where(mask,np.einsum('nav,v->na',xf,w),-np.inf);norm=logsumexp(z,axis=1);p=np.exp(z-norm[:,None])
            return float(np.mean(norm-z[a,y])),np.einsum('na,nav->v',p,xf)/len(y)-xf[a,y].mean(0)
        init=[.87,7.3]+[0.]*(x.shape[-1]-2)
        opt=minimize(objective,init,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-12,gtol=1e-8,maxiter=200))
        assert opt.success,opt.message
        result[name]=dict(coefficients=opt.x.tolist(),bounds=bounds,metrics=score(np.einsum('nav,v->na',x,opt.x)))
    report=dict(stage='Development pilot, no golden CM claim',results=result)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
