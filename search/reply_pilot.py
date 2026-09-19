"""One-ply values for all legal moves and policy-expectation over top-four replies.

Tail replies retain the child value. This is human-policy expectation, not minimax.
Only development roots from the original pilot are used; no golden scoring here.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import chess
import numpy as np
from scipy.special import softmax
from allie_mcts import MOVES, MOVE_ID, clone_board
from client import Oracle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/search-v1/reply-pilot'


def board_for(prefix):
    b=chess.Board()
    for t in prefix[11:]: b.push(chess.Move.from_uci(MOVES[t-378]))
    return b


def outcome_value(board, root_side):
    out=board.outcome(claim_draw=False)
    if out is None:return None
    return .5 if out.winner is None else float(out.winner==root_side)


def batch(rows, oracle):
    q1=np.full((len(rows),1968),np.nan,np.float32);q2=q1.copy();coverage=np.zeros_like(q1)
    where=[];prefixes=[];boards=[];started=time.monotonic();calls=0;tokens=0
    for i,p in enumerate(rows):
        root=board_for(p['prefix'])
        for move in root.legal_moves:
            t=MOVE_ID[move.uci()];child=clone_board(root);child.push(move)
            terminal=outcome_value(child,root.turn)
            if terminal is not None:
                q1[i,t-378]=q2[i,t-378]=terminal
            else:
                where.append((i,t));prefixes.append(p['prefix']+[t]);boards.append(child)
    predictions=[]
    for start in range(0,len(prefixes),512):
        group=prefixes[start:start+512];predictions.extend(oracle(group));calls+=1;tokens+=sum(map(len,group))
    grand_prefix=[];grand_where=[]
    for (i,t),prefix,board,logits in zip(where,prefixes,boards,predictions):
        value=1-softmax(logits[2413:2416].astype(np.float64))@np.array([1.,.5,0.])
        q1[i,t-378]=q2[i,t-378]=value
        legal=np.array([MOVE_ID[m.uci()] for m in board.legal_moves]);probs=softmax(logits[legal].astype(np.float64))
        for j in np.argsort(probs)[::-1][:4]:
            move=int(legal[j]);weight=float(probs[j]);gboard=clone_board(board)
            gboard.push(chess.Move.from_uci(MOVES[move-378]));coverage[i,t-378]+=weight
            terminal=outcome_value(gboard,gboard.turn)
            if terminal is not None:
                q2[i,t-378]+=weight*(terminal-value)
            else:
                grand_prefix.append(prefix+[move]);grand_where.append((i,t,weight,value))
    for start in range(0,len(grand_prefix),512):
        group=grand_prefix[start:start+512];pred=oracle(group,columns=[2413,2414,2415]);calls+=1;tokens+=sum(map(len,group))
        vals=softmax(pred.astype(np.float64),axis=-1)@np.array([1.,.5,0.])
        for (i,t,weight,v1),v2 in zip(grand_where[start:start+512],vals):q2[i,t-378]+=weight*(v2-v1)
    for i,p in enumerate(rows):
        assert np.isfinite(q1[i,np.array(p['legal'])-378]).all()
        assert np.isfinite(q2[i,np.array(p['legal'])-378]).all()
    assert np.nanmin(q2)>=0 and np.nanmax(q2)<=1
    return dict(q1=q1,q2=q2,reply_coverage=coverage),dict(seconds=time.monotonic()-started,requests=calls,
        evaluated_leaves=len(prefixes)+len(grand_prefix),useful_prefix_tokens=tokens)


def main():
    OUT.mkdir(exist_ok=True);oracle=Oracle();path=ROOT/'results/search-v1/dev.json'
    rows=json.loads(path.read_text())['positions']
    manifest=dict(data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint=oracle.ready['checkpoint_sha256'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),replies=4,tail='root-side value at the child')
    plan=OUT/'plan.json'
    if plan.exists():assert json.loads(plan.read_text())==manifest
    else:plan.write_text(json.dumps(manifest,indent=2)+'\n')
    for lo in range(0,len(rows),64):
        dest=OUT/f'{lo:05d}.npz'
        if dest.exists():continue
        result,stats=batch(rows[lo:lo+64],oracle);tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez(f,**result,stats=json.dumps(stats))
        tmp.replace(dest);print(lo+len(result['q1']),stats,flush=True);gc.collect()


if __name__=='__main__':main()
