"""Four-ply truncated human-policy expectation, with value-preserving tails.

Every legal root action is evaluated. Later widths are 4, 2, 2. Unexpanded
probability retains that node's value; terminal outcomes override predictions.
The value and policy come only from the frozen model, never human future moves.
"""
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path

import chess
import numpy as np
from scipy.special import softmax

from allie_mcts import MOVES, MOVE_ID, clone_board
from client import Oracle
from reply_pilot import board_for, outcome_value

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
OUT=ROOT/'deeper-pilot'
WIDTHS=(4,2,2)


def batch(rows,oracle,widths=WIDTHS):
    started=time.monotonic();requests=0;tokens=0;leaves=[];limited=0
    q=np.full((len(widths)+1,len(rows),1968),np.nan,np.float64)
    active=[]
    for i,row in enumerate(rows):
        board=board_for(row['prefix'])
        for move in board.legal_moves:
            token=MOVE_ID[move.uci()];child=clone_board(board);child.push(move)
            terminal=outcome_value(child,board.turn)
            if terminal is not None:q[0,i,token-378]=terminal
            else:active.append(dict(i=i,a=token-378,prefix=row['prefix']+[token],board=child,mass=1.,parent=0.))
    terminal_delta=np.zeros((len(rows),1968),np.float64)
    for depth in range(1,len(widths)+2):
        if depth>1:q[depth-1]=q[depth-2]+terminal_delta
        leaves.append(len(active))
        next_nodes=[]
        for start in range(0,len(active),256):
            group=active[start:start+256];prefixes=[n['prefix'] for n in group]
            assert all(len(p)<=1025 for p in prefixes)
            preds=oracle(prefixes);requests+=1;tokens+=sum(map(len,prefixes))
            values=softmax(preds[:,2413:2416].astype(np.float64),axis=1)@np.array([1.,.5,0.])
            if depth%2:values=1-values
            for node,logits,value in zip(group,preds,values):
                i,a,mass=node['i'],node['a'],node['mass']
                if depth==1:q[0,i,a]=value
                else:q[depth-1,i,a]+=mass*(value-node['parent'])
                if depth==len(widths)+1:continue
                if len(node['prefix'])>=1025:
                    limited+=1;continue
                legal=np.array([MOVE_ID[m.uci()] for m in node['board'].legal_moves])
                probs=softmax(logits[legal].astype(np.float64))
                for ix in np.argsort(probs)[::-1][:widths[depth-1]]:
                    token=int(legal[ix]);weight=mass*float(probs[ix]);child=clone_board(node['board'])
                    child.push(chess.Move.from_uci(MOVES[token-378]))
                    # Root side follows directly from root prefix move parity.
                    root_side=chess.WHITE if (len(rows[i]['prefix'])-11)%2==0 else chess.BLACK
                    terminal=outcome_value(child,root_side)
                    next_nodes.append(dict(i=i,a=a,prefix=node['prefix']+[token],board=child,
                                           mass=weight,parent=float(value),terminal=terminal))
        # A terminal contributes at its first horizon and remains through later
        # horizons, with no additional model call or expansion.
        if depth<len(widths)+1:
            terminal_delta=np.zeros((len(rows),1968),np.float64)
            active=[]
            for node in next_nodes:
                terminal=node.pop('terminal')
                if terminal is None:active.append(node)
                else:terminal_delta[node['i'],node['a']]+=node['mass']*(terminal-node['parent'])
    for i,row in enumerate(rows):
        legal=np.array(row['legal'])-378
        assert np.isfinite(q[:,i,legal]).all()
    assert np.nanmin(q)>-1e-10 and np.nanmax(q)<1+1e-10
    return q.astype(np.float32),dict(seconds=time.monotonic()-started,requests=requests,
        useful_prefix_tokens=tokens,leaves_by_depth=leaves,context_limited=limited)


def task(lo,rows):
    path=OUT/f'{lo:05d}.npz'
    if path.exists():return lo
    q,stats=batch(rows,Oracle());tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,q=q,stats=json.dumps(stats))
    tmp.replace(path)
    return lo


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();rows=json.loads((ROOT/'dev.json').read_text())['positions']
    plan=dict(widths=WIDTHS,checkpoint=Oracle().ready['checkpoint_sha256'],
        data_sha256=hashlib.sha256((ROOT/'dev.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tail='retain the current node value for all unexpanded replies',batch=16)
    plan=json.loads(json.dumps(plan));OUT.mkdir(exist_ok=True);path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2)+'\n')
    started=time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(task,i,rows[i:i+16]) for i in range(0,len(rows),16)]
        for done,future in enumerate(cf.as_completed(futures),1):
            future.result()
            if done%8==0:print('deeper blocks',done,'/',len(futures),'seconds',time.monotonic()-started,flush=True)


if __name__=='__main__':main()
