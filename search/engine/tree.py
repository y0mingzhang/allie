"""Batched human-policy continuation expectation with native chess rules."""
import time
import numpy as np
from scipy.special import softmax
from .native_board import from_prefix


def batch(rows,oracle,widths=(4,2,2),batch_size=1024):
    start=time.monotonic();roots=[from_prefix(r['prefix']) for r in rows]
    root_logits=oracle([r['prefix'] for r in rows])
    white=[b.white for b in roots];q=np.full((len(widths)+1,len(rows),1968),np.nan,np.float64)
    active=[];leaves=[];calls=1;limited=0
    nodes_by_root=[]
    for i,(row,board) in enumerate(zip(rows,roots)):
        for token in board.legal():
            child=board.child(token);terminal=child.outcome()
            if terminal>=0:q[0,i,token-378]=terminal if white[i] else 1-terminal
            else:active.append((i,token-378,row['prefix']+[token],child,1.,0.))
    terminal_delta=np.zeros((len(rows),1968),np.float64)
    for depth in range(1,len(widths)+2):
        if depth>1:q[depth-1]=q[depth-2]+terminal_delta
        leaves.append(len(active));next_nodes=[]
        nodes_by_root.append(np.bincount([n[0] for n in active],minlength=len(rows)).tolist())
        last=depth==len(widths)+1
        for lo in range(0,len(active),batch_size):
            nodes=active[lo:lo+batch_size];prefixes=[n[2] for n in nodes]
            preds=oracle(prefixes,columns=[2413,2414,2415] if last else None);calls+=1
            wdl=preds if last else preds[:,2413:2416]
            value=softmax(wdl.astype(np.float64),axis=1)@np.array([1.,.5,0.])
            if depth%2:value=1-value
            if not last:
                legal=[n[3].legal() for n in nodes]
                indices=np.zeros((len(nodes),max(map(len,legal))),np.int32)
                mask=np.zeros_like(indices,bool)
                for j,tokens in enumerate(legal):indices[j,:len(tokens)]=tokens;mask[j,:len(tokens)]=True
                probs=softmax(np.where(mask,preds[np.arange(len(nodes))[:,None],indices],-np.inf).astype(np.float64),axis=1)
            for j,((i,a,prefix,board,mass,parent),v) in enumerate(zip(nodes,value)):
                if depth==1:q[0,i,a]=v
                else:q[depth-1,i,a]+=mass*(v-parent)
                if last:continue
                if len(prefix)>=1025:limited+=1;continue
                p=probs[j,:len(legal[j])]
                for ix in np.argsort(p)[::-1][:widths[depth-1]]:
                    token=legal[j][ix];child=board.child(token);terminal=child.outcome()
                    if terminal>=0 and not white[i]:terminal=1-terminal
                    next_nodes.append((i,a,prefix+[token],child,mass*float(p[ix]),float(v),terminal))
        if not last:
            terminal_delta=np.zeros((len(rows),1968),np.float64);active=[]
            for i,a,prefix,board,mass,parent,terminal in next_nodes:
                if terminal<0:active.append((i,a,prefix,board,mass,parent))
                else:terminal_delta[i,a]+=mass*(terminal-parent)
    for i,row in enumerate(rows):assert np.isfinite(q[:,i,np.array(row['legal'])-378]).all()
    assert np.nanmin(q)>-1e-10 and np.nanmax(q)<1+1e-10
    return q.astype(np.float32),root_logits,dict(seconds=time.monotonic()-start,requests=calls,
        leaves_by_depth=leaves,context_limited=limited,nodes_by_depth_and_root=nodes_by_root)
