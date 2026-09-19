"""Released Allie MCTS with native board/history handling and cached inference.

Simulations remain sequential within every tree. Selection, backup, time budgets,
and regularized output reuse the audited reference adapter without modification.
"""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .native_board import from_prefix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import allie_mcts as reference


class Node(reference.Node):
    def materialize(self):
        if self.board is None:
            self.board=self.parent.board.child(self.move)
            self.prefix=self.parent.prefix+[self.move]
        return self


def terminal_value(node):
    result=node.board.outcome()
    if result<0:return None
    if result==.5:return 0.
    # Only checkmate is a decisive rules-only terminal (never resignation).
    assert (result==1)!=node.board.white
    return 1.


def expand(node,logits):
    legal=node.board.legal();assert legal
    policy=softmax(np.asarray(logits,np.float64)[legal])
    node.children=[Node(float(p),node,m) for m,p in zip(legal,policy)]
    wdl=softmax(np.asarray(logits,np.float64)[2413:2416])
    return float(wdl[2]-wdl[0])


def run(rows,root_logits,oracle,*,adaptive=False,n_sims=50,mean_n_sims=50):
    start=time.monotonic();roots=[]
    for row,logits in zip(rows,root_logits):
        root=Node(board=from_prefix(row['prefix']),prefix=list(row['prefix']))
        assert terminal_value(root) is None
        expand(root,logits);roots.append(root)
    ns=reference.budgets(root_logits,mean_n_sims) if adaptive else np.full(len(rows),n_sims,int)
    cp=1.25*np.sqrt(mean_n_sims/np.maximum(ns,1)) if adaptive else np.full(len(rows),1.25)
    stats=dict(simulations=int(ns.sum()),evaluated_leaves=0,terminal_visits=0,
        useful_prefix_tokens=0,requests=0,max_depth=0)
    for iteration in range(int(ns.max(initial=0))):
        paths=[]
        for i in np.flatnonzero(ns>iteration):
            path=reference.select_path(roots[i],float(cp[i]),min(100,1025-len(roots[i].prefix)))
            stats['max_depth']=max(stats['max_depth'],len(path)-1)
            value=terminal_value(path[-1])
            if value is None:paths.append(path)
            else:reference.backup(path,value);stats['terminal_visits']+=1
        if not paths:continue
        prefixes=[p[-1].prefix for p in paths];assert max(map(len,prefixes))<=1025
        predictions=oracle(prefixes);stats['requests']+=1;stats['evaluated_leaves']+=len(paths)
        stats['useful_prefix_tokens']+=sum(map(len,prefixes))
        for path,logits in zip(paths,predictions):reference.backup(path,expand(path[-1],logits))
    out=np.zeros((len(rows),1968),np.float32);visits=np.zeros_like(out,dtype=np.int16);values=np.zeros_like(out)
    for i,root in enumerate(roots):
        ids=np.array([c.move-378 for c in root.children]);counts=np.array([c.n for c in root.children])
        q=np.array([c.value() for c in root.children]);assert counts.sum()==ns[i]==root.n
        out[i,ids]=reference.regularized_policy([c.prior for c in root.children],q,int(ns[i]),float(cp[i]))
        visits[i,ids],values[i,ids]=counts,q
    stats['seconds']=time.monotonic()-start
    assert np.allclose(out.sum(1),1.,atol=1e-6)
    return dict(policy=out,visits=visits,values=values,simulations=ns),stats
