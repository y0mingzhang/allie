"""Allie MCTS: C++ selection/expansion/backup, identical Python output solver."""
import time
import numpy as np
from .mcts import reference
from _allie_board import NativeMCTS


def run(rows,root_logits,oracle,*,adaptive=False,n_sims=50,mean_n_sims=50):
    start=time.monotonic()
    ns=reference.budgets(root_logits,mean_n_sims) if adaptive else np.full(len(rows),n_sims,int)
    cp=1.25*np.sqrt(mean_n_sims/np.maximum(ns,1)) if adaptive else np.full(len(rows),1.25)
    tree=NativeMCTS([r['prefix'] for r in rows],root_logits,ns.tolist(),cp.tolist())
    while not tree.done:
        prefixes=tree.select()
        if prefixes:tree.update(oracle(prefixes))
    out=np.zeros((len(rows),1968),np.float32);visits=np.zeros_like(out,dtype=np.int16);values=np.zeros_like(out)
    for i,(ids,counts,q,prior) in enumerate(tree.summaries()):
        out[i,ids]=reference.regularized_policy(prior,q,int(ns[i]),float(cp[i]))
        visits[i,ids],values[i,ids]=counts,q
    stats=tree.stats();stats['seconds']=time.monotonic()-start
    assert np.allclose(out.sum(1),1.,atol=1e-6)
    return dict(policy=out,visits=visits,values=values,simulations=ns),stats
