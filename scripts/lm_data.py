"""Original packed rows, with deterministic global shuffling and exact resume."""
import copy
from pathlib import Path
import numpy as np

class Packed:
    def __init__(self,path,split,seed=42,limit=0):
        self.paths=sorted(Path(path).glob(f'*/{split}.npy'))
        if limit:self.paths=self.paths[:limit]
        if not self.paths:raise ValueError('No original packed data found')
        self.arrays=[np.load(p,mmap_mode='r').reshape(-1,1025) for p in self.paths]
        self.counts=np.array([len(a) for a in self.arrays]);self.ends=np.cumsum(self.counts)
        self.rng=np.random.default_rng(seed);self.order=None;self.order_seed=None
        self.pos=0;self.epochs=0;self.seen=0

    def rows(self,indices):
        out=np.empty((len(indices),1025),np.int64)
        shards=np.searchsorted(self.ends,indices,side='right');starts=np.r_[0,self.ends[:-1]]
        for s in np.unique(shards):
            mask=shards==s;out[mask]=self.arrays[s][indices[mask]-starts[s]]
        return out

    def batch(self,n,rank=0,world=1):
        # All ranks advance the same global stream, then take disjoint portions.
        remaining=n*world;pieces=[]
        while remaining:
            if self.order is None or self.pos==len(self.order):
                self.order_seed=int(self.rng.integers(0,2**63-1))
                self.order=np.random.default_rng(self.order_seed).permutation(int(self.ends[-1]))
                self.pos=0;self.epochs+=1
            take=min(remaining,len(self.order)-self.pos)
            pieces.append(self.order[self.pos:self.pos+take]);self.pos+=take;remaining-=take
        indices=np.concatenate(pieces);self.seen+=len(indices)
        return self.rows(indices[rank*n:(rank+1)*n])

    def state_dict(self):
        return dict(rng=copy.deepcopy(self.rng.bit_generator.state),order_seed=self.order_seed,
            pos=self.pos,epochs=self.epochs,seen=self.seen,
            files=[dict(path=str(p.relative_to(p.parents[1])),size=p.stat().st_size,rows=int(n)) for p,n in zip(self.paths,self.counts)])

    def load_state_dict(self,state):
        if state['files']!=self.state_dict()['files']:raise ValueError('Checkpoint corpus differs from current corpus')
        self.rng.bit_generator.state=state['rng'];self.order_seed=state['order_seed']
        self.pos=state['pos'];self.epochs=state['epochs'];self.seen=state['seen']
        self.order=None if self.order_seed is None else np.random.default_rng(self.order_seed).permutation(int(self.ends[-1]))
