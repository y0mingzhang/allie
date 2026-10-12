"""The original corpus's packed rows (validation)."""
from pathlib import Path
import numpy as np

class Packed:
    def __init__(self,path,split):
        self.paths=sorted(Path(path).glob(f'*/{split}.npy'))
        if not self.paths:raise ValueError('No original packed data found')
        self.arrays=[np.load(p,mmap_mode='r').reshape(-1,1025) for p in self.paths]
        self.ends=np.cumsum([len(a) for a in self.arrays])

    def rows(self,indices):
        out=np.empty((len(indices),1025),np.int64)
        shards=np.searchsorted(self.ends,indices,side='right');starts=np.r_[0,self.ends[:-1]]
        for s in np.unique(shards):
            mask=shards==s;out[mask]=self.arrays[s][indices[mask]-starts[s]]
        return out
