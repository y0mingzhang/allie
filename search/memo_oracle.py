"""Bounded, lossless cache for repeated causal-prefix inference.

Cache full root/leaf outputs. A later request for only WDL columns can reuse them;
partial-output misses remain partial to avoid unnecessary network traffic.
"""
from collections import OrderedDict
import numpy as np


class MemoOracle:
    def __init__(self,oracle,capacity=20000):
        self.oracle=oracle;self.capacity=capacity;self.cache=OrderedDict()
        self.counts=dict(requested_prefixes=0,cached_prefixes=0,evaluated_prefixes=0,
                         evaluated_prefix_tokens=0,underlying_requests=0)

    def prime(self,prefixes,logits):
        for p,row in zip(prefixes,logits):self.put(tuple(p),row)

    def put(self,key,row):
        half=row.astype(np.float16)
        self.cache[key]=half if np.array_equal(half.astype(row.dtype),row) else row.copy()
        self.cache.move_to_end(key)
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)

    def __call__(self,prefixes,columns=None,*,all_tokens=False,compact=False):
        if all_tokens:return self.oracle(prefixes,columns,all_tokens=True,compact=compact)
        self.counts['requested_prefixes']+=len(prefixes)
        out=[None]*len(prefixes);miss=OrderedDict()
        for i,p in enumerate(prefixes):
            key=tuple(p)
            if key in self.cache:
                row=self.cache[key];out[i]=row if columns is None else row[columns]
                self.cache.move_to_end(key);self.counts['cached_prefixes']+=1
            else:miss.setdefault(key,[]).append(i)
        if miss:
            keys=list(miss);batch=[list(k) for k in keys]
            pred=self.oracle(batch,columns=columns)
            self.counts['underlying_requests']+=1
            self.counts['evaluated_prefixes']+=len(keys)
            self.counts['evaluated_prefix_tokens']+=sum(map(len,keys))
            for key,row in zip(keys,pred):
                if columns is None:self.put(key,row)
                for i in miss[key]:out[i]=row
                self.counts['cached_prefixes']+=len(miss[key])-1
        return np.asarray(out,dtype=np.float16 if compact else np.float32)


def test():
    class Fake:
        def __init__(self):self.calls=[]
        def __call__(self,prefixes,columns=None,**kwargs):
            self.calls.append(prefixes)
            out=np.array([[sum(p),len(p),p[-1]] for p in prefixes],np.float32)
            return out if columns is None else out[:,columns]
    raw=Fake();memo=MemoOracle(raw,capacity=2)
    x=memo([[1,2],[3,4],[1,2]])
    assert len(raw.calls[0])==2
    assert np.array_equal(x,[[3,2,2],[7,2,4],[3,2,2]])
    assert np.array_equal(memo([[3,4],[1,2]],columns=[2,0]),[[4,7],[2,3]])
    assert len(raw.calls)==1
    memo([[9,10]]);assert len(memo.cache)==2
    assert memo.counts['cached_prefixes']==3
    print('PASS: duplicate-prefix elimination, full/partial outputs, order, bounded eviction')


if __name__=='__main__':test()
