"""Batched output solver matches the released per-tree FP32 algorithm exactly."""
import sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .policy import solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from allie_mcts import regularized_policy


def main():
    rng=np.random.default_rng(9024);summaries=[];ns=[];cp=[];expected=[]
    for i in range(4096):
        count=int(rng.integers(1,96));q=rng.uniform(-1,1,count);p=softmax(rng.uniform(-23,0,count))
        if i%7==0:q[:]=0
        if i%11==0:p[:]=1/count
        n=int(rng.integers(0,201));c=float(rng.uniform(.05,5.))
        summaries.append((list(range(count)),[0]*count,q,p));ns.append(n);cp.append(c)
        expected.append(regularized_policy(p,q,n,c))
    actual=solve(summaries,np.array(ns),np.array(cp))
    for i,x in enumerate(expected):np.testing.assert_array_equal(actual[i,:len(x)],x)
    print('PASS: 4096 random/tied/zero-budget policies are bit-identical to released solver')


if __name__=='__main__':main()
