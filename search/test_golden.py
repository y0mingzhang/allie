"""CPU-only end-to-end scorer check with synthetic logits and development moves."""
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import golden
from chess_vocab import MOVE_ID


class FakeOracle:
    calls=0
    def __call__(self,prefixes,columns=None,all_tokens=False,compact=False):
        type(self).calls+=1
        n=sum(map(len,prefixes)) if all_tokens else len(prefixes)
        x=np.zeros((n,2432),np.float32)
        # Constant logits ensure no future-token information can enter a root.
        x[:,378:2346]=np.linspace(-1,1,1968)
        x[:,2350:2413]=-80
        x[:,2350]=80  # zero predicted seconds -> zero adaptive simulations
        return x if columns is None else x[:,columns]


def main():
    header=[2348,201,10,2,5,0,0,1,8,0,0]
    moves=['e2e4','e7e5','g1f3','b8c6','f1b5','a7a6']
    tokens=header+[MOVE_ID[m] for m in moves]
    labels=[-1]*11+[7,5,7,5,7,5]
    doc=dict(prefix=tokens,labels=labels,game='synthetic-dev',row=0,start=0)
    expected=list(golden.golden_data.positions(doc))
    plan=dict(reply=dict(alpha=1.,beta=4.),reply_calibrated=dict(alpha=.85,beta=7.3))
    original=golden.OUT;oracle=golden.Oracle
    golden.Oracle=FakeOracle
    with tempfile.TemporaryDirectory(prefix='golden-unit-') as tmp:
        golden.OUT=Path(tmp)
        golden.task(0,[doc],plan)
        with np.load(Path(tmp)/'00000.npz') as z:
            assert np.array_equal(z['cell'],labels[11:])
            assert np.array_equal(z['ply'],np.arange(6))
            assert z['nll'].shape==(6,len(golden.NAMES))
            for i,p in enumerate(expected):
                x=FakeOracle()([p['prefix']])[0,378:2346].astype(np.float64)
                target=p['target']-378;legal=np.array(p['legal'])-378
                raw=logsumexp(x)-x[target];base=logsumexp(x[legal])-x[target]
                assert abs(z['nll'][i,0]-raw)<1e-12
                assert abs(z['nll'][i,1]-base)<1e-12
                assert abs(z['nll'][i,2]-base)<2e-6
                assert abs(z['nll'][i,3]-base)<1e-10
                calibrated=logsumexp(.85*x[legal])-.85*x[target] if p['cell']%4==3 else base
                assert abs(z['nll'][i,4]-calibrated)<1e-10
            nonexpert=z['cell']%4!=3
            assert np.array_equal(z['nll'][nonexpert,2:],np.repeat(z['nll'][nonexpert,1:2],3,axis=1))
        calls=FakeOracle.calls;golden.task(0,[doc],plan);assert FakeOracle.calls==calls
    golden.OUT=original;golden.Oracle=oracle
    # Equal-cell macro must not silently become move-weighted mean.
    counts=np.arange(1,17);cells=np.repeat(np.arange(16),counts);loss=cells/10.
    r=golden.golden_data.aggregate(loss,cells,dict(scored_moves=counts.tolist(),cells=list(map(str,range(16)))))
    assert abs(r['macro']-.75)<1e-12 and abs(r['expert_macro']-.9)<1e-12
    print('PASS: exact targets, expert-only scope, normalization, cached resume, macro weighting; no golden model scoring')


if __name__=='__main__':main()
