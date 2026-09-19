"""Default equivalence plus first-simulation and depth-boundary regression checks."""
import json
import numpy as np
from .native_board import ROOT, MOVES
from .mcts import reference
from .policy import solve
from test_deeper import FakeOracle
import _allie_board_v2 as cpp
cpp.initialize(MOVES)


def main():
    rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][::128]
    prefixes=[r['prefix'] for r in rows];root=FakeOracle()(prefixes)
    for n in (0,1,8,52):
        paths=[]
        def recorded(seq):
            paths.extend(seq);return FakeOracle()(seq)
        expected,_=reference.run(rows,root,recorded,n_sims=n)
        tree=cpp.NativeMCTS(prefixes,root,[n]*len(rows),[1.25]*len(rows));actual=[]
        while not tree.done:
            seq=tree.select();actual.extend(seq)
            if seq:tree.update(FakeOracle()(seq))
        assert actual==paths
        policy=solve(tree.summaries(),np.full(len(rows),n),np.full(len(rows),1.25))
        np.testing.assert_allclose(policy,expected['policy'],atol=1e-7,rtol=0)
    tree=cpp.NativeMCTS(prefixes,root,[1]*len(rows),[1.25]*len(rows));tree.first_prior=True
    selected=tree.select()
    for row,z,seq in zip(rows,root,selected):
        assert seq[-1]==max(row['legal'],key=lambda t:float(z[t]))
    tree=cpp.NativeMCTS(prefixes,root,[256]*len(rows),[1.25]*len(rows))
    tree.first_prior=tree.preserve_depth=True;tree.max_search_depth=1
    seen=set()
    while not tree.done:
        seq=tree.select()
        for s in seq:
            assert tuple(s) not in seen,'Re-evaluated a depth-limited expanded node'
            seen.add(tuple(s))
        if seq:tree.update(FakeOracle()(seq))
    assert tree.stats()['depth_limited_visits']>0
    for _,counts,_,_ in tree.summaries():assert sum(counts)==256
    print('PASS: default paths/output; first visit follows prior; depth-limited nodes preserve statistics and cached values')


if __name__=='__main__':main()
