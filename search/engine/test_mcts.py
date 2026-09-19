"""Check native Allie trees against the audited Python adapter, path for path."""
import json
import numpy as np
from .mcts import run,reference
from .native_mcts import run as cpp
from .native_board import ROOT,MOVE_ID
from test_deeper import FakeOracle


class Recorded(FakeOracle):
    def __init__(self):self.paths=[]
    def __call__(self,seq,**kwargs):
        self.paths.extend(tuple(s) for s in seq)
        return super().__call__(seq,**kwargs)


def main():
    all_rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions']
    rows=all_rows[::32]
    prefix=rows[0]['prefix'][:11]+[MOVE_ID[m] for m in ['f2f3','e7e5','g2g4']]
    rows.append(dict(prefix=prefix))
    root=FakeOracle()([r['prefix'] for r in rows])
    for adaptive in (False,True):
        for sims in (0,52):
            a,b,c=Recorded(),Recorded(),Recorded()
            x,s1=reference.run(rows,root,a,adaptive=adaptive,n_sims=sims)
            y,s2=run(rows,root,b,adaptive=adaptive,n_sims=sims)
            z,s3=cpp(rows,root,c,adaptive=adaptive,n_sims=sims)
            assert a.paths==b.paths
            assert a.paths==c.paths
            for key in x:np.testing.assert_array_equal(x[key],y[key])
            for key in x:np.testing.assert_allclose(x[key],z[key],rtol=0,atol=1e-7)
            for key in s1:
                if key!='seconds':assert s1[key]==s2[key]==s3[key],key
    print('PASS: native MCTS identical paths, policies, visits, values and budgets; fixed/adaptive/terminal/zero budget')


if __name__=='__main__':main()
