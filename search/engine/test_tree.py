"""Native/vectorized tree equals the existing search with fixed predictions."""
import json
import sys
import numpy as np
from .native_board import ROOT,MOVE_ID
from .tree import batch as fast
sys.path.insert(0,str(ROOT/'search'))
from test_deeper import FakeOracle
from deeper_pilot import batch as original
from reply_pilot import board_for


def main():
    rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions'][:3]
    prefix=rows[0]['prefix'][:11]+[MOVE_ID[m] for m in ['f2f3','e7e5','g2g4']]
    rows.append(dict(prefix=prefix,legal=[MOVE_ID[m.uci()] for m in board_for(prefix).legal_moves]))
    for width in [(4,),(4,2,2)]:
        a,_=original(rows,FakeOracle(),widths=width)
        for bs in (17,1024):
            b,_,stats=fast(rows,FakeOracle(),widths=width,batch_size=bs)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-7,equal_nan=True)
    print('PASS: native/vectorized tree matches original at every horizon, batch sizes 17/1024, terminal mate')


if __name__=='__main__':main()
