"""Independent fixed-horizon equivalence and budget/fallback invariants."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .expectation_pilot import batch
from .tree import batch as exhaustive


def main():
    rows=json.loads((Path(__file__).resolve().parents[2]/'results/search-v1/dev.json').read_text())['positions'][:8]
    def oracle(prefixes,columns=None):
        zs=[]
        for p in prefixes:
            seed=int.from_bytes(hashlib.blake2b(np.array(p,np.int32).tobytes(),digest_size=8).digest(),'little')
            z=np.random.default_rng(seed).normal(size=2432).astype(np.float32)
            zs.append(z if columns is None else z[columns])
        return np.array(zs)
    reference,_,_=exhaustive(rows,oracle,widths=(2,2))
    for priority in ('mass','policy','variance'):
        q,_,stats=batch(rows,oracle,budget=10000,width=2,depth=3,priority=priority)
        np.testing.assert_allclose(q[0],reference[0],rtol=0,atol=1e-7,equal_nan=True)
        np.testing.assert_allclose(q[1],reference[2],rtol=0,atol=1e-7,equal_nan=True)
        q,_,stats=batch(rows,oracle,budget=50,width=2,depth=8,priority=priority)
        assert all(v<=50 for v in stats['per_root_leaves'])
        for i,r in enumerate(rows):assert np.isfinite(q[:,i,np.array(r['legal'])-378]).all()
    print('PASS: all three priorities recover fixed-horizon expectation when fully expanded, obey node caps and retain every legal root move.')


if __name__=='__main__':main()
