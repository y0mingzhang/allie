"""Transport parity against original algorithms, deterministic fake neural net."""
import hashlib,json
import numpy as np
from search.engine.service import ROOT
from search.engine.tree import batch
from search.engine.native_board import from_prefix
from .baselines import shallow,native


def pred(p):
    rng=np.random.default_rng(int(hashlib.sha256(np.array(p,np.int16).tobytes()).hexdigest()[:8],16))
    return rng.normal(size=2432).astype(np.float32)


def main():
    rows=json.loads((ROOT/'aug-tune-expanded-v1/sample.json').read_text())['positions'][:4]
    class Fake:
        def __init__(self):self.known={i:r['prefix'] for i,r in enumerate(rows)};self.queries=0
        def __call__(self,handles):
            result=[]
            for node,parent,token,length in handles:
                p=self.known[int(parent)]+[int(token)];assert len(p)==length
                self.known[int(node)]=p;result.append(pred(p));self.queries+=1
            return np.array(result)
    bridge=Fake();q,nodes=shallow(rows,bridge)
    def oracle(prefixes,columns=None):
        z=np.array([pred(p) for p in prefixes]);return z if columns is None else z[:,columns]
    original,z,stats=batch(rows,oracle)
    np.testing.assert_allclose(q,original,atol=6e-8,rtol=0)
    np.testing.assert_array_equal(nodes,np.array(stats['nodes_by_depth_and_root']).cumsum(0))
    # Compile reference binding exported by the same unchanged NativeMCTS include.
    module=native()
    for repairs in (False,True):
        ns=[50,80,0,24];cp=[1.25]*4
        a=module.Tree([r['prefix'] for r in rows],z,ns,cp);b=module.Reference([r['prefix'] for r in rows],z,ns,cp)
        a.first_prior=b.first_prior=repairs;a.preserve_depth=b.preserve_depth=repairs;bridge=Fake()
        while not a.done:
            h=a.select();ps=b.select();assert len(h)==len(ps)
            if len(h):
                v=bridge(h);np.testing.assert_array_equal(v,oracle(ps));a.update(v);b.update(v)
        assert a.stats()==b.stats()
        for x,y in zip(a.summaries(),b.summaries()):
            for xx,yy in zip(x,y):np.testing.assert_array_equal(xx,yy)
    print('PASS shallow values/counts and released/repaired Allie transports',flush=True)


if __name__=='__main__':main()
