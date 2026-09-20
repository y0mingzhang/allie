"""No-label allocation and cost-bound tests (CPU only)."""
import numpy as np
from .analysis import allocate,meta,design,costs

def main():
 rng=np.random.default_rng(7);cells=np.repeat(np.arange(16),20);p=np.tile([0,-.01,-.02,-.025],(320,1));c=np.tile([128,256,512,1000],(320,1));j=rng.random(p.shape)*1e-8
 for cap in [0,128,460,1000,2000]:
  a,_,actual=allocate(p,c,j,cells,cap);assert actual<=max(128,cap)+1e-8
  assert np.isfinite(actual)
 d=meta('aug');x,norm=design(d,'time',d['fit']);changed=dict(d,target=rng.permutation(d['target']),rows=[dict(r,target=0) for r in d['rows']]);xx,_=design(changed,'time',normalizer=norm);np.testing.assert_array_equal(x,xx)
 table=np.tile([125,250,498,972],(16,1));cc=costs(d,table);assert (cc[d['forced']]==0).all()
 # Deliberately re-use the SAME coefficients: labels may never influence the result.
 coef=rng.normal(size=(x.shape[1],4));a,*_=allocate(x@coef,cc,d['jitter'],d['cells']);b,*_=allocate(xx@coef,cc,d['jitter'],d['cells']);np.testing.assert_array_equal(a,b)
 print('PASS feasible/infeasible mean cost, forced zero cost, target changes leave features/decisions identical')

if __name__=='__main__':main()
