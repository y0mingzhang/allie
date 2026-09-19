"""Difference-estimator invariants: equal cells, paired bootstrap, no false SE."""
import numpy as np
from .analyze_balanced import cellmean,bootstrap_deltas


def main():
    # Unequal observed counts must not turn the macro metric into a pooled mean.
    cells=np.repeat(np.arange(16),np.arange(40,56));n=len(cells)
    known=np.linspace(1.,2.,16);reference=np.sin(np.arange(n))+known[cells]
    change=np.linspace(-.16,-.01,16);method=reference+change[cells]
    got=known+cellmean(method-reference,cells)
    np.testing.assert_allclose(got,known+change,atol=1e-12)
    assert not np.isclose(got.mean(),method.mean())
    # Pairing must cancel identical methods exactly, even with correlated games.
    games=np.array([str(i//2) for i in range(n)])
    differences=np.stack([method-reference,method-reference,np.zeros(n)],axis=1)
    draws=bootstrap_deltas(differences,cells,games,repeats=64)
    np.testing.assert_allclose(draws[:,:,0],change[None,:]+np.zeros((64,16)),atol=1e-12)
    np.testing.assert_array_equal(draws[:,:,0]-draws[:,:,1],0.)
    np.testing.assert_array_equal(draws[:,:,2],0.)
    print('PASS: equal-cell aggregation, exact paired differences and whole-game bootstrap invariants')


if __name__=='__main__':main()
