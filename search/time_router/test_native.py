"""Growing repaired MCTS must match one uninterrupted reference search exactly."""
import json
import numpy as np
from search.engine.service import ROOT
from search.transfer.baselines import native as reference
from search.test_deeper import FakeOracle
from .collect import native

def main():
 rows=json.loads((ROOT/'dev.json').read_text())['positions'][:5]
 prefixes=[r['prefix'] for r in rows];z=FakeOracle()(prefixes);cp=[1.25]*len(rows)
 ref=reference().Reference(prefixes,z,[512]*len(rows),cp);ref.first_prior=ref.preserve_depth=True
 expected=[]
 for _ in range(512):
  seq=ref.select();expected.extend(seq)
  if seq:ref.update(FakeOracle()(seq))
 tree=native().Tree(prefixes,z,[64]*len(rows),cp);tree.first_prior=tree.preserve_depth=True
 mapping={i:p for i,p in enumerate(prefixes)};actual=[]
 for b in [64,128,512]:
  if b!=64:tree.grow([b]*len(rows))
  while not tree.done:
   h=tree.select();seq=[]
   for node,parent,token,length in h:
    p=mapping[int(parent)]+[int(token)];assert len(p)==length;mapping[int(node)]=p;seq.append(p)
   actual.extend(seq)
   if seq:tree.update(FakeOracle()(seq))
  for _,count,_,_ in tree.summaries():assert sum(count)==b
 assert actual==expected
 for a,b in zip(tree.summaries(),ref.summaries()):
  for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
 try:tree.grow([256]*len(rows))
 except ValueError:pass
 else:raise AssertionError('accepted shrink')
 print('PASS exact repaired-Allie paths, visits, priors and values:64→128→512 equals uninterrupted512; shrink rejected')

if __name__=='__main__':main()
