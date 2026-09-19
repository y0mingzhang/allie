import json
import numpy as np
from scipy.special import logsumexp
from .service import ROOT,atomic
from .header_adaptation import run as collect,condition,history,OFFSETS

def run(oracle,spec):
    result=collect(oracle,dict(smoke=True))
    rows=json.loads((ROOT/'aug-tune-expanded-v1/sample.json').read_text())['positions'][:16]
    with np.load(ROOT/'header-adaptation-smoke-v1/000000.npz') as f:evidence=f['evidence']
    diffs=[]
    for vi,delta in enumerate(OFFSETS):
        seqs=[];expected=[]
        for i,r in enumerate(rows):
            p=condition(r['prefix'],delta);ix=history(p)
            for j,target_position in enumerate(ix[:2]):
                seqs.append(p[:target_position]);expected.append((p[target_position],evidence[vi,i,j]))
        oracle.reset();z=oracle(seqs).astype(float)
        got=z[np.arange(len(z)),[t for t,_ in expected]]-logsumexp(z[:,378:2346],axis=1)
        diffs.extend((got-np.array([v for _,v in expected])).tolist())
    assert np.isfinite(diffs).all()
    assert max(abs(v) for v in diffs)<.25,diffs
    result['past_loglik_abs_difference_quantiles']=np.quantile(np.abs(diffs),[0,.5,.9,1]).tolist()
    result['past_comparisons']=len(diffs)
    atomic(ROOT/'header-adaptation-smoke-v1/results.json',result)
    return result
