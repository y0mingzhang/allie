"""Vectorized released Allie output solve, preserving each legal-count reduction."""
import numpy as np


def solve(summaries,ns,cp):
    out=np.zeros((len(summaries),1968),np.float32)
    groups={}
    for i,(ids,counts,q,prior) in enumerate(summaries):groups.setdefault(len(ids),[]).append(i)
    for length,rows in groups.items():
        ids=np.array([summaries[i][0] for i in rows]);q=np.array([summaries[i][2] for i in rows],np.float32)
        p=np.array([summaries[i][3] for i in rows],np.float32);n=ns[rows];c=cp[rows]
        lam=(c*n/(length+n)).astype(np.float32)
        lo=(q+lam[:,None]*p).max(1);hi=(q+lam[:,None]).max(1)
        result=np.empty_like(p);active=n!=0
        result[~active]=p[~active]/p[~active].sum(1,keepdims=True)
        for _ in range(1000):
            ix=np.flatnonzero(active)
            if not len(ix):break
            alpha=((lo[ix]+hi[ix])/2).astype(np.float32)
            prob=lam[ix,None]*p[ix]/(alpha[:,None]-q[ix]);z=prob.sum(1)
            above=z>1;lo[ix]=np.where(above,alpha,lo[ix]);hi[ix]=np.where(above,hi[ix],alpha)
            done=np.isclose(z,1.,rtol=1e-3,atol=1e-8)|np.isclose(lo[ix],hi[ix],rtol=1e-3,atol=1e-8)
            assert np.isfinite(prob[done]).all() and (prob[done]>0).all()
            result[ix[done]]=prob[done]/z[done,None];active[ix[done]]=False
        assert not active.any(),'Reference policy solver failed to converge'
        out[np.array(rows)[:,None],ids]=result
    return out
