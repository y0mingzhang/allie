"""Shared August retrieval controls, refitted separately inside each game-CV fold."""
import hashlib,json
import numpy as np
from .service import ROOT
from .balanced_eval import digest
from .fit_policy import fit,loss_gradient

def data():
    rows=json.loads((ROOT/'aug-tune-v1/sample.json').read_text())['positions']
    n=len(rows);ar=np.arange(n);cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows]);fm=np.array([r['fold']==0 for r in rows])
    cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):
        ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    control=ROOT/'aug-adaptive-root-v1';cp=json.loads((control/'plan.json').read_text())
    assert cp['sample_sha256']==digest(ROOT/'aug-tune-v1/sample.json')
    root=np.zeros((n,2432));q=np.zeros((n,k));cost=np.zeros(n)
    for lo in range(0,n,cp['roots_per_batch']):
        with np.load(control/'quota_static'/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1]
            np.testing.assert_array_equal(z['game'],games[lo:hi]);np.testing.assert_array_equal(z['ids'],ids[lo:hi,:kk])
            root[lo:hi]=z['z'];q[lo:hi,:kk]=z['q'][cp['budgets'].index(1000)];cost[lo:hi]=z['evaluated_nodes'][cp['budgets'].index(1000)]
    logits=root[:,378:2346][ar[:,None],ids];group=cells%4
    def base_fit(q,train):
        p=np.zeros_like(q);params={}
        for g in range(4):
            take=train&(group==g);pred=group==g;count=np.bincount(cells[take],minlength=16)
            f=fit(logits[take],q[take],mask[take],target[take],'forward',1/count[cells[take]])
            assert f['converged'];params[str(g)]=f
            p[pred],_=loss_gradient([f['alpha'],f['beta']],logits[pred],q[pred],mask[pred],target[pred],'forward',return_policy=True)
        return p,params
    controls={name:[base_fit(bq,fm)]+[base_fit(bq,fm&(cv!=f)) for f in range(3)] for name,bq in [('direct',np.zeros_like(q)),('search',q)]}
    return dict(rows=rows,cells=cells,games=games,fit=fm,cv=cv,ids=ids,mask=mask,target=target,cost=cost,controls=controls)
