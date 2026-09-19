"""Score-independent metadata for an immutable stratified search sample."""
import hashlib,json
import numpy as np
from .service import ROOT

def read(folder):
    rows=json.loads((ROOT/folder/'sample.json').read_text())['positions']
    n=len(rows);cells=np.array([r['cell'] for r in rows]);games=np.array([r['game'] for r in rows])
    fm=np.array([r['fold']==0 for r in rows]);cv=np.array([int(hashlib.sha256(('cv:'+g).encode()).hexdigest(),16)%3 for g in games])
    k=max(len(r['legal']) for r in rows);ids=np.zeros((n,k),int);mask=np.zeros((n,k),bool);target=np.zeros(n,int)
    for i,r in enumerate(rows):
        ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;target[i]=r['legal'].index(r['target'])
    assert not set(games[fm])&set(games[~fm])
    for f in range(3):assert (np.bincount(cells[fm&(cv==f)],minlength=16)>0).all()
    return dict(rows=rows,cells=cells,games=games,fit=fm,cv=cv,ids=ids,mask=mask,target=target)
