"""Compare incremental states to frozen full-prefix encoding and known past clocks."""
import json
from pathlib import Path
import numpy as np
from .board import encode
from .context import advance_boards,advance_clocks,root_other_previous


def main():
    root=Path(__file__).resolve().parents[2]/'results/search-v1'
    sample=json.loads((root/'aug-tune-expanded-v1/sample.json').read_text())['positions'][:1024]
    previous=[];moves=[];want=[]
    for row in sample:
        p=row['prefix'];b=encode(np.array(p,np.int64)[None])[0]
        previous.extend(b[10:-1]);moves.extend(p[11:]);want.extend(b[11:])
    got=advance_boards(np.array(previous),moves);np.testing.assert_array_equal(got,want)
    with np.load(root/'aug-tune-v1/feats.npz') as f:side=f['feats']
    n=0
    for row in sample:
        p=row['prefix'];start=row['column']-len(p);feat=side[row['row'],start:row['column']]
        inc=p[2]-10 if 10<=p[2]<191 else -1
        # First-move PGNs can include berserk resets; future resets are not inference inputs.
        for length in range(13,len(p)):
            parent=feat[length-1];child=feat[length]
            if inc<0 or (parent[:2]<0).any() or (child[:2]<0).any():continue
            elapsed=parent[0]-child[1]+(inc if length>=13 else 0)
            if elapsed<0:continue
            other=root_other_previous(p[:length],feat[:length],inc)
            got,_=advance_clocks(parent[None],np.array([other]),np.array([length]),np.array([inc]),np.array([elapsed]))
            np.testing.assert_array_equal(got[0],child);n+=1
    parent=np.array([[100.,120.,5.]])
    child,other=advance_clocks(parent,np.array([7.]),np.array([31]),np.array([2]),np.array([4.]))
    np.testing.assert_array_equal(child,[[120,98,7]])
    grand,_=advance_clocks(child,other,np.array([32]),np.array([2]),np.array([6.]))
    np.testing.assert_array_equal(grand,[[98,116,4]])
    unknown,_=advance_clocks(np.full((1,3),-1),np.array([-1]),np.array([31]),np.array([2]),np.array([4.]))
    np.testing.assert_array_equal(unknown,[[-1,-1,-1]])
    print('PASS',len(moves),'board transitions;',n,'past-clock transitions; same-mover think-time and missingness',flush=True)


if __name__=='__main__':main()
