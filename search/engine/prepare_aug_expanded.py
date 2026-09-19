"""Nested, score-independent expansion of the existing August development sample."""
import hashlib,json
from pathlib import Path
import numpy as np
from .service import ROOT,atomic
from .balanced_eval import digest
from .native_board import from_prefix

def main():
    folder=ROOT/'aug-tune-v1';out=ROOT/'aug-tune-expanded-v1';assert not out.exists()
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,sha in manifest['files_sha256'].items():assert digest(folder/name)==sha
    old=json.loads((folder/'sample.json').read_text());assert old['per_cell_per_fold']==128
    with np.load(folder/'strat.npz') as z:tokens=z['rows'];labels=z['labels']
    games=json.loads((folder/'games.json').read_text());cells=[[[] for _ in range(16)] for _ in range(2)]
    previous={(r['row'],r['column']) for r in old['positions']}
    for game in games:
        fold=int(hashlib.sha256(('search-dev-1926734:'+game['token_sha256']).encode()).hexdigest(),16)%2
        ri,start,length=game['row'],game['start'],game['length']
        for j in np.flatnonzero(labels[ri,start:start+length]>=0):
            if (ri,start+int(j)) not in previous:
                cells[fold][int(labels[ri,start+j])].append((game,int(j)))
    rng=np.random.default_rng(1926791);selected=list(old['positions'])
    for fold in range(2):
        for cell in range(16):
            candidates=cells[fold][cell];assert len(candidates)>=384,(fold,cell,len(candidates))
            for i in rng.choice(len(candidates),384,replace=False):
                game,j=candidates[i];ri,start=game['row'],game['start']
                prefix=tokens[ri,start:start+j].astype(int).tolist();target=int(tokens[ri,start+j]);legal=from_prefix(prefix).legal()
                assert len(prefix)>=11 and len(prefix)<1025 and target in legal
                selected.append(dict(game=game['token_sha256'],site=game['game'],fold=fold,cell=cell,ply=j-11,row=ri,column=start+j,prefix=prefix,target=target,legal=legal))
    selected.sort(key=lambda r:(r['row'],r['column']))
    assert len(selected)==16384 and len({(r['row'],r['column']) for r in selected})==16384
    assert not {r['game'] for r in selected if r['fold']==0}&{r['game'] for r in selected if r['fold']==1}
    for fold in range(2):assert np.bincount([r['cell'] for r in selected if r['fold']==fold],minlength=16).tolist()==[512]*16
    lookup={(r['row'],r['column']):r for r in selected}
    assert all(lookup[(r['row'],r['column'])]==r for r in old['positions'])
    out.mkdir()
    atomic(out/'sample.json',dict(positions=selected,seed=1926791,per_cell_per_fold=512,
        parent_sample_sha256=digest(folder/'sample.json'),manifest_sha256=digest(folder/'manifest.json'),source_sha256=digest(Path(__file__)),
        inventory=str(folder),semantics='Original128 uniform moves/cell/fold plus384 uniform remaining moves, selected independently of model scores. Same immutable August inventory, masks and game-fold assignment. Both folds may be training-seen. No new golden data or golden parameter selection. Side channels remain original row/column references.'))
    print('Expanded',len(selected),'positions',len({r['game'] for r in selected}),'games; old sample retained exactly')
if __name__=='__main__':main()
