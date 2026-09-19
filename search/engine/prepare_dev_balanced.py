"""Score-independent, game-disjoint fit/confirmation sample from private dev."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .native_board import from_prefix
from .service import ROOT,atomic


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    folder=ROOT/'aug-tune-v1';out=folder/'sample.json'
    assert not out.exists(),'Do not redraw a published development sample'
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():assert digest(folder/name)==expected
    with np.load(folder/'strat.npz') as z:tokens=z['rows'];labels=z['labels']
    games=json.loads((folder/'games.json').read_text())
    cells=[[[] for _ in range(16)] for _ in range(2)]
    for game in games:
        fold=int(hashlib.sha256(('search-dev-1926734:'+game['token_sha256']).encode()).hexdigest(),16)%2
        ri,start,length=game['row'],game['start'],game['length']
        for j in np.flatnonzero(labels[ri,start:start+length]>=0):
            cell=int(labels[ri,start+j]);cells[fold][cell].append((game,int(j)))
    rng=np.random.default_rng(1926734);selected=[];per_cell=128
    for fold in range(2):
        for cell in range(16):
            candidates=cells[fold][cell];assert len(candidates)>=per_cell,(fold,cell,len(candidates))
            for i in rng.choice(len(candidates),per_cell,replace=False):
                game,j=candidates[i];ri,start=game['row'],game['start']
                prefix=tokens[ri,start:start+j].astype(int).tolist()
                target=int(tokens[ri,start+j]);legal=from_prefix(prefix).legal()
                assert len(prefix)>=11 and len(prefix)<1025 and target in legal
                selected.append(dict(game=game['token_sha256'],site=game['game'],fold=fold,
                    cell=cell,ply=j-11,row=ri,column=start+j,prefix=prefix,target=target,legal=legal))
    # Locality ordering does not depend on model scores or the next move.
    selected.sort(key=lambda r:(r['row'],r['column']))
    assert not {r['game'] for r in selected if r['fold']==0}&{r['game'] for r in selected if r['fold']==1}
    assert len({(r['row'],r['column']) for r in selected})==len(selected)
    atomic(out,dict(positions=selected,seed=1926734,per_cell_per_fold=per_cell,
        manifest_sha256=digest(folder/'manifest.json'),source_sha256=digest(__file__),
        semantics='128 uniform August moves per cell per game-disjoint fold; 16 equal-weight cell means. Fit on fold0, report all preregistered arms on fold1. Both folds may be training-seen; July golden is the held-out benchmark. No golden CM conversion.',
        side_channels='This fixed checkpoint has no clock/Elo/feature side-channel inputs; strength/time-control headers are unchanged. No observed future is an input.'))
    print('Prepared',len(selected),'positions;',len({r['game'] for r in selected}),'games',flush=True)


if __name__=='__main__':main()
