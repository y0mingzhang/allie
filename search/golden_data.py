"""Read the EXISTING golden arrays. Never resample, rebuild, or alter labels.

This adapter exposes game prefixes to search; aggregation remains the exact
16-cell mean used by the frozen evaluator. No model scoring at import time.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import chess
from allie_mcts import MOVES, MOVE_ID

DATA=Path('/data/group_data/dei-group/yimingz3/allie/strat-eval-v1')


def load():
    manifest=json.loads((DATA/'manifest.json').read_text())
    with (DATA/'strat.npz').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['sha256']
    with np.load(DATA/'strat.npz') as z:
        rows=z['rows'].astype(np.int64);labels=z['labels'].astype(np.int64)
    assert rows.shape==labels.shape==(manifest['rows'],1025)
    assert np.array_equal(np.bincount(labels[:,1:][labels[:,1:]>=0],minlength=16),manifest['scored_moves'])
    return rows,labels,manifest


def games(rows,labels):
    for ri,(row,lab) in enumerate(zip(rows,labels)):
        starts=np.flatnonzero(row==2348)
        assert starts[0]==0
        for a,b in zip(starts,np.r_[starts[1:],len(row)]):
            a,b=int(a),int(b)
            if b-a<12:
                assert (lab[a:b]<0).all()
                continue
            tokens=row[a:b].tolist();cells=lab[a:b].tolist()
            # A document is one existing game; repeated identical documents
            # share a bootstrap cluster rather than pretending independence.
            game=hashlib.sha256(np.asarray(tokens,np.int16).tobytes()).hexdigest()[:24]
            yield dict(row=ri,start=a,prefix=tokens,labels=cells,game=game)


def positions(game):
    tokens=game['prefix'];board=chess.Board()
    for j in range(11,len(tokens)):
        t=tokens[j];cell=game['labels'][j]
        if not 378<=t<2346:
            assert cell<0
            continue
        if cell>=0:
            legal=[MOVE_ID[m.uci()] for m in board.legal_moves]
            assert t in legal,(game['row'],game['start'],j,'illegal labeled target')
            yield dict(game=game['game'],ply=j-11,cell=cell,prefix=tokens[:j],
                target=t,legal=legal,root_index=j-1)
        board.push(chess.Move.from_uci(MOVES[t-378]))


def aggregate(losses,cells,manifest):
    sums=np.bincount(cells,weights=losses,minlength=16)
    counts=np.bincount(cells,minlength=16)
    assert np.array_equal(counts,manifest['scored_moves'])
    mean=sums/counts
    return dict(macro=float(mean.mean()),expert_macro=float(mean[3::4].mean()),
        cells=dict(zip(manifest['cells'],mean.tolist())),counts=counts.tolist())


if __name__=='__main__':
    rows,labels,manifest=load();counts=np.zeros(16,np.int64);n=0
    for game in games(rows,labels):
        n+=1
        for p in positions(game):counts[p['cell']]+=1
    assert np.array_equal(counts,manifest['scored_moves'])
    assert n==manifest['games']
    print(json.dumps(dict(games=n,scored_moves=int(counts.sum()),counts=counts.tolist(),
        source_sha256=manifest['sha256'],note='Structural/index audit only; no model scores'),indent=2))
