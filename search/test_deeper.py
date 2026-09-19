"""Check horizon expectations against the old two-ply code and recursion."""
import json
import numpy as np
from scipy.special import softmax
import chess

from allie_mcts import MOVE_ID, MOVES, clone_board
from reply_pilot import board_for, outcome_value, batch as two_ply
from deeper_pilot import ROOT, batch


class FakeOracle:
    def __call__(self,prefixes,columns=None):
        x=np.empty((len(prefixes),2432),np.float32)
        for i,p in enumerate(prefixes):
            s=sum((j+1)*t for j,t in enumerate(p))%997
            x[i]=np.sin(np.arange(2432)*.017+s*.19)
        return x if columns is None else x[:,columns]


def main():
    oracle=FakeOracle();rows=json.loads((ROOT/'dev.json').read_text())['positions'][:3]
    header=rows[0]['prefix'][:11];prefix=header+[MOVE_ID[m] for m in ['f2f3','e7e5','g2g4']]
    board=board_for(prefix)
    rows.append(dict(prefix=prefix,legal=[MOVE_ID[m.uci()] for m in board.legal_moves]))
    q,stats=batch(rows,oracle);old,_=two_ply(rows,oracle)
    assert np.allclose(q[0],old['q1'],atol=1e-7,equal_nan=True)
    assert np.allclose(q[1],old['q2'],atol=1e-7,equal_nan=True)
    mate=MOVE_ID['d8h4']-378
    assert np.array_equal(q[:,3,mate],np.ones(4))
    assert stats['leaves_by_depth'][0]>0
    def recursive(prefix,board,side,widths):
        terminal=outcome_value(board,side)
        if terminal is not None:return terminal
        logits=oracle([prefix])[0];v=softmax(logits[2413:2416].astype(float))@np.array([1.,.5,0.])
        if board.turn!=side:v=1-v
        if not widths:return v
        legal=np.array([MOVE_ID[m.uci()] for m in board.legal_moves]);p=softmax(logits[legal].astype(float))
        out=v
        for ix in np.argsort(p)[::-1][:widths[0]]:
            token=int(legal[ix]);child=clone_board(board);child.push(chess.Move.from_uci(MOVES[token-378]))
            out+=p[ix]*(recursive(prefix+[token],child,side,widths[1:])-v)
        return out
    row=rows[-1];board=board_for(row['prefix'])
    for move in board.legal_moves:
        token=MOVE_ID[move.uci()];child=clone_board(board);child.push(move)
        expected=recursive(row['prefix']+[token],child,board.turn,[4,2,2])
        assert abs(q[3,-1,token-378]-expected)<1e-7
    print('PASS: horizon 1/2 parity, four-ply recursive expectation, terminal exactness, legal coverage')


if __name__=='__main__':main()
