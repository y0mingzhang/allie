"""Rules, move order and automatic draws match the original python-chess path."""
import json
import random
import chess
from .native_board import ROOT,Position,MOVES,MOVE_ID,from_prefix


def check(a,b):
    expected=[MOVE_ID[m.uci()] for m in a.legal_moves];actual=b.legal()
    assert expected==actual,(a.fen(),expected,actual)
    o=a.outcome(claim_draw=False);v=-1 if o is None else .5 if o.winner is None else float(o.winner)
    assert v==b.outcome(),(a.fen(),v,b.outcome())


def main():
    rng=random.Random(11);n=0
    rows=json.loads((ROOT/'results/search-v1/dev.json').read_text())['positions']
    for row in rows:
        a=chess.Board();b=Position()
        for t in row['prefix'][11:]:a.push_uci(MOVES[t-378]);b.push(t)
        check(a,b);n+=1
        for t in rng.sample(b.legal(),min(4,len(b.legal()))):
            child=a.copy();child.push_uci(MOVES[t-378]);check(child,b.child(t));n+=1
    for fen in ['8/8/8/8/8/2k5/8/2K5 w - - 149 1',
                '8/8/8/8/8/2k5/8/R1K5 w - - 149 1',
                '8/8/8/8/8/2k5/8/R1K5 w - - 150 1',
                '8/8/8/8/B3B3/2k5/8/2K1B3 w - - 0 1',
                '7k/P7/8/8/8/8/7p/4K3 w - - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '8/8/8/3pP3/8/2k5/8/2K5 w - d6 0 1']:
        check(chess.Board(fen),Position(fen));n+=1
    a=chess.Board();b=Position()
    for u in ['g1f3','g8f6','f3g1','f6g8']*4:
        a.push_uci(u);b.push(MOVE_ID[u]);check(a,b);n+=1
    for game in range(30):
        a=chess.Board();b=Position()
        for ply in range(300):
            check(a,b);n+=1
            if a.outcome(claim_draw=False):break
            move=rng.choice(list(a.legal_moves));a.push(move);b.push(MOVE_ID[move.uci()])
    print(f'PASS: {n} positions; exact legal-move order, outcomes, 75-move and fivefold rules')


if __name__=='__main__':main()
