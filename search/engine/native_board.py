"""Load our compiled rules adapter and the checkpoint's exact move vocabulary."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/search-v1/runtime/native'))
sys.path.insert(0,'/data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours')
from chess_vocab import MOVES,MOVE_ID
from _allie_board import Position,initialize
initialize(MOVES)


def from_prefix(prefix):
    board=Position()
    for token in prefix[11:]:board.push(token)
    return board
