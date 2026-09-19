"""Differential test against original classes, without loading an old model."""
import ast
import functools
import hashlib
import json
import math
from collections import namedtuple
from pathlib import Path

import chess
import numpy as np
import torch
from scipy.special import softmax
import allie_mcts as ours

ROOT = Path(__file__).resolve().parents[1]


class Game:
    def __init__(self, prefix):
        self.prefix = prefix
        self.board = chess.Board()
        for t in prefix[11:]:
            self.board.push(chess.Move.from_uci(ours.MOVES[t - 378]))
        self.legal_moves = [m.uci() for m in self.board.legal_moves]

    def make_move(self, move):
        return Game(self.prefix + [ours.MOVE_ID[move]])


def fake_oracle(prefixes):
    outputs = []
    for prefix in prefixes:
        seed = int(hashlib.sha256(np.array(prefix, np.int16).tobytes()).hexdigest()[:8], 16)
        x = np.random.default_rng(seed).normal(0, 2, 2432)
        x[2350:2413] = -100
        x[2356] = 0  # deterministic six-second time prediction
        outputs.append(x)
    return np.array(outputs)


ScoreResult = namedtuple('ScoreResult', 'move_probabilities think_time resigned')


class Policy:
    def init(self, **kwargs):
        pass

    def score_full(self, game, moves):
        logits = fake_oracle([game.prefix])[0]
        return ScoreResult(torch.tensor(softmax(logits[[ours.MOVE_ID[m] for m in moves]])), 6., False)


def reference():
    file = ROOT / 'vendor/allie-iclr/src/evaluation/decode.py'
    tree = ast.parse(file.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('MCTSNode', 'MCTS', 'AdaptiveMCTS')]
    ns = dict(Policy=Policy, Game=Game, torch=torch, chess=chess, math=math, functools=functools,
              ScoreResult=ScoreResult, TIME_MEAN=ours.TIME_MEAN, TIME_STDEV=6.16533)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), 'exec'), ns)

    def evaluate(self, node):
        outcome = node.board.outcome()
        if outcome is not None:
            return 0. if outcome.winner is None else 1.
        logits = fake_oracle([node.game.prefix])[0]
        policy = softmax(logits[[ours.MOVE_ID[m] for m in node.legal_moves]])
        for move, prior in zip(node.legal_moves, policy):
            node.children[move] = ns['MCTSNode'](float(prior), node.game.make_move(move))
        value = softmax(logits[2413:2416])
        return float(value[2] - value[0])

    def evaluate_root(self, node):
        value = evaluate(self, node)
        seconds = float(ours.expected_seconds(fake_oracle([node.game.prefix]))[0])
        return value, self.convert_time_to_sims((seconds - ours.TIME_MEAN) / 6.16533), seconds

    ns['MCTS'].evaluate_node = evaluate
    ns['AdaptiveMCTS'].evaluate_root = evaluate_root
    return ns


def main():
    refs = reference()
    rows = json.loads((ROOT / 'results/search-v1/dev.json').read_text())['positions'][:3]
    # Add a position with checkmate available, keeping the true game history.
    rows += [dict(prefix=rows[0]['prefix'][:11] + [ours.MOVE_ID[m] for m in ('f2f3','e7e5','g2g4')])]
    for adaptive, count in [(False, 1), (False, 8), (False, 50), (True, 1), (True, 8), (True, 50)]:
        pred = fake_oracle([r['prefix'] for r in rows])
        got, _ = ours.run(rows, pred, fake_oracle, adaptive=adaptive, n_sims=count, mean_n_sims=count)
        for i, row in enumerate(rows):
            cls = refs['AdaptiveMCTS' if adaptive else 'MCTS']
            mcts = cls()
            mcts.init(**{'mean_n_sims' if adaptive else 'n_sims': count})
            game = Game(row['prefix'])
            old = mcts.score_full(game, game.legal_moves).move_probabilities.numpy()
            ids = [ours.MOVE_ID[m] - 378 for m in game.legal_moves]
            np.testing.assert_allclose(got['policy'][i, ids], old, atol=2e-7, rtol=2e-6)
            assert got['policy'][i].argmax() == ids[old.argmax()]
    low = fake_oracle([rows[0]['prefix']]); low[:,2350:2413] = -100; low[:,2350] = 0
    got, stats = ours.run(rows[:1], low, fake_oracle, adaptive=True)
    ids = [ours.MOVE_ID[m.uci()] - 378 for m in Game(rows[0]['prefix']).board.legal_moves]
    np.testing.assert_allclose(got['policy'][0,ids], softmax(low[0,np.array(ids)+378]), atol=1e-7)
    assert stats['simulations'] == stats['evaluated_leaves'] == 0
    root = ours.Node(board=chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'), prefix=[])
    assert ours.terminal_value(root) == 1.
    root.board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
    assert ours.terminal_value(root) == 0.
    parent=chess.Board()
    for move in ['g1f3','g8f6','f3g1','f6g8']*4:parent.push_uci(move)
    original=parent.fen();stack=[m.uci() for m in parent.move_stack]
    fast=ours.clone_board(parent);slow=parent.copy(stack=True)
    assert fast.outcome()==slow.outcome() and fast.is_fivefold_repetition()
    for _ in range(8):
        fast.pop();slow.pop()
        assert fast.fen()==slow.fen() and fast.outcome()==slow.outcome()
    assert parent.fen()==original and [m.uci() for m in parent.move_stack]==stack
    print('PASS: 24 differential tree comparisons, zero-budget prior, mate/stalemate orientation, history clone/repetition isolation')


if __name__ == '__main__':
    main()
