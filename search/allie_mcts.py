"""Batched-across-roots adaptation of the released Allie ICLR MCTS.

Reference: ippolito-cmu/allie@a50f2d86618798cec2195e37e3484da579631328,
src/evaluation/decode.py. See ALLIE_LICENSE and PLAN.md for provenance/adapters.
Each tree runs strictly sequential simulations; batching changes only model calls.
"""
import math
import sys
import time
from pathlib import Path

import chess
import numpy as np
from scipy.special import softmax

SOURCE = Path('/data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours')
sys.path.insert(0, str(SOURCE))
from chess_vocab import MOVES, MOVE_ID

REFERENCE = 'a50f2d86618798cec2195e37e3484da579631328'
TIME_MEAN = 4.64001
BIN_SECONDS = np.r_[np.arange(16), 16 * np.exp(np.arange(47) / 7.06)]


def expected_seconds(logits):
    return softmax(np.asarray(logits, np.float64)[..., 2350:2413], axis=-1) @ BIN_SECONDS


def budgets(logits, mean_n_sims=50):
    # Algebraically identical to original standardized-scalar conversion,
    # with categorical expectation substituted for its scalar time prediction.
    return np.clip(np.rint(expected_seconds(logits) * mean_n_sims / TIME_MEAN), 0, 200).astype(int)


class Node:
    __slots__ = ('prior', 'parent', 'move', 'board', 'prefix', 'children', 'n', 'w', 'depth')

    def __init__(self, prior=0., parent=None, move=None, board=None, prefix=None):
        self.prior, self.parent, self.move = prior, parent, move
        self.board, self.prefix = board, prefix
        self.children = []
        self.n, self.w = 0, 0.
        self.depth = 0 if parent is None else parent.depth + 1

    def materialize(self):
        if self.board is None:
            self.board = self.parent.board.copy(stack=True)
            self.board.push(chess.Move.from_uci(MOVES[self.move - 378]))
            self.prefix = self.parent.prefix + [self.move]
        return self

    def value(self):
        return self.w / self.n if self.n else 0.


def terminal_value(node):
    outcome = node.board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.
    assert outcome.winner != node.board.turn
    return 1.  # all node values are from the parent mover's perspective


def expand(node, logits):
    legal = [MOVE_ID[m.uci()] for m in node.board.legal_moves]
    assert legal, 'Terminal nodes must bypass expansion'
    policy = softmax(np.asarray(logits, np.float64)[legal])
    node.children = [Node(float(p), node, m) for m, p in zip(legal, policy)]
    wdl = softmax(np.asarray(logits, np.float64)[2413:2416])
    return float(wdl[2] - wdl[0])  # negate the next mover's W-L estimate


def select_path(root, cpuct=1.25, depth_limit=100):
    path = [root]
    while path[-1].children and len(path) <= depth_limit:
        node = path[-1]
        factor = (math.log((node.n + 19652. + 1) / 19652.) + cpuct) * math.sqrt(node.n)
        # max is stable: first simulation chooses first legal child at N(root)=0.
        child = max(node.children, key=lambda c: c.value() + factor * c.prior / (1 + c.n))
        path.append(child.materialize())
    return path


def backup(path, value):
    for node in reversed(path):
        node.n += 1
        node.w += value
        value = -value


def regularized_policy(prior, values, n, cpuct):
    """Released reverse-KL policy, using its original bisection stopping rule."""
    prior, values = np.asarray(prior, np.float32), np.asarray(values, np.float32)
    if n == 0:
        return prior / prior.sum()
    lam = np.float32(cpuct * n / (len(prior) + n))
    lo, hi = (values + lam * prior).max(), (values + lam).max()
    for _ in range(1000):
        alpha = np.float32((lo + hi) / 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            p = lam * prior / (alpha - values)
        z = p.sum()
        if z > 1:
            lo = alpha
        else:
            hi = alpha
        if np.isclose(z, 1., rtol=1e-3, atol=1e-8) or np.isclose(lo, hi, rtol=1e-3, atol=1e-8):
            assert np.isfinite(p).all() and (p > 0).all(), 'Numerical failure in reference policy'
            return p / z
    raise RuntimeError('Allie policy bisection did not converge')


def run(rows, root_logits, oracle, *, adaptive=False, n_sims=50, mean_n_sims=50):
    """Return distributions/statistics; human target and outcomes are never read."""
    roots = []
    for row, logits in zip(rows, root_logits):
        board = chess.Board()
        for t in row['prefix'][11:]:
            board.push(chess.Move.from_uci(MOVES[t - 378]))
        root = Node(board=board, prefix=list(row['prefix']))
        assert terminal_value(root) is None, 'Scored root is automatically terminal'
        expand(root, logits)
        roots.append(root)
    ns = budgets(root_logits, mean_n_sims) if adaptive else np.full(len(rows), n_sims, int)
    cp = 1.25 * np.sqrt(mean_n_sims / np.maximum(ns, 1)) if adaptive else np.full(len(rows), 1.25)
    stats = dict(simulations=int(ns.sum()), evaluated_leaves=0, terminal_visits=0,
                 useful_prefix_tokens=0, requests=0, max_depth=0, seconds=0.)
    started = time.monotonic()
    for iteration in range(int(ns.max(initial=0))):
        paths = []
        for i in np.flatnonzero(ns > iteration):
            path = select_path(roots[i], float(cp[i]))
            stats['max_depth'] = max(stats['max_depth'], len(path) - 1)
            value = terminal_value(path[-1])
            if value is None:
                paths.append(path)
            else:
                backup(path, value)
                stats['terminal_visits'] += 1
        if not paths:
            continue
        prefixes = [p[-1].prefix for p in paths]
        assert max(map(len, prefixes)) <= 1025, 'Search exceeded model context; do not silently truncate'
        predictions = oracle(prefixes)
        stats['requests'] += 1
        stats['evaluated_leaves'] += len(paths)
        stats['useful_prefix_tokens'] += sum(map(len, prefixes))
        for path, logits in zip(paths, predictions):
            backup(path, expand(path[-1], logits))
    out = np.zeros((len(rows), 1968), np.float32)
    visits = np.zeros_like(out, dtype=np.int16)
    values = np.zeros_like(out)
    for i, root in enumerate(roots):
        ids = np.array([c.move - 378 for c in root.children])
        counts = np.array([c.n for c in root.children])
        q = np.array([c.value() for c in root.children])
        assert counts.sum() == ns[i] == root.n
        out[i, ids] = regularized_policy([c.prior for c in root.children], q, int(ns[i]), float(cp[i]))
        visits[i, ids], values[i, ids] = counts, q
    stats['seconds'] = time.monotonic() - started
    assert np.allclose(out.sum(1), 1., atol=1e-6)
    return dict(policy=out, visits=visits, values=values, simulations=ns), stats
