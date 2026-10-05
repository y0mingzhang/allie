"""The Rust coverage tree (allie_fast.Coverage, treers.Coverage) against today's C++ tree and Python bookkeeping
(tree.Coverage): the same handles call by call, the same compact arrays, values, boards and clocks, and the same
search outputs on the tiny model through the torch and the C++ fast backends, with and without views."""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from allie.lichess.engine import Engine, Game
from allie.lichess.model import Model

from .test_tokens import random_game

tree = pytest.importorskip("allie.lichess.tree")
treers = pytest.importorskip("allie.lichess.treers")
allie_fast = pytest.importorskip("allie_fast")
if not hasattr(allie_fast, "Coverage"):
    pytest.skip("allie_fast built without Coverage", allow_module_level=True)
pytest.importorskip("pybind11")
native = Path(
    os.environ.get(
        "ALLIE_CHESS_INCLUDE",
        str(Path(tree.__file__).parents[1] / "search/native/chess-library"),
    )
)
if not (native / "chess.hpp").exists() or not shutil.which("c++"):
    pytest.skip("native search", allow_module_level=True)

from allie.search.native import load


def setup(engine, seed, plies, base=1800, inc=20, speed="classical"):
    g = Game(engine, 2400, 2300, base, inc, speed)
    g.update(random_game(seed, plies), 1700, 1690)
    return g


def same(a, b):
    """Two searches' (moves, probabilities, prior, values) bit for bit."""
    assert a[0] == b[0]
    for x, y in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(x, y)


def test_trees_agree_call_by_call(tiny):
    """One evaluator (tree.Nodes on the torch backend) feeds both trees: the Rust tree's handles equal the C++ tree's
    at every call, then its compact arrays, values, boards and clock features equal the C++ tree's and Nodes'."""
    game = setup(Engine(tiny), 3, 17)
    feats = np.array(game.features(), np.float32)
    z = game.sync()
    par = treers.Coverage().parameters
    cpp, value = load(), load("value")
    for budget in (5, 32):
        nodes = tree.Tree(game, z, capacity=4 * budget + 256).handles(
            [game.tokens], [feats]
        )
        root = nodes.root_logits
        rs = allie_fast.Coverage(
            list(game.tokens), root[0], feats, game.inc, budget, par["cpuct"]
        )
        cx = cpp.Coverage([list(game.tokens)], root, [budget], [par["cpuct"]], 2)
        calls = 0
        while not rs.done:
            h = rs.select()
            np.testing.assert_array_equal(h, cx.select())
            if len(h):
                zz = nodes(h)
                rs.update(zz)
                cx.update(zz)
                calls += 1
        assert cx.done and rs.evals == list(cx.evals) == [nodes.queries]
        s = rs.stats()
        assert (
            s["evaluated_leaves"] == nodes.queries
            and s["requests"] == calls
            and s["simulations"] == budget
        )
        mine, theirs = rs.compact(), cx.compact()
        assert set(mine) == set(theirs)
        for k in theirs:
            np.testing.assert_array_equal(mine[k], theirs[k], err_msg=k)
        ids = (
            np.array([allie_fast.Position.from_tokens(game.tokens).legal()], np.int64)
            - 378
        )
        b = par["backup"]
        q = rs.reduce(np.log(b["tau"]), b["exponent"], b["count_scale"], budget, ids)
        qc = value.Backup(theirs, budget, b["count_scale"]).reduce(
            np.log(b["tau"]), b["exponent"], ids.astype(np.int32)
        )
        np.testing.assert_array_equal(q, qc)
        all_ids = [
            i for i in range(len(theirs["parent"])) if nodes.length[i]
        ]  # evaluated: not the terminal nodes
        assert len(all_ids) == nodes.queries + 1
        np.testing.assert_array_equal(
            rs.feats(all_ids), nodes.feats[all_ids].astype(np.float32)
        )
        boards = np.frombuffer(
            b"".join(nodes.board[i] for i in all_ids), np.uint8
        ).reshape(-1, 68)
        np.testing.assert_array_equal(rs.boards(all_ids), boards)
        assert (
            rs.feats(all_ids[1:]) >= 0
        ).all()  # the game has clocks: every node's are known
    with pytest.raises(RuntimeError):
        rs.select()  # finished


def test_zero_clock_rule_and_views_clocks(tiny):
    """Under the "zero" rule the clocks advance by no think time; a clockless view's nodes have no clocks, a
    view with another increment its own, all from one tree."""
    game = setup(Engine(tiny), 4, 15)
    feats = np.array(game.features(), np.float32)
    root = game.sync().double().numpy()
    rs = allie_fast.Coverage(list(game.tokens), root, feats, game.inc, 8, 2.5, "zero")
    assert rs.add_view(np.full_like(feats, -1), -1) == 1
    assert rs.add_view(feats, 2) == 2
    while not rs.done:
        h = rs.select()
        if len(h):
            rs.update(np.random.default_rng(len(h)).normal(0, 1, (len(h), 2432)))
    depth1 = [
        i for i in range(1, rs.stats()["nodes"]) if rs.compact()["parent"][i] == 0
    ]
    f0, f1, f2 = (rs.feats(depth1, v) for v in range(3))
    np.testing.assert_array_equal(
        f0[:, 1], feats[-1, 0] + game.inc
    )  # nothing spent, the increment added
    np.testing.assert_array_equal(f0[:, 0], feats[-1, 1])
    assert (f1 == -1).all()
    np.testing.assert_array_equal(f2[:, 1], feats[-1, 0] + 2)
    with pytest.raises(RuntimeError):
        rs.add_view(feats, 2)  # too late
    with pytest.raises(ValueError):
        allie_fast.Coverage(list(game.tokens), root, feats, game.inc, 8, 2.5, "elapsed")
    with pytest.raises(ValueError):
        allie_fast.Coverage(list(game.tokens), root[:10], feats, game.inc, 8, 2.5)


@pytest.mark.parametrize("budget", [5, 8, 32, 128])
def test_coverage_matches_cpp_torch_backend(tiny, budget):
    engine = Engine(tiny)
    for seed, plies in ((1, 12), (2, 21), (3, 30)):
        game = setup(engine, seed, plies)
        same(treers.Coverage()(game, budget), tree.Coverage()(game, budget))


@pytest.mark.parametrize(
    "views",
    [
        ("true", "r2800"),
        ("noclock", "swap/noclock", "r3000"),
        ("tc180+2/swap/noclock",),
    ],
)
def test_coverage_views_match_cpp(tiny, views):
    game = setup(Engine(tiny), 8, 14, base=1800, inc=10)
    same(treers.Coverage(views=views)(game, 32), tree.Coverage(views=views)(game, 32))
    assert set(views) - {"true"} <= set(game.views)


def test_forced_move_and_blitz_cells(tiny):
    """One legal move: no search (budget 0), the policy of budget 128; the cell from the game's speed."""
    engine = Engine(tiny)
    game = Game(engine, 1200, 1300, 180, 2, "blitz")
    game.update(
        ["f2f3", "e7e5", "g2g4"], 178, 179
    )  # black mates in one, but White's move is not forced: a real search
    same(treers.Coverage()(game, 32), tree.Coverage()(game, 32))
    import chess

    forced = None
    for seed in range(200):
        b = chess.Board()
        moves = random_game(seed, 120)
        for i, m in enumerate(moves):
            b.push_uci(m)
            if b.legal_moves.count() == 1 and not b.is_game_over():
                forced = moves[: i + 1]
                break
        if forced:
            break
    assert forced, "a position with one legal move"
    game = Game(engine, 1200, 1300, 180, 2, "blitz")
    game.update(forced, 170, 160)
    out = treers.Coverage()(game, 32)
    same(out, tree.Coverage()(game, 32))
    assert len(out[0]) == 1 and out[1][0] == 1.0


def fast_model(tiny_path, threads):
    from allie.lichess import fast

    try:
        fast.library()
    except Exception as e:
        pytest.skip(f"no fast kernels: {e}")
    return Model(tiny_path, dtype=torch.bfloat16, backend="fast", threads=threads)


@pytest.mark.parametrize("views", [("true",), ("true", "r2800")])
def test_coverage_matches_cpp_fast_backend(tiny_path, views):
    """On the C++ fast backend the search runs on the game's thread through the engine's batches."""
    engine = Engine(fast_model(tiny_path, 2))
    for seed, plies, budget in ((5, 14, 32), (6, 23, 128)):
        game = setup(engine, seed, plies)
        same(
            treers.Coverage(views=views)(game, budget),
            tree.Coverage(views=views)(game, budget),
        )


def test_deadline(tiny_path, monkeypatch):
    """Past its deadline the search gives None (before any node, or at the nodes after), and the game searches
    again after; the nodes are accounted."""
    engine = Engine(fast_model(tiny_path, 2))
    game, search = setup(engine, 7, 14), treers.Coverage()
    calls, inner = [], tree.Nodes.__call__
    monkeypatch.setattr(
        tree.Nodes, "__call__", lambda self, h: calls.append(len(h)) or inner(self, h)
    )
    assert search(game, 32, deadline=time.monotonic() - 1) is None
    assert len(calls) == 1  # Late raised at the first nodes
    assert (
        treers.Coverage(views=("true", "r2800"))(
            game, 32, deadline=time.monotonic() - 1
        )
        is None
    )
    clock = iter(range(10**6))
    monkeypatch.setattr(tree, "monotonic", lambda: next(clock))
    assert search(game, 32, deadline=2.5) is None
    assert (
        len(calls) == 2 and calls[1] == 32
    )  # stopped within its first nodes (the chunks' clock passed 2.5)
    monkeypatch.setattr(tree, "monotonic", time.monotonic)
    out = search(game, 32)
    assert out is not None and sorted(out[0]) == sorted(
        m.uci() for m in game.board.legal_moves
    )
    assert abs(out[1].sum() - 1) < 1e-9 and abs(out[2].sum() - 1) < 1e-9


def test_kl_forests_agree_call_by_call(tiny):
    """One evaluator feeds both forests (the Rust one and kl.py's): the same handles at every call, then the same
    values, moves and priors, boards and clocks."""
    from allie.search import kl
    from allie.search.native import from_prefix

    game = setup(Engine(tiny), 11, 19)
    feats = np.array(game.features(), np.float32)
    z = game.sync()
    for budget in (8, 40):
        cap = 2 * budget + 256
        nodes = tree.Tree(game, z, cap).handles([game.tokens], [feats], "zero")
        rs = allie_fast.KL(
            list(game.tokens), [nodes.root_logits[0]], None, feats, game.inc, cap
        )
        seen = []

        def bridge(h, nodes=nodes, rs=rs, seen=seen, budget=budget):
            want = rs.select(budget, **treers.KL.GROW)
            while want is not None and not len(
                want
            ):  # a call of terminal children only: no evaluation
                want = rs.select(budget, **treers.KL.GROW)
            np.testing.assert_array_equal(h, want)
            zz = nodes(h)
            rs.update([zz])
            seen.append(h)
            return zz

        bridge.root_logits = nodes.root_logits
        F = kl.Forest(
            bridge, [from_prefix(np.asarray(game.tokens))], [len(game.tokens)], cap
        )
        kl.grow(F, budget, **treers.KL.GROW)
        while (
            h := rs.select(budget, **treers.KL.GROW)
        ) is not None:  # whatever kl.grow still took (terminal children)
            assert not len(h)
        assert (
            F.spent[0] == rs.spent == budget
            and F.size == rs.size
            and F.calls == rs.calls
        )
        V, sigma, rest = kl.backup(F.view(), **treers.KL.READ)
        for mine, theirs in zip(rs.values(**treers.KL.READ), (V, sigma, rest)):
            np.testing.assert_array_equal(mine, theirs)
        moves, prior, q = kl.root_q(F, V, 0)
        m2, p2, q2 = rs.root_q(**treers.KL.READ)
        np.testing.assert_array_equal(m2, moves)
        np.testing.assert_array_equal(p2, prior)
        fill = np.arctanh(0.95 * np.clip(F.value[0], -1, 1))
        np.testing.assert_array_equal(q2, np.where(np.isnan(q), fill, q))
        ids = [0, *np.concatenate(seen)[:, 0].tolist()]
        np.testing.assert_array_equal(
            rs.feats(ids), nodes.feats[ids].astype(np.float32)
        )
        boards = np.frombuffer(b"".join(nodes.board[i] for i in ids), np.uint8).reshape(
            -1, 68
        )
        np.testing.assert_array_equal(rs.boards(ids), boards)


@pytest.mark.parametrize("leaves", [8, 24, 64])
def test_kl_matches_python_torch_backend(tiny, leaves):
    engine = Engine(tiny)
    for seed, plies in ((1, 12), (7, 14), (3, 30)):
        game = setup(engine, seed, plies)
        same(treers.KL()(game, leaves), tree.KL()(game, leaves))


@pytest.mark.parametrize(
    "views", [("r3000/noclock",), ("r3000/tc1800+20/noclock", "true"), ("true", "true")]
)
def test_kl_views_match_python(tiny, views):
    game = setup(Engine(tiny), 9, 14, base=1800, inc=10)
    same(treers.KL(views=views)(game, 16), tree.KL(views=views)(game, 16))


def test_kl_matches_python_fast_backend(tiny_path):
    engine = Engine(fast_model(tiny_path, 2))
    for seed, plies, leaves in ((5, 14, 16), (6, 23, 48)):
        game = setup(engine, seed, plies)
        same(treers.KL()(game, leaves), tree.KL()(game, leaves))
        same(
            treers.KL(views=("r3000/noclock", "true"))(game, leaves),
            tree.KL(views=("r3000/noclock", "true"))(game, leaves),
        )
    assert treers.KL()(game, 16, deadline=time.monotonic() - 1) is None
    assert treers.KL(views=("r2800",))(game, 16, deadline=time.monotonic() - 1) is None
    with pytest.raises(ValueError):
        allie_fast.KL(
            list(game.tokens),
            [game.sync().double().numpy()],
            None,
            np.array(game.features(), np.float32),
            20,
            300,
        ).select(8, root=1.0)
