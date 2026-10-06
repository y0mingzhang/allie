"""The Rust coverage tree (allie_fast.Coverage, treers.Coverage) against today's C++ tree and Python bookkeeping
(tree.Coverage): the same handles call by call, the same compact arrays, values, boards and clocks, and the same
search outputs on the tiny model through the torch backend and natively on the Rust engine, with and without
views."""

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


def test_deadline(tiny, monkeypatch):
    """Off the Rust engine (handles through tree.Nodes): past its deadline the search gives None (before any node,
    or at the nodes after), and the game searches again after."""
    engine = Engine(tiny)
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
    assert search(game, 128, deadline=0.5) is None
    assert len(calls) == 3  # the first nodes ran (clock 0), the second raised (clock 1)
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


# --- the native loop: the trees evaluating their leaves through the Server (allie_fast.Server) ---


def test_kl_root_tilt_not_ported(tiny):
    game = setup(Engine(tiny), 5, 14)
    with pytest.raises(ValueError):
        allie_fast.KL(
            list(game.tokens),
            [game.sync().double().numpy()],
            None,
            np.array(game.features(), np.float32),
            20,
            300,
        ).select(8, root=1.0)


def rust_model(tiny_path, threads=3):
    return Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=threads)


def subtree(parent, root):
    """The ids under `root` (itself first), in id order."""
    inside, ids = np.zeros(len(parent), bool), []
    for i in range(len(parent)):
        if i == root or (parent[i] >= 0 and inside[parent[i]]):
            inside[i], _ = True, ids.append(i)
    return ids


def grandchild(cov):
    """The most visited root child and its most visited expanded child: (ids, their move tokens)."""
    c, v = cov.compact(), np.array(cov.visits())
    kids = np.flatnonzero(c["parent"] == 0)
    k1 = kids[np.argmax(v[kids])]
    gk = np.flatnonzero(c["parent"] == k1)
    gk = gk[c["degree"][gk] > 0]
    k2 = gk[np.argmax(v[gk])]
    return (k1, k2), [int(c["move"][k1]) + 378, int(c["move"][k2]) + 378]


@pytest.mark.parametrize("views", [("true",), ("true", "r2800"), ("r2800",), ("tc60+0", "true")])
def test_native_coverage_matches_handles(tiny_path, views):
    """On the Rust backend treers.Coverage runs natively (no Python per leaf): with one step per call and view
    (merged=False) its outputs are bitwise the handle-driven tree.Coverage's, a lone view other than the game's
    read as Mixed reads it and a first view's clocks on its own increment; with the views' leaves merged into one
    step the kernels' sums round differently (same moves, close values)."""
    engine = Engine(rust_model(tiny_path))
    for seed, plies, budget in ((5, 14, 32), (6, 23, 128)):
        game = setup(engine, seed, plies)
        want = tree.Coverage(views=views)(game, budget)
        before = engine.forwards
        same(treers.Coverage(views=views, merged=False)(game, budget), want)
        s = game.last_search
        assert 0 < s["evaluated"] <= budget and not s["reused"] and not s["late"]
        assert engine.forwards > before
        got = treers.Coverage(views=views)(game, budget)
        assert game.last_search["evaluated"] == s["evaluated"]
        assert got[0] == want[0] and np.abs(got[1] - want[1]).max() < 5e-3


@pytest.mark.parametrize("views", [("true",), ("r3000/noclock", "true")])
def test_native_kl_matches_handles(tiny_path, views):
    engine = Engine(rust_model(tiny_path))
    for seed, plies, leaves in ((5, 14, 16), (6, 23, 48)):
        game = setup(engine, seed, plies)
        want = tree.KL(views=views)(game, leaves)
        same(treers.KL(views=views, merged=False)(game, leaves), want)
        assert game.last_search["evaluated"] == leaves
        got = treers.KL(views=views)(game, leaves)
        assert (
            got[0] == want[0]
            and np.array_equal(got[1], want[1])
            and np.abs(got[2] - want[2]).max() < 5e-2
        )


@pytest.mark.parametrize("searcher", ["coverage", "kl"])
def test_native_searches_merge_across_games(tiny_path, searcher):
    """Concurrent native searches of several games share the Server's steps (the engine held while their first
    requests queue, so they must merge: widest > 8) and give one-at-a-time's results up to batching noise."""
    import threading

    engine = Engine(rust_model(tiny_path))
    search, budget = (
        (treers.Coverage(), 32) if searcher == "coverage" else (treers.KL(), 16)
    )
    games = [setup(engine, s, 12 + 2 * s) for s in range(4)]
    alone = [search(g, budget) for g in games]
    release, together = threading.Event(), [None] * len(games)
    blocker = threading.Thread(target=engine.run, args=(lambda: release.wait(10),))
    blocker.start()
    before = engine.server.stats()
    threads = [
        threading.Thread(
            target=lambda i=i: together.__setitem__(i, search(games[i], budget))
        )
        for i in range(len(games))
    ]
    for t in threads:
        t.start()
    time.sleep(0.5)
    assert engine.server.stats()["queued"] == len(
        games
    )  # every first request waits for the blocker
    release.set()
    for t in [blocker, *threads]:
        t.join()
    after = engine.server.stats()
    assert (
        engine.widest > 8
        and after["requests"] - before["requests"] > after["steps"] - before["steps"]
    )
    for a, b in zip(alone, together):  # coverage's probabilities, kl's prior; then the searched values
        assert a[0] == b[0] and np.abs(np.asarray(a[1]) - np.asarray(b[1])).max() < 5e-3
        assert np.abs(np.asarray(a[-1]) - np.asarray(b[-1])).max() < 5e-2


def test_native_deadline(tiny_path):
    """Past its deadline the native search gives None before its next call: at once when the deadline has passed,
    or after the call in flight when it passes during one (the Server held so the first reply comes late)."""
    import threading

    engine = Engine(rust_model(tiny_path))
    game = setup(engine, 7, 14)
    for search in (
        treers.Coverage(),
        treers.KL(),
        treers.Coverage(views=("true", "r2800")),
    ):
        assert search(game, 32, deadline=time.monotonic() - 1) is None
        assert game.last_search["late"] and game.last_search["evaluated"] == 0
    out = [None]
    release = threading.Event()
    blocker = threading.Thread(target=engine.run, args=(lambda: release.wait(10),))
    blocker.start()
    t = threading.Thread(
        target=lambda: out.__setitem__(
            0, treers.Coverage()(game, 128, deadline=time.monotonic() + 0.2)
        )
    )
    t.start()
    time.sleep(0.5)
    release.set()
    for x in (blocker, t):
        x.join()
    assert (
        out[0] is None
        and game.last_search["late"]
        and 0 < game.last_search["evaluated"] < 128
    )
    out = treers.Coverage()(game, 32)
    assert out is not None and sorted(out[0]) == sorted(
        m.uci() for m in game.board.legal_moves
    )


def test_context_end_plays_the_policy(tiny_path, monkeypatch):
    """A root with no room for a node below it (the model's context full) is not searched: None, the policy plays."""
    engine = Engine(rust_model(tiny_path))
    game = setup(engine, 7, 14)
    monkeypatch.setattr(treers, "CONTEXT", len(game.tokens))
    assert treers.Coverage()(game, 8) is None and treers.KL()(game, 8) is None


def test_pause_waits_for_the_step_in_flight(tiny_path):
    """Engine.run(fn) on the Rust backend: no step runs during fn, not even the one in flight when it was called."""
    import threading

    engine = Engine(rust_model(tiny_path))
    game, srv = setup(engine, 5, 14), engine.server
    out = []
    search = threading.Thread(target=lambda: out.append(treers.Coverage()(game, 1024)))
    search.start()
    while srv.stats()["steps"] < 3 and search.is_alive():
        time.sleep(0.001)

    def held():
        a = srv.stats()["steps"]
        time.sleep(0.02)
        return srv.stats()["steps"] - a

    held_during = [engine.run(held) for _ in range(12) if search.is_alive()]
    search.join()
    # a pause landing on a step in flight about half the time: 12 pauses all catch the old behaviour
    assert len(held_during) >= 8 and not any(held_during) and out[0] is not None


def test_kl_reroot_takes_the_new_capacity():
    """A forest built for 16 leaves re-rooted for 512 reaches 512 evaluations; re-rooted at its old capacity it
    stops short and says so (full)."""
    rng = np.random.default_rng(0)
    toks = [2348, 199, 12, 1, 5, 0, 0, 1, 5, 0, 0] + [
        378 + allie_fast.moves().index(m) for m in random_game(3, 20)
    ]
    feats = np.full((len(toks), 3), -1, np.float32)
    feats2 = np.full((len(toks) + 2, 3), -1, np.float32)

    def grow(f, budget, seen):
        while (h := f.select(budget, **treers.KL.GROW)) is not None:
            seen.extend(h.tolist())
            if len(h):
                f.update([rng.normal(0, 2, (len(h), 2432))])

    for cap, short in ((2 * 512 + 256, False), (2 * 16 + 256, True)):
        f, seen = allie_fast.KL(toks, [rng.normal(0, 2, 2432)], None, feats, -1, 2 * 16 + 256), []
        grow(f, 16, seen)
        ids = {r[0]: r for r in seen}
        g = next(r for r in seen if r[1] in ids and ids[r[1]][1] == 0)  # an evaluated grandchild
        moves = [ids[g[1]][2], g[2]]
        assert f.reroot(moves, [rng.normal(0, 2, 2432)], None, [feats2], cap) is not None and not f.full
        grow(f, 512, [])
        assert (f.spent == 512, f.full) == (not short, short)


def test_reroot_keeps_the_grandchilds_subtree(tiny_path):
    """After the two plies of its most visited line, a finished coverage tree re-roots at the grandchild: the kept
    nodes' arrays equal the old subtree's (ids remapped in order, depths less two, born reset, the root re-expanded
    from the game's logits, depth-1 priors the new root's), visits kept, the pulls and leaves accounted; then the
    search continues to the budget. A reply missing from the tree gives None (a fresh start)."""
    engine = Engine(rust_model(tiny_path))
    srv, par = engine.server, treers.Coverage().parameters
    game = setup(engine, 3, 16)
    feats = np.array(game.features(), np.float32)
    cov = allie_fast.Coverage(
        list(game.tokens),
        game.sync().double().numpy(),
        feats,
        game.inc,
        64,
        par["cpuct"],
    )
    assert cov.run(srv, [treers.ref(game.cache)])
    c, v = cov.compact(), np.array(cov.visits())
    (k1, k2), moves = grandchild(cov)
    kept_old = subtree(c["parent"], k2)
    game.update(game.moves + [tree.MOVES[t - 378] for t in moves], 1690, 1680)
    root2, feats2 = game.sync().double().numpy(), np.array(game.features(), np.float32)
    with pytest.raises(ValueError):
        cov.reroot(moves, root2, [feats], 64)  # the features of the old prefix
    kept = cov.reroot(moves, root2, [feats2], 64)
    assert kept == kept_old
    new = {o: n for n, o in enumerate(kept)}
    c2, v2 = cov.compact(), np.array(cov.visits())
    assert c2["parent"][0] == -1 and all(
        c2["parent"][n] == new[c["parent"][o]] for n, o in enumerate(kept) if n
    )
    np.testing.assert_array_equal(c2["depth"], c["depth"][kept] - 2)
    assert (c2["born"] == 0).all() and (c2["move"][1:] == c["move"][kept][1:]).all()
    for k in ("degree", "terminal"):
        np.testing.assert_array_equal(c2[k], c[k][kept])
    for k in ("boot", "mass"):
        np.testing.assert_array_equal(c2[k][1:], c[k][kept][1:])
    deeper = c2["depth"] >= 2
    np.testing.assert_array_equal(c2["prior"][deeper], c["prior"][kept][deeper])
    legal = allie_fast.Position.from_tokens(game.tokens).legal()
    prior = np.exp(
        root2[legal].astype(np.float32).astype(np.float64)
        - root2[legal].astype(np.float32).max()
    )
    prior /= prior.sum()
    for n in np.flatnonzero(c2["depth"] == 1):
        want = prior[legal.index(int(c2["move"][n]) + 378)]
        assert abs(c2["prior"][n] - want) <= 1e-15
    np.testing.assert_array_equal(v2[1:], v[kept][1:])
    s = cov.stats()
    assert v2[0] == s["kept_pulls"] == v[k2] - 1 == v2[c2["parent"] == 0].sum()
    assert (
        s["reused_leaves"] == int((c["degree"][kept_old] > 0).sum()) - 1
        and s["evaluated_leaves"] == 0
    )
    assert cov.run(srv, [treers.ref(game.cache)])
    s = cov.stats()
    assert (
        s["evaluated_leaves"] + s["terminal_visits"] + s.get("depth_limited_visits", 0)
        == 64 - s["kept_pulls"]
    )
    assert len(cov.compact()["parent"]) >= len(kept) + s["evaluated_leaves"]
    q = cov.reduce(
        np.log(par["backup"]["tau"]),
        par["backup"]["exponent"],
        par["backup"]["count_scale"],
        64,
        np.array([legal]) - 378,
    )
    assert np.isfinite(q).all()
    with pytest.raises(ValueError):
        cov.grow(128)  # the schedule cannot be extended after a re-root
    # the bot's move in the tree, a reply that is not: nothing to keep
    c3 = cov.compact()
    k1 = np.flatnonzero(c3["parent"] == 0)[0]
    other = setup(engine, 3, 16)
    other.update(game.moves + [tree.MOVES[c3["move"][k1]]], 1680, 1680)
    seen = set(c3["move"][c3["parent"] == k1])
    reply = next(
        t
        for t in allie_fast.Position.from_tokens(other.tokens).legal()
        if t - 378 not in seen
    )
    other.update(other.moves + [tree.MOVES[reply - 378]], 1680, 1670)
    assert (
        cov.reroot(
            [int(c3["move"][k1]) + 378, reply],
            other.sync().double().numpy(),
            [np.array(other.features(), np.float32)],
            64,
        )
        is None
    )


def test_reuse_through_the_searchers(tiny_path):
    """treers.Coverage(reuse=True) / KL(reuse=True) keep a game's trees and re-root them when the game follows the
    tree; a game off the tree (or a KL grandchild never expanded) starts fresh; reuse off keeps nothing."""
    engine = Engine(rust_model(tiny_path))
    search = treers.Coverage(reuse=True)
    game = setup(engine, 4, 15)
    out = search(game, 64)
    assert out is not None and game.last_search["reused"] == 0
    cov = game.trees[("coverage", ("true",))][0]
    _, moves = grandchild(cov)
    game.update(game.moves + [tree.MOVES[t - 378] for t in moves], 1690, 1680)
    out = search(game, 64)
    s = game.last_search
    assert (
        out is not None
        and s["reused"] >= 1
        and s["kept_pulls"] >= 1
        and s["evaluated"] <= 64 - s["kept_pulls"]
    )
    assert (
        sorted(out[0]) == sorted(m.uci() for m in game.board.legal_moves)
        and abs(out[1].sum() - 1) < 1e-9
    )
    assert game.trees[("coverage", ("true",))][0] is cov  # the same tree, re-rooted
    _, moves = grandchild(cov)
    game.update(game.moves + [tree.MOVES[t - 378] for t in moves], 1680, 1670)
    game.clocks[-3] = (
        1600  # a clock learnt late revises a row the tree was searched on: no reuse
    )
    search(game, 64)
    assert (
        game.last_search["reused"] == 0
        and game.trees[("coverage", ("true",))][0] is not cov
    )
    cov = game.trees[("coverage", ("true",))][0]
    game.update(game.moves[:-1], 1690, 1680)  # a takeback: the prefix no longer matches
    search(game, 64)
    assert (
        game.last_search["reused"] == 0
        and game.trees[("coverage", ("true",))][0] is not cov
    )
    kl = treers.KL(reuse=True)
    game = setup(engine, 6, 15)
    kl(game, 64)
    forest = game.trees[("kl", ("true",), "zero")][0]
    tokens, _, _ = forest.root_q(**treers.KL.READ)
    game.update(game.moves + [tree.MOVES[tokens[0] - 378]], 1690, 1680)
    game.update(game.moves + [next(iter(game.board.legal_moves)).uci()], 1690, 1670)
    out = kl(game, 64)
    assert (
        out is not None
        and game.last_search["evaluated"] + game.last_search["reused"] == 64
    )
    plain = treers.Coverage()
    game = setup(engine, 8, 15)
    plain(game, 32)
    assert "trees" not in game.__dict__ or ("coverage", ("true",)) not in game.trees


@pytest.mark.parametrize("moves_first", [True, False])
def test_moves_go_first(tiny_path, moves_first):
    """A game's move (a plain-only request) queued behind another game's search leaves runs first, in a step of its
    own (moves_first, the default); without it the two merge into one step."""
    import threading

    engine = Engine(rust_model(tiny_path))
    srv = engine.server
    srv.moves_first = moves_first
    searching, mover = setup(engine, 3, 14), setup(engine, 4, 15)
    z, feats = (
        searching.sync().double().numpy(),
        np.array(searching.features(), np.float32),
    )
    par = treers.Coverage().parameters
    first = len(
        allie_fast.Coverage(
            list(searching.tokens), z, feats, searching.inc, 32, par["cpuct"]
        ).select()
    )
    mover.sync()
    mover.update(mover.moves + [next(iter(mover.board.legal_moves)).uci()], 1680, 1680)
    release = threading.Event()
    blocker = threading.Thread(target=engine.run, args=(lambda: release.wait(10),))
    blocker.start()
    time.sleep(0.1)
    srv.sizes()
    search = threading.Thread(target=treers.Coverage(), args=(searching, 32))
    search.start()
    time.sleep(0.3)
    move = threading.Thread(target=mover.sync)
    move.start()
    time.sleep(0.3)
    assert srv.stats()["queued"] == 2
    release.set()
    for t in (blocker, search, move):
        t.join()
    sizes = srv.sizes()
    assert first > 1 and sizes[:2] == (
        [1, first] if moves_first else [first + 1, sizes[1]]
    )


def test_native_errors_raise(tiny_path):
    """A step the engine refuses surfaces as a RuntimeError naming the step's leaves, and the server keeps serving;
    a request whose spans do not fit is refused before it is queued."""
    engine = Engine(rust_model(tiny_path))
    game = setup(engine, 7, 14)
    z, feats = game.sync().double().numpy(), np.array(game.features(), np.float32)
    cov = allie_fast.Coverage(list(game.tokens), z, feats, game.inc, 32, 2.5)
    k, v, e, cap, n = treers.ref(game.cache)
    with pytest.raises(
        RuntimeError, match=r"a search step of \d+ leaves: tokens, paths or slots"
    ):
        cov.run(
            engine.server, [(k, v, e, cap, cap + 1)]
        )  # more rows than the cache holds
    ids = torch.zeros(1, dtype=torch.int64)
    meta = [0, 2**63 - 1, cap, 0, 0, 0, 0, 0]
    assert (
        engine.server.step(1, 1, ids.data_ptr(), 0, 0, meta, [k, v, e, 0, 0, 0], [], 0)
        == 3
    )
    out = treers.Coverage()(game, 8)
    assert out is not None and engine.server.stats()["queued"] == 0
    handled = allie_fast.Coverage(list(game.tokens), z, feats, game.inc, 32, 2.5)
    h = handled.select()
    handled.update(np.zeros((len(h), 2432)))
    with pytest.raises(RuntimeError, match="select"):  # its nodes have no slot rows
        handled.run(engine.server, [treers.ref(game.cache)])
    forest = allie_fast.KL(list(game.tokens), [z], None, feats, game.inc, 300)
    h = forest.select(16, **treers.KL.GROW)
    forest.update([np.zeros((len(h), 2432))])
    with pytest.raises(RuntimeError, match="select"):
        forest.run(engine.server, [treers.ref(game.cache)], 16, **treers.KL.GROW)
    native = allie_fast.Coverage(list(game.tokens), z, feats, game.inc, 8, 2.5)
    assert native.run(engine.server, [treers.ref(game.cache)])
    with pytest.raises(RuntimeError, match="run"):
        native.select()
    fast, real = engine.model.fast, engine.model.fast.handle
    item = (game.cache, torch.tensor([400]), torch.zeros(1, 3), torch.zeros(1, 68, dtype=torch.uint8))
    for code, kind, why in ((4, ValueError, "alias"), (5, RuntimeError, "panic"), (-1, RuntimeError, "stopped")):
        fast.handle = type("Stub", (), {"step": lambda self, *a, c=code: c})()
        try:
            with pytest.raises(kind, match=why):
                fast._step([item])
        finally:
            fast.handle = real


def test_cli_picks_the_native_searcher(tiny_path, monkeypatch):
    """The bot's searcher on the Rust backend is treers.Coverage, and it searches without the C++ tree; on torch
    tree.Coverage."""
    from allie.lichess import cli
    from allie.lichess.config import Config, Play

    import allie.search.native as cpp

    def refuse(kind):
        raise AssertionError(f"the C++ {kind} module was loaded")

    c = Config(model=str(tiny_path), threads=2, backend="rust", play=Play(mode="calibrated"))
    reference = Model(tiny_path, backend="torch")
    assert cli.coverage(c, reference) is None  # calibrated searches only on the engine
    strongest = Config(model=str(tiny_path), play=Play(mode="strongest", search=8))
    assert isinstance(cli.coverage(strongest, reference), tree.Coverage)
    before = torch.get_num_threads()
    try:
        m = cli.model(c)
    finally:
        torch.set_num_threads(before)
    search = cli.coverage(c, m)
    assert isinstance(search, treers.Coverage)
    monkeypatch.setattr(cpp, "_load", refuse)
    game = setup(Engine(m), 5, 14)
    assert search(game, 32) is not None and not game.last_search["late"]

