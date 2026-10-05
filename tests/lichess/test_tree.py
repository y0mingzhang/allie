import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from allie.lichess.engine import Engine, Game, Play
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import MOVE_ID, header

from .test_tokens import random_game

tree = pytest.importorskip("allie.lichess.tree")


def fast_model(tiny_path, backend, threads):
    """A tiny Model on the C++ or the Rust CPU backend, or skip."""
    from allie.lichess import fast

    try:
        fast.library()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no fast kernels: {e}")
    if backend == "rust":
        pytest.importorskip("allie_fast")
    return Model(tiny_path, dtype=torch.bfloat16, backend=backend, threads=threads)


def test_nodes_match_prefill(tiny):
    game = Game(Engine(tiny), 1500, 1600, 180, 2)
    game.update(random_game(4, 21), 170, 165)
    feats = np.array(game.features(), np.float32)
    nodes = tree.Tree(game, game.sync(), 16).handles([game.tokens], [feats])
    n, tok = len(game.tokens), lambda m: MOVE_ID[m.uci()]
    first = list(game.board.legal_moves)[:2]
    replies = []
    for m in first:
        b = game.board.copy()
        b.push(m)
        replies.append(next(iter(b.legal_moves)))
    z1 = nodes([[1, 0, tok(first[0]), n + 1], [2, 0, tok(first[1]), n + 1]])
    z2 = nodes([[3, 1, tok(replies[0]), n + 2], [4, 2, tok(replies[1]), n + 2]])
    for path, z in zip(([1], [2], [1, 3], [2, 4]), (*z1, *z2)):
        ids = torch.tensor(game.tokens + [int(nodes.token[i]) for i in path])
        f = torch.tensor(game.features() + [nodes.feats[i].tolist() for i in path])
        b = b"".join(game.boards + [nodes.board[i] for i in path])
        b = torch.tensor(np.frombuffer(b, np.uint8).reshape(-1, 68))
        ref = step(tiny, [(Cache(tiny), ids, f.float(), b)])[0]
        np.testing.assert_allclose(z, ref.double().numpy(), atol=3e-5)


native = Path(os.environ.get("ALLIE_CHESS_INCLUDE", str(Path(tree.__file__).parents[1] / "search/native/chess-library")))


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
def test_coverage_search_plays_legal_moves(tiny):
    pytest.importorskip("pybind11")
    game = Game(Engine(tiny), 2800, 2800, 180, 2)
    game.update(random_game(6, 12), 175, 176)
    search = tree.Coverage()
    play = Play(mode="strongest", search=5)
    moves, p = search(game, 5)[:2]
    assert sorted(moves) == sorted(m.uci() for m in game.board.legal_moves)
    assert abs(p.sum() - 1) < 1e-9
    assert game.decide(play, search).move in moves


def test_ragged_paths_ignore_unwritten_slots(tiny):
    game = Game(Engine(tiny), 1500, 1600, 180, 2)
    game.update(random_game(8, 15), 170, 165)
    feats = np.array(game.features(), np.float32)
    t = tree.Tree(game, game.sync(), capacity=16)
    t.k[:, :, 1:], t.v[:, :, 1:] = float("nan"), float("nan")  # never-written slots
    nodes = t.handles([game.tokens], [feats])
    n, tok = len(game.tokens), lambda m: MOVE_ID[m.uci()]
    first, second = list(game.board.legal_moves)[:2]
    b = game.board.copy()
    b.push(first)
    reply = next(iter(b.legal_moves))
    nodes([[1, 0, tok(first), n + 1]])
    z = nodes([[2, 1, tok(reply), n + 2], [3, 0, tok(second), n + 1]])  # depths 2 and 1
    assert np.isfinite(z).all()


def node_logits(model):
    """Logits of two depth-1 nodes and their depth-2 children, and each node's full sequence:
    (z [4, 2432], [(ids, feats, boards)])."""
    game = Game(Engine(model), 1500, 1600, 180, 2)
    game.update(random_game(4, 21), 170, 165)
    feats = np.array(game.features(), np.float32)
    nodes = tree.Tree(game, game.sync(), 16).handles([game.tokens], [feats])
    n, tok = len(game.tokens), lambda m: MOVE_ID[m.uci()]
    first = list(game.board.legal_moves)[:2]
    replies = []
    for m in first:
        b = game.board.copy()
        b.push(m)
        replies.append(next(iter(b.legal_moves)))
    z1 = nodes([[1, 0, tok(first[0]), n + 1], [2, 0, tok(first[1]), n + 1]])
    z2 = nodes([[3, 1, tok(replies[0]), n + 2], [4, 2, tok(replies[1]), n + 2]])
    seqs = []
    for path in ([1], [2], [1, 3], [2, 4]):
        ids = torch.tensor(game.tokens + [int(nodes.token[i]) for i in path])
        f = torch.tensor(game.features() + [nodes.feats[i].tolist() for i in path]).float()
        b = b"".join(game.boards + [nodes.board[i] for i in path])
        seqs.append((ids, f, torch.tensor(np.frombuffer(b, np.uint8).reshape(-1, 68))))
    return torch.tensor(np.concatenate([z1, z2])), seqs


@pytest.mark.parametrize("backend", ["fast", "rust"])
def test_fast_nodes_match_reference(tiny_path, backend, tiny):
    """The fast backend's nodes (the game's cache copied, the path appended) differ from the
    reference tree's by BF16 rounding only: no further from an FP32 prefill than the reference."""
    model = fast_model(tiny_path, backend, 3)
    assert model.fast is not None
    got, seqs = node_logits(model)
    ref, _ = node_logits(Model(tiny_path, dtype=torch.bfloat16, backend="torch"))
    hi = torch.cat([step(tiny, [(Cache(tiny), *s)]) for s in seqs])
    assert (got - hi).abs().mean() < 1.25 * (ref - hi).abs().mean() + 1e-3
    assert (got - hi).abs().max() < 2 * (ref - hi).abs().max() + 0.02
    p, q = (torch.softmax(z[:, 378:2346].float(), -1) for z in (got, ref))
    assert (p - q).abs().max() < 1e-3


def test_rust_leaves_match_copied_nodes(tiny_path):
    """On the Rust backend Nodes.fast evaluates a call's nodes in place (Leaf items, one step); the same nodes
    through the copying code (Nodes.copied, one step too) give bitwise the same logits and slot rows."""
    model = fast_model(tiny_path, "rust", 3)
    game = Game(Engine(model), 1500, 1600, 180, 2)
    game.update(random_game(4, 21), 170, 165)
    feats, n = np.array(game.features(), np.float32), len(game.tokens)
    trees = [tree.Tree(game, game.sync(), 64) for _ in range(2)]
    a, b = (t.handles([game.tokens], [feats]) for t in trees)
    b.fast = lambda ids, chunk=8: b.copied(ids, len(ids))  # today's way, the same batch
    boards, calls, next_id, frontier = {0: game.board.copy()}, [], 1, [0]
    for depth in range(1, 6):  # every frontier node's first two legal moves, one call a depth
        handles, grown = [], []
        for p in frontier:
            for mv in list(boards[p].legal_moves)[:2]:
                boards[next_id] = boards[p].copy()
                boards[next_id].push(mv)
                handles.append([next_id, p, MOVE_ID[mv.uci()], n + depth])
                grown.append(next_id)
                next_id += 1
        frontier = grown[: 8 // depth + 1]
        calls.append(handles)
    for h in calls:
        za, zb = a(h), b(h)
        assert np.array_equal(za, zb) and np.isfinite(za).all(), f"depth {h[0][3] - n}"
    s = slice(1, next_id)
    x, y = trees
    assert torch.equal(x.k[:, :, s], y.k[:, :, s]) and torch.equal(x.v[:, :, s], y.v[:, :, s]) and torch.equal(x.e[s], y.e[s])
    assert game.cache.n == n and next_id > 20 and not x.caches and len(y.caches) == max(map(len, calls))


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
@pytest.mark.parametrize("backend", ["fast", "rust"])
def test_concurrent_searches_match_sequential(tiny_path, backend):
    """On the fast backend, searches run on the games' threads and their nodes share the engine's
    batches with other games' moves: the same results as one at a time, up to BF16 batching noise.
    The engine is held busy while the requests queue, so they must merge."""
    pytest.importorskip("pybind11")
    import threading

    model = fast_model(tiny_path, backend, 3)
    engine, search = Engine(model), tree.Coverage()

    def setup(s):
        g = Game(engine, 2400, 2400, 1800, 20, "classical")
        g.update(random_game(s, 12 + 2 * s), 1700, 1690)
        return g

    games = [setup(s) for s in range(4)]
    alone = [search(g, 32) for g in games]
    other = setup(9)  # a game whose move arrives during the searches
    moves = other.moves + [next(iter(other.board.legal_moves)).uci()]
    ref = setup(9)
    ref.update(moves, 1690, 1690)
    expected = ref.sync()
    other.sync()
    other.update(moves, 1690, 1690)
    release, together, got = threading.Event(), [None] * len(games), []
    blocker = threading.Thread(target=engine.run, args=(lambda: release.wait(10),))
    blocker.start()

    def go(i):
        together[i] = search(games[i], 32)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(games))]
    threads.append(threading.Thread(target=lambda: got.append(other.sync())))
    for t in threads:
        t.start()
    time.sleep(0.5)  # every first request is queued behind the blocker
    release.set()
    for t in [blocker, *threads]:
        t.join()
    assert engine.widest > 8  # more than one search chunk in a forward: requests merged
    for (m1, p1, *_), (m2, p2, *_) in zip(alone, together):
        assert m1 == m2
        assert np.abs(np.asarray(p1) - np.asarray(p2)).max() < 2e-3
    p, q = (torch.softmax(z[378:2346].float(), -1) for z in (got[0], expected))
    assert (p - q).abs().max() < 1e-3


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
@pytest.mark.parametrize("backend", ["fast", "rust"])
def test_a_search_past_its_deadline_stops(tiny_path, backend, monkeypatch):
    """A search past its deadline stops at its next nodes and gives no distribution; the game
    searches again after."""
    pytest.importorskip("pybind11")
    model = fast_model(tiny_path, backend, 2)
    game, search = Game(Engine(model), 2400, 2400, 1800, 20, "classical"), tree.Coverage()
    game.update(random_game(5, 14), 1700, 1690)
    calls, inner = [], tree.Nodes.__call__
    monkeypatch.setattr(tree.Nodes, "__call__", lambda self, h: calls.append(len(h)) or inner(self, h))
    assert search(game, 32, deadline=time.monotonic() - 1) is None
    assert len(calls) == 1
    steps, clock = [], iter(range(10**6))
    monkeypatch.setattr(game.engine, "steps", lambda items, f=game.engine.steps: steps.append(1) or f(items))
    monkeypatch.setattr(tree, "monotonic", lambda: next(clock))  # nodes 0, their chunks 1, 2, ...
    assert search(game, 32, deadline=1.5) is None
    assert calls[1] > 8 and len(steps) == 1  # stopped within the first nodes, after one chunk
    monkeypatch.setattr(tree, "monotonic", time.monotonic)
    moves, p = search(game, 32)[:2]
    assert len(calls) > 3
    assert sorted(moves) == sorted(m.uci() for m in game.board.legal_moves) and abs(p.sum() - 1) < 1e-9


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
def test_lookahead_values_every_move(tiny):
    pytest.importorskip("pybind11")
    game = Game(Engine(tiny), 2400, 2500, 1800, 10)
    game.update(random_game(7, 14), 1700, 1690)
    moves, prior, q = tree.Lookahead(m=4, k=2)(game, 3)
    assert sorted(moves) == sorted(m.uci() for m in game.board.legal_moves)
    assert abs(prior.sum() - 1) < 1e-9 and np.isfinite(q).all() and (np.abs(q) <= 1).all()


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
@pytest.mark.parametrize("backend", ["fast", "rust"])
def test_fast_lookahead_runs_concurrently_and_stops_late(tiny_path, backend):
    """On the fast backend lookahead runs on the games' threads through the engine's batches: the
    same values as one at a time, up to BF16 batching noise, with the engine held busy while the
    requests queue so they must merge; past its deadline it gives None."""
    pytest.importorskip("pybind11")
    import threading

    engine, search = Engine(fast_model(tiny_path, backend, 3)), tree.Lookahead(m=4, k=2)
    games = []
    for s in range(3):
        games.append(Game(engine, 2400, 2400, 1800, 20, "classical"))
        games[-1].update(random_game(s, 12 + 2 * s), 1700, 1690)
    alone, together = [search(g, 4) for g in games], [None] * len(games)
    release = threading.Event()
    blocker = threading.Thread(target=engine.run, args=(lambda: release.wait(10),))
    blocker.start()

    def go(i):
        together[i] = search(games[i], 4)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(games))]
    for t in threads:
        t.start()
    time.sleep(0.5)  # every first request is queued behind the blocker
    release.set()
    for t in [blocker, *threads]:
        t.join()
    assert engine.widest > 8  # more than one game's nodes in a forward: requests merged
    for (m1, p1, q1), (m2, p2, q2) in zip(alone, together):
        assert m1 == m2 and np.abs(p1 - p2).max() < 1e-9 and np.abs(q1 - q2).max() < 1e-2
    assert search(games[0], 4, deadline=time.monotonic() - 1) is None


def test_mixed_averages_wdl():
    rng = np.random.default_rng(0)
    zs = [rng.normal(0, 3, (5, 2432)) for _ in range(3)]
    z = tree.mixed(zs)
    soft = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
    np.testing.assert_allclose(soft(z[:, tree.WDL]), np.mean([soft(x[:, tree.WDL]) for x in zs], 0), atol=1e-12)
    np.testing.assert_array_equal(np.delete(z, np.r_[tree.WDL], 1), np.delete(zs[0], np.r_[tree.WDL], 1))


def test_view_follows_a_takeback(tiny):
    game, moves, clock = Game(Engine(tiny), 2400, 2500, 180, 2), random_game(5, 12), [180, 180]
    for k in range(1, len(moves) + 1):  # every move's clock known: a takeback keeps the earlier features
        clock[(k - 1) % 2] -= 3
        game.update(moves[:k], *clock)
    view = tree.View(game, "r2800")
    view.sync()
    game.update(moves[:-2], *clock)
    fresh = tree.View(game, "r2800")
    assert view.tokens == fresh.tokens and fresh.tokens[3:11] == [2, 8, 0, 0, 2, 8, 0, 0]
    np.testing.assert_allclose(view.sync(), fresh.sync(), atol=1e-4)  # incremental vs one prefill
    assert view.cache.n == len(game.tokens)


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
def test_coverage_views_play_legal_moves(tiny):
    pytest.importorskip("pybind11")
    game = Game(Engine(tiny), 2400, 2500, 1800, 10)
    game.update(random_game(8, 14), 1700, 1690)
    alone = tree.Coverage()(game, 8)
    for views in (("true", "r2800"), ("noclock", "swap/noclock", "r3000"), ("r3000/noclock", "tc180+2/swap/noclock")):
        moves, p, prior, q = tree.Coverage(views=views)(game, 8)
        assert sorted(moves) == sorted(m.uci() for m in game.board.legal_moves)
        assert abs(p.sum() - 1) < 1e-9 and abs(prior.sum() - 1) < 1e-9 and np.isfinite(q).all()
        assert alone[0] == moves and np.allclose(alone[2], prior)  # the game's own prior: only the values mix
    assert {"r2800", "noclock", "swap/noclock", "r3000"} <= set(game.views)


def test_view_specs(tiny):
    game = Game(Engine(tiny), 2400, 2500, 1800, 10)
    game.update(random_game(8, 6), 1700, 1690)
    own = game.tokens[3:11]
    assert tree.View(game, "swap").tokens[3:11] == own[4:] + own[:4]
    assert tree.View(game, "r3000/noclock").tokens[3:11] == [3, 0, 0, 0] * 2
    assert tree.View(game, "swap").features() == game.features()
    assert all(f == [-1] * 3 for f in tree.View(game, "swap/noclock").features())
    assert any(f != [-1] * 3 for f in game.features())
    blitz = tree.View(game, "tc180+2/swap/noclock").tokens
    assert blitz[1:3] == header(180, 2, 0, 0)[1:3] and blitz[3:11] == own[4:] + own[:4] and blitz[11:] == game.tokens[11:]
    assert tree.View(game, "tc180+2/noclock").inc == 2 and tree.View(game, "swap").inc == game.inc == 10
    with pytest.raises(ValueError):
        tree.View(game, "tc180")


def test_coverage_views_respect_the_deadline_in_sync(tiny):
    game = Game(Engine(tiny), 2400, 2500, 1800, 10)
    game.update(random_game(8, 14), 1700, 1690)
    assert tree.Coverage(views=("true", "r2800"))(game, 8, deadline=time.monotonic() - 1) is None


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
def test_kl_values_searched_moves(tiny, monkeypatch):
    """The KL searcher spends exactly its leaves on the bot's cache and reads the tree it grew: every legal
    move, Q on the log-odds scale, unexpanded moves at the root's own value."""
    pytest.importorskip("pybind11")
    from allie.search import kl

    game = Game(Engine(tiny), 2400, 2500, 1800, 10)
    game.update(random_game(7, 14), 1700, 1690)
    grown, grow = [], kl.grow
    monkeypatch.setattr(kl, "grow", lambda F, *a, **k: grown.append(F) or grow(F, *a, **k))
    search = tree.KL()
    moves, prior, q = search(game, 24)
    F = grown[0]
    assert F.spent[0] == 24 and sorted(moves) == sorted(m.uci() for m in game.board.legal_moves)
    assert abs(prior.sum() - 1) < 1e-9 and np.isfinite(q).all() and (np.abs(q) <= np.arctanh(0.95)).all()
    kid = F.ekid[F.start[0] : F.start[0] + F.count[0]]
    V = kl.backup(F.view(), **search.read)[0]
    assert np.allclose(q[kid >= 0], -V[kid[kid >= 0]])
    assert np.allclose(q[kid < 0], np.arctanh(0.95 * F.value[0]))


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
def test_kl_views_read_every_node_under_them(tiny, monkeypatch):
    """KL views as Coverage's: the tree is the first view's, its W/D/L the views' mean, the root's prior the
    game's own. Two readings of the game itself give the plain search."""
    pytest.importorskip("pybind11")
    from allie.search import kl

    game, played = Game(Engine(tiny), 2400, 2500, 1800, 10), random_game(9, 14)
    game.update(played, 1700, 1690)
    grown, grow = [], kl.grow
    monkeypatch.setattr(kl, "grow", lambda F, *a, **k: grown.append(F) or grow(F, *a, **k))
    alone = tree.KL()(game, 16)
    twice = tree.KL(views=("true", "true"))(game, 16)
    assert alone[0] == twice[0] and np.allclose(alone[1], twice[1]) and np.allclose(alone[2], twice[2], atol=1e-9)
    soft = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    for views in (("r3000/noclock",), ("r3000/tc1800+20/noclock", "true")):
        moves, prior, q = tree.KL(views=views)(game, 16)
        F = grown[-1]
        assert F.spent[0] == 16 and sorted(moves) == sorted(m.uci() for m in game.board.legal_moves)
        assert alone[0] == moves and np.allclose(alone[1], prior) and np.isfinite(q).all()
        view = game.views[views[0]].sync().double().numpy()
        np.testing.assert_allclose(F.wdlv[0, 0], soft(view[tree.WDL]), atol=1e-12)
        assert not np.allclose(F.wdlv[0, 0], soft(game.sync().double().numpy()[tree.WDL]))
    tree.KL(views=("r3000/noclock", "true"))(game, 16)
    F = grown[-1]
    live = ~F.terminal[: F.size]
    np.testing.assert_allclose(F.wdl[: F.size][live], F.wdlv[: F.size][live].mean(1), atol=1e-12)
    for c in np.flatnonzero((F.parent[: F.size] == 0) & live)[:4]:  # below the root, each node read as the view reads it
        g = Game(game.engine, 2400, 2500, 1800, 10)
        g.update(played + [tree.MOVES[int(F.token[c]) - tree.MOVE_START]], 1700, 1690)
        np.testing.assert_allclose(F.wdlv[c, 0], soft(tree.View(g, "r3000/noclock").sync().double().numpy()[tree.WDL]), atol=1e-4)
    calls, inner = [], tree.Nodes.__call__
    monkeypatch.setattr(tree.Nodes, "__call__", lambda self, h: calls.append(len(h)) or inner(self, h))
    assert tree.KL(views=("r2800",))(game, 8, deadline=time.monotonic() - 1) is None and not calls


@pytest.mark.skipif(
    not (native / "chess.hpp").exists() or not shutil.which("c++"),
    reason="native search",
)
@pytest.mark.parametrize("views", [("true",), ("r3000/noclock", "true")])
@pytest.mark.parametrize("backend", ["fast", "rust"])
def test_fast_kl_runs_concurrently_and_stops_late(tiny_path, backend, views):
    """As lookahead: on the fast backend KL runs on the games' threads through the engine's batches, the same
    values as one at a time up to BF16 batching noise; past its deadline it gives None."""
    pytest.importorskip("pybind11")
    import threading

    engine, search = Engine(fast_model(tiny_path, backend, 3)), tree.KL(views=views)
    games = []
    for s in range(3):
        games.append(Game(engine, 2400, 2400, 1800, 20, "classical"))
        games[-1].update(random_game(s, 12 + 2 * s), 1700, 1690)
    alone, together = [search(g, 16) for g in games], [None] * len(games)
    release = threading.Event()
    blocker = threading.Thread(target=engine.run, args=(lambda: release.wait(10),))
    blocker.start()

    def go(i):
        together[i] = search(games[i], 16)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(games))]
    for t in threads:
        t.start()
    time.sleep(0.5)
    release.set()
    for t in [blocker, *threads]:
        t.join()
    assert engine.widest > 8
    for (m1, p1, q1), (m2, p2, q2) in zip(alone, together):
        assert m1 == m2 and np.abs(p1 - p2).max() < 1e-9 and np.abs(q1 - q2).max() < 5e-2
    assert search(games[0], 16, deadline=time.monotonic() - 1) is None
