import numpy as np
import pytest
import torch

from allie.lichess import calibration
from allie.lichess.engine import Engine, Game, Play
from allie.lichess.model import Model

from .test_tokens import random_game


def test_rule():
    quick, long_think = np.zeros(63), np.zeros(63)
    quick[2], long_think[-1] = 1, 1  # 2.5 s; over an hour
    assert calibration.budget(800, quick) < calibration.budget(2600, quick) <= calibration.HW * 2.5
    assert calibration.budget(2600, long_think) == calibration.CAP
    assert calibration.budget(2600, long_think, speed="bullet") == 0


def test_the_clock_caps_the_rung_searched():
    """The hard limit applies to the rung actually searched, after the random rounding: a tenth of
    the clock above the reserve (review: a 0.2 s clock could round up to a search)."""
    rng = np.random.default_rng(0)
    for clock, most in ((0.2, 0), (1.0, 0), (10.0, 32), (None, calibration.CAP)):
        cap = calibration.ceiling(clock, reserve=1.0)
        draws = {calibration.pick(n, rng, cap) for n in (0, 5, 40, 200, 256) for _ in range(500)}
        assert max(draws) <= most and max(draws) * 1.0 / calibration.HW <= (clock or 1e9) / 10 + 1e-9


def test_pick_rounds_onto_the_ladder_in_expectation():
    rng = np.random.default_rng(0)
    for n in (0, 3, 8, 20, 100, 256):
        draws = np.array([calibration.pick(n, rng) for _ in range(20000)])
        assert set(draws) <= set(calibration.LADDER)
        assert np.log2(1 + draws).mean() == pytest.approx(np.log2(1 + n), abs=0.03)


class Search:
    """Records budgets; returns the legal moves with the last one certain."""

    def __init__(self):
        self.budgets = []

    def __call__(self, game, n):
        self.budgets.append(n)
        moves = [m.uci() for m in game.board.legal_moves]
        return moves, np.eye(len(moves))[-1]


def test_calibrated_mode(tiny_path, tiny, monkeypatch):
    legal_of = lambda g: [m.uci() for m in g.board.legal_moves]
    play = Play(mode="calibrated", think_time=False, resign=False, draws=False)
    search = Search()
    game = Game(Engine(tiny), 2000, 2000, 600, 5, "rapid", seed=0)  # FP32: the reference backend
    game.update(random_game(3, 20), 590, 585)
    monkeypatch.setattr(calibration, "K", 1e6)
    assert game.decide(play, search, 590).move in legal_of(game)
    assert search.budgets == []  # no search off the fast backend
    from allie.lichess import fast

    try:
        fast.library()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no fast kernels: {e}")
    game = Game(Engine(Model(tiny_path, dtype=torch.bfloat16, backend="fast", threads=2)), 2000, 2000, 600, 5, "rapid", seed=0)
    game.update(random_game(3, 20), 590, 585)
    assert game.decide(play, None, 590).move in legal_of(game)  # no searcher: the policy
    assert game.decide(play, search, 590).move == legal_of(game)[-1]  # the cap: the searched distribution
    assert search.budgets == [calibration.CAP]
    monkeypatch.setattr(calibration, "K", 0.0)
    game.decide(play, search, 590)
    assert search.budgets == [calibration.CAP]  # budget 0: no search


def test_close_serves_queued_requests(tiny):
    """close() behind queued work: every request is answered, then the inference thread ends."""
    import threading
    import time

    engine, release, got = Engine(tiny), threading.Event(), []
    blocker = threading.Thread(target=engine.run, args=(lambda: release.wait(10),), daemon=True)
    blocker.start()
    waiter = threading.Thread(target=lambda: got.append(engine.run(lambda: 42)), daemon=True)
    waiter.start()
    time.sleep(0.2)
    closer = threading.Thread(target=engine.close, daemon=True)
    closer.start()
    time.sleep(0.2)
    release.set()
    for t in (blocker, waiter, closer):
        t.join(5)
    assert got == [42] and not engine.thread.is_alive()
