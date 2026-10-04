import time

import numpy as np
import pytest
import torch

from allie.lichess import behaviour, calibration
from allie.lichess.engine import Engine, Game, Play
from allie.lichess.model import Model

from .test_tokens import random_game


def test_cells_bin_the_rating(monkeypatch):
    low, mid, high = ("coverage", 32, None), ("coverage", 128, None), ("lookahead", 8, 4.0)
    monkeypatch.setattr(calibration, "CELLS", {("blitz", 800): low, ("blitz", 2000): mid, ("blitz", 2600): high})
    assert calibration.cell(2000, "blitz") == calibration.cell(2199, "blitz") == mid
    assert calibration.cell(500, "blitz") == low and calibration.cell(3100, "blitz") == high
    assert calibration.cell(2200, "blitz") is calibration.cell(2000, "bullet") is None


def test_the_clock_caps_the_budget():
    """The largest rung up to the cell's budget that fits a tenth of the clock above the reserve
    (review: a 0.2 s clock must not search); a search may run to a fifth before it stops."""
    for kind, budget in (("coverage", 256), ("coverage", 128), ("lookahead", 8)):
        for clock in (None, 0.2, 2.0, 5.0, 10.0, 30.0, 100.0, 1000.0):
            n = calibration.affordable(kind, budget, clock, reserve=1.0)
            tenth = calibration.limit(clock, 1.0) / 2
            assert n in (0, *calibration.LADDER[kind]) and n <= budget and n * calibration.COST[kind] <= tenth
            assert all(b * calibration.COST[kind] > tenth for b in calibration.LADDER[kind] if n < b <= budget)
    assert calibration.affordable("coverage", 256, 0.2) == 0 and calibration.affordable("coverage", 256, None) == 256
    assert calibration.limit(51.0, 1.0) == 10.0


def test_tilt():
    prior, q = np.array([0.5, 0.3, 0.2]), np.array([-1.0, 0.0, 1.0])
    assert calibration.tilt(prior, q, 0.0) == pytest.approx(prior)
    p = calibration.tilt(prior, q, 2.0)
    assert p.sum() == pytest.approx(1) and p[2] / p[0] == pytest.approx(0.2 / 0.5 * np.exp(4))


class Search:
    """Records budgets and deadlines; gives the legal moves, coverage's calibrated distribution with
    the first move certain, and a flat prior with the last move valued 1 and the others -1."""

    def __init__(self, kind):
        self.kind, self.budgets, self.deadlines, self.late = kind, [], [], False

    def __call__(self, game, n, deadline=np.inf):
        self.budgets.append(n)
        self.deadlines.append(deadline)
        if self.late:
            return None
        moves = [m.uci() for m in game.board.legal_moves]
        last, prior = np.arange(len(moves)) == len(moves) - 1, np.full(len(moves), 1 / len(moves))
        if self.kind == "coverage":
            return moves, (np.arange(len(moves)) == 0).astype(float), prior, np.where(last, 1.0, -1.0)
        return moves, prior, np.where(last, 1.0, -1.0)


def test_calibrated_mode(tiny_path, tiny, monkeypatch):
    legal_of = lambda g: [m.uci() for m in g.board.legal_moves]
    play = Play(mode="calibrated", think_time=False, resign=False, draws=False)
    search = dict(coverage=Search("coverage"), lookahead=Search("lookahead"))
    monkeypatch.setattr(calibration, "CELLS", {("rapid", 2000): ("coverage", 256, None)})
    game = Game(Engine(tiny), 2000, 2000, 600, 5, "rapid", seed=0)  # FP32: the reference backend
    game.update(random_game(3, 20), 590, 585)
    assert game.decide(play, search, 590).move in legal_of(game)
    assert search["coverage"].budgets == []  # no search off the fast backend
    from allie.lichess import fast

    try:
        fast.library()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no fast kernels: {e}")
    engine = Engine(Model(tiny_path, dtype=torch.bfloat16, backend="fast", threads=2))
    game = Game(engine, 2000, 2000, 600, 5, "rapid", seed=0)
    game.update(random_game(3, 20), 590, 585)
    assert game.decide(play, None, 590).move in legal_of(game)  # no searcher: the policy
    assert game.decide(play, search, 590).move == legal_of(game)[0]  # coverage's calibrated distribution
    monkeypatch.setattr(calibration, "CELLS", {("rapid", 2000): ("coverage", 256, 40.0)})
    assert game.decide(play, search, 590).move == legal_of(game)[-1]  # coverage's values: prior exp(40 Q)
    assert search["coverage"].budgets == [256, 256]
    monkeypatch.setattr(calibration, "CELLS", {("rapid", 2000): ("lookahead", 8, 40.0)})
    assert game.decide(play, search, 590).move == legal_of(game)[-1]  # prior exp(40 Q)
    assert search["lookahead"].budgets == [8]
    monkeypatch.setattr(calibration, "CELLS", {})
    game.decide(play, search, 590)
    assert search["coverage"].budgets == [256, 256] and search["lookahead"].budgets == [8]  # no cell: no search
    monkeypatch.setattr(calibration, "CELLS", {("rapid", 2000): ("coverage", 256, None)})
    t = time.monotonic()
    reserve = behaviour.PARAMETERS["guard"]["reserve"]
    search["coverage"].late = True  # past its deadline: the move comes from the policy
    d = game.decide(play, search, 590)
    assert search["coverage"].budgets == [256] * 3
    assert search["coverage"].deadlines[-1] - t == pytest.approx((590 - reserve) / 5, abs=0.5)
    legal, p, _, _ = game.position()
    assert d.probability == pytest.approx(p[legal.index(d.move)])


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
