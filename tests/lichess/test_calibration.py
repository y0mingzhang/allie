import time

import numpy as np
import pytest
import torch

from allie.lichess import behaviour, calibration
from allie.lichess.engine import Engine, Game, Play
from allie.lichess.model import Model

from .test_tokens import random_game


def test_beta_grows_with_rating_and_think_time():
    quick, slow = np.zeros(63), np.zeros(63)
    quick[2], slow[30] = 1, 1  # 2.5 s; about 2 min
    b = calibration.beta
    assert b(1700, quick) < b(1700, slow) and b(1200, slow) < b(1700, slow) < b(2400, slow)
    t = float(calibration.SECONDS[30])
    assert b(2700, slow) == pytest.approx(calibration.BETA0 * np.exp(calibration.GAMMA) * (t / 10) ** (calibration.DELTA + calibration.ETA))


def test_the_clock_caps_the_budget():
    """The largest rung that fits a tenth of the clock above the reserve (review: a 0.2 s clock must
    not search) and the think time; a search may run to a fifth before it stops."""
    rungs, sim = calibration.LADDER, calibration.COST
    for clock in (None, 0.2, 2.0, 5.0, 10.0, 30.0, 100.0, 154.0, 156.0, 258.0, 260.0, 1000.0):
        n = calibration.affordable(clock, reserve=1.0)
        tenth = calibration.limit(clock, 1.0) / 2
        assert n in (0, *rungs) and n * sim <= tenth and all(b * sim > tenth for b in rungs if b > n)
    assert calibration.affordable(0.2) == 0 and calibration.affordable(None) == max(rungs)
    at = lambda seconds: calibration.affordable(10 * seconds + 1.0)  # a tenth above the reserve
    top, below = rungs[-1], rungs[-2]
    assert at(top * sim) == top and at(top * sim - 0.01) == below and at(8 * sim - 0.001) == 0  # then the policy
    assert calibration.affordable(1e4, think=top * sim - 0.01) == below  # the think time caps it too
    assert calibration.affordable(1e4, think=8 * sim - 0.001) == 0
    assert calibration.limit(51.0, 1.0) == 10.0


def test_tilt():
    prior = np.array([0.5, 0.3, 0.2])
    assert calibration.tilt(prior, np.array([-1.0, 0.0, 1.0]), 0.0) == pytest.approx(prior)
    p = calibration.tilt(prior, np.array([-0.5, 0.0, 0.5]), 2.0)  # log odds, tilted as given
    assert p.sum() == pytest.approx(1) and p[2] / p[0] == pytest.approx(0.2 / 0.5 * np.exp(2.0))
    assert np.isfinite(calibration.tilt(prior, np.array([-5.0, 0.0, 5.0]), 64.0)).all()  # clear wins stay finite


class Search:
    """Records budgets and deadlines; gives the legal moves in reverse board order (the mode maps
    them by name), a calibrated distribution it must not use, a flat prior, and the board's last move
    valued 1, the others -1."""

    def __init__(self):
        self.budgets, self.deadlines, self.late, self.seconds = [], [], False, 0.0

    def __call__(self, game, n, deadline=np.inf):
        self.budgets.append(n)
        self.deadlines.append(deadline)
        time.sleep(self.seconds)
        if self.late:
            return None
        moves = [m.uci() for m in game.board.legal_moves][::-1]
        first = np.arange(len(moves)) == len(moves) - 1  # the board's first move
        return moves, first.astype(float), np.full(len(moves), 1 / len(moves)), np.where(np.arange(len(moves)) == 0, 1.0, -1.0)


def test_calibrated_mode(tiny_path, tiny, monkeypatch):
    legal_of = lambda g: [m.uci() for m in g.board.legal_moves]
    play = Play(mode="calibrated", think_time=False, resign=False, draws=False)
    search = Search()
    game = Game(Engine(tiny), 2000, 2000, 600, 5, "rapid", seed=0)  # FP32: the reference backend
    game.update(random_game(3, 20), 590, 585)
    assert game.decide(play, search, 590).move in legal_of(game)
    assert search.budgets == []  # no search off the fast backend
    pytest.importorskip("allie_fast")
    engine = Engine(Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2))
    game = Game(engine, 2000, 1500, 600, 5, "rapid", seed=0)
    game.update(random_game(3, 20), 590, 585)
    assert len(game.moves) == 20  # white to move
    assert game.decide(play, None, 590).move in legal_of(game)  # no searcher: the policy
    betas, beta = [], calibration.beta
    monkeypatch.setattr(calibration, "beta", lambda r, t: betas.append(r) or 40.0)
    assert game.decide(play, search, 590).move == legal_of(game)[-1]  # the values' tilt, not the calibrated distribution
    assert search.budgets == [calibration.LADDER[-1]] and betas == [2000]  # the clock allows the top rung; white's rating
    reserve, rungs = behaviour.PARAMETERS["guard"]["reserve"], calibration.LADDER
    clock = 10 * rungs[-2] * calibration.COST + (reserve + 1) / 2  # a tenth above the reserve just misses rungs[-2]
    assert game.decide(play, search, clock).move == legal_of(game)[-1]
    assert search.budgets[-1] == calibration.affordable(clock, reserve) == rungs[-3]
    monkeypatch.setattr(calibration, "beta", beta)
    t = time.monotonic()
    search.late = True  # past its deadline: the move comes from the policy
    d = game.decide(play, search, 590)
    assert d.sims == search.budgets[-1] and d.stopped
    assert search.deadlines[-1] - t == pytest.approx((590 - reserve) / 5, abs=0.5)
    legal, p, _, _ = game.position()
    assert d.probability == pytest.approx(p[legal.index(d.move)])


def test_the_think_time_caps_the_search(tiny_path, monkeypatch):
    """The think time is drawn before the search, caps its rung and stops it MARGIN before; the
    Decision waits that same think time."""
    pytest.importorskip("allie_fast")
    play = Play(mode="calibrated", think_time=True, resign=False, draws=False)
    search = Search()
    game = Game(Engine(Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)), 2000, 2000, 600, 5, "rapid", seed=0)
    game.update(random_game(3, 20), 590, 585)
    sim, margin = calibration.COST, calibration.MARGIN
    for think, rung in ((256 * sim + margin + 0.05, 256), (256 * sim + margin - 0.01, 128), (8 * sim + margin - 0.001, None)):
        draws = iter([think, 1e3])  # a second draw would wait 1,000 s
        monkeypatch.setattr(game, "think", lambda play, clock, time, d=draws: next(d))
        n, t = len(search.budgets), time.monotonic()
        d = game.decide(play, search, 590)
        assert d.think == think  # drawn once, before the search
        assert search.budgets[n:] == ([] if rung is None else [rung])  # no rung fits: the policy
        assert (d.sims, d.stopped) == (rung or 0, False)
        if rung:  # it stops MARGIN before the think time from the decision's start
            assert t <= search.deadlines[-1] - (think - margin) <= time.monotonic()
    game.engine.sim_cost = 2 * sim  # recent searches measured twice COST (a busy machine)
    monkeypatch.setattr(game, "think", lambda play, clock, time: 256 * sim + margin + 0.05)
    search.seconds = 64 * sim  # a quarter of COST a simulation
    d = game.decide(play, search, 590)
    assert (d.sims, d.stopped) == (128, False) and d.searched >= search.seconds
    assert search.budgets[-1] == 128 and game.engine.sim_cost == pytest.approx(0.7 * 2 * sim + 0.3 * sim / 2, rel=0.1)
    game.engine.sim_cost = 1e3  # a stalled search: the price stays at 4 COST, small rungs still search
    monkeypatch.setattr(game, "think", lambda play, clock, time: 32 * 4 * sim + margin + 0.05)
    search.seconds = 0.0
    game.decide(play, search, 590)
    assert search.budgets[-1] == 32


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

