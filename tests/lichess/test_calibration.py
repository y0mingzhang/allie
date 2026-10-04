import numpy as np
import pytest

from allie.lichess import calibration
from allie.lichess.engine import Engine, Game, Play

from .test_tokens import random_game


def test_rule():
    assert calibration.temperature(1700) == pytest.approx(np.exp(calibration.TAU0))
    assert calibration.temperature(2600) < calibration.temperature(800)
    quick, long_think = np.zeros(63), np.zeros(63)
    quick[0], long_think[-1] = 1, 1  # 0.5 s; over an hour
    assert calibration.budget(800, quick) < calibration.budget(2600, quick) <= calibration.HW * 0.5
    assert calibration.budget(2600, long_think) == calibration.CAP
    assert calibration.budget(2600, long_think, clock=5) == pytest.approx(calibration.HW * 5 / 10)
    assert np.allclose(calibration.sharpen([0.2, 0.3, 0.5], 1), [0.2, 0.3, 0.5])
    assert calibration.sharpen([0.2, 0.3, 0.5], 0.5).argmax() == 2


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


def test_calibrated_mode(tiny, monkeypatch):
    game = Game(Engine(tiny), 2000, 2000, 600, 5, "rapid", seed=0)
    game.update(random_game(3, 20), 590, 585)
    legal = [m.uci() for m in game.board.legal_moves]
    play = Play(mode="calibrated", think_time=False, resign=False, draws=False)
    assert game.decide(play, None, 590).move in legal  # no search: the policy
    search = Search()
    monkeypatch.setattr(
        calibration, "K", 1e6
    )  # always the cap: the searched distribution
    assert game.decide(play, search, 590).move == legal[-1]
    assert search.budgets == [calibration.CAP]
    monkeypatch.setattr(calibration, "K", 0.0)
    game.decide(play, search, 590)
    assert search.budgets == [calibration.CAP]  # budget 0: no search
