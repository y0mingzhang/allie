import numpy as np
import pytest

from allie.lichess import behaviour

if not behaviour.PARAMETERS:
    pytest.skip("behaviour.json not fitted", allow_module_level=True)


def time_probs(bin_):
    p = np.zeros(63)
    p[bin_] = 1
    return p


def test_think_draws_from_the_head():
    rng = np.random.default_rng(0)
    t = [behaviour.think(time_probs(5), rng, 600, 0, 30) for _ in range(200)]
    assert 4.5 <= min(t) and max(t) <= 5.5  # bin 5 is 5 s
    t = behaviour.think(time_probs(40), rng, 3600, 0, 30)  # bin 40: about 16 * e^(24 / 7.06) s
    assert 400 < t < 600
    assert all(0.5 <= behaviour.think(time_probs(62), rng, 60, 0, k) <= 2 for k in (0, 1))


def test_think_never_spends_the_reserve():
    rng = np.random.default_rng(0)
    reserve = behaviour.PARAMETERS["guard"]["reserve"]
    for clock in (0.0, 0.5, 1, 1.5, 3, 10, 60, 600):
        for inc in (0, 2, 30, 180):  # the increment arrives only after the move
            for b in (0, 20, 62):
                t = behaviour.think(time_probs(b), rng, clock, inc, 40)
                assert 0 <= t <= max(0.0, clock - reserve) + 1e-9, (clock, inc, b, t)


def test_think_spends_less_of_a_short_clock():
    """Under a minute one move takes at most a quarter of the clock above the reserve (live, a bullet move
    took 13 s of 30), half of it from two minutes."""
    rng, g = np.random.default_rng(0), behaviour.PARAMETERS["guard"]
    assert g["reserve"] >= 5
    long = time_probs(40)  # minutes
    for clock, share in ((30, 0.25), (60, 0.25), (90, 0.375), (120, 0.5), (600, 0.5)):
        t = behaviour.think(long, rng, clock, 0, 40)
        assert abs(t - share * g["share"] / 0.5 * (clock - g["reserve"])) < 1e-9, (clock, t)
    assert behaviour.think(long, rng, 30, 2, 40) == pytest.approx(0.25 * 25 + 2)


def test_resign_only_when_lost():
    rng = np.random.default_rng(0)
    floor = behaviour.PARAMETERS["resign"]["floor"]
    lost, even = (0.001, 0.009, 0.99), (0.4, 0.2, 0.4)
    assert not any(behaviour.resign(even, 60, 1500, 1, 100, 180, rng) for _ in range(1000))
    assert even[2] < floor <= lost[2]
    hits = sum(behaviour.resign(lost, 60, 1500, 1, 100, 180, rng) for _ in range(1000))
    assert hits > 0
    knots = behaviour.PARAMETERS["resign"]["knots"]
    x = behaviour.resign_features(lost, 60, 1500, 1, 100, 180, True, knots)
    assert abs(hits / 1000 - behaviour.hazard("resign", x)) < 0.05
    assert not any(behaviour.resign(lost, 1, 1500, 1, 100, 180, rng) for _ in range(100))
    floor = behaviour.PARAMETERS["resign"]["min_ply"]
    assert floor == 20  # live py0J32VQ resigned at move 3
    assert not any(behaviour.resign(lost, floor - 1, 1500, 1, 100, 180, rng, on_turn) for on_turn in (True, False)
                   for _ in range(500))  # fmt: skip
    assert any(behaviour.resign(lost, floor, 1500, 1, 100, 180, rng) for _ in range(500))
    off = sum(behaviour.resign(lost, 60, 1500, 1, 100, 180, rng, on_turn=False) for _ in range(1000))
    xo = behaviour.resign_features(lost, 60, 1500, 1, 100, 180, False, knots)
    assert abs(off / 1000 - behaviour.hazard("resign", xo)) < 0.05


def test_draws():
    assert behaviour.accept_draw((0.3, 0.4, 0.3), 40)  # even
    assert not behaviour.accept_draw((0.9, 0.08, 0.02), 40)  # clearly better
    assert not behaviour.accept_draw((0.3, 0.4, 0.3), 29)  # too early (live DYe6KEiU: agreed at move 3)
    assert behaviour.accept_draw((0.3, 0.4, 0.3), 30)
    rng = np.random.default_rng(0)
    assert not behaviour.offer_draw((0.3, 0.4, 0.3), 5, 1500, 1, 100, 180, rng)  # too early
