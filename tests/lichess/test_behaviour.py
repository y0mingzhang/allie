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
    t = behaviour.think(
        time_probs(40), rng, 3600, 0, 30
    )  # bin 40: about 16 * e^(24 / 7.06) s
    assert 400 < t < 600
    assert all(
        0.5 <= behaviour.think(time_probs(62), rng, 60, 0, k) <= 2 for k in (0, 1)
    )


def test_think_never_spends_the_reserve():
    rng = np.random.default_rng(0)
    g = behaviour.PARAMETERS["guard"]
    for clock in (0.5, 1, 3, 10, 60):
        for inc in (0, 2):
            t = behaviour.think(time_probs(62), rng, clock, inc, 40)
            assert (
                t
                <= max(0, clock - g["reserve"]) * g["share"]
                + inc * g["increment"]
                + 1e-9
            )


def test_resign_only_when_lost():
    rng = np.random.default_rng(0)
    floor = behaviour.PARAMETERS["resign"]["floor"]
    lost, even = (0.001, 0.009, 0.99), (0.4, 0.2, 0.4)
    assert not any(
        behaviour.resign(even, 60, 1500, 1, 100, 180, rng) for _ in range(1000)
    )
    assert even[2] < floor <= lost[2]
    hits = sum(behaviour.resign(lost, 60, 1500, 1, 100, 180, rng) for _ in range(1000))
    assert hits > 0
    x = behaviour.features(lost, 60, 1500, 1, 100, 180)
    assert abs(hits / 1000 - behaviour.hazard("resign", x)) < 0.05


def test_draws():
    assert behaviour.accept_draw((0.3, 0.4, 0.3))  # even
    assert not behaviour.accept_draw((0.9, 0.08, 0.02))  # clearly better
    rng = np.random.default_rng(0)
    assert not behaviour.offer_draw(
        (0.3, 0.4, 0.3), 5, 1500, 1, 100, 180, rng
    )  # too early
