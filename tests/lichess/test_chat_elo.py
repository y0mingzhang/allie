import importlib.util
import pathlib

import numpy as np

path = pathlib.Path(__file__).parents[2] / "analysis/lichess/chat_elo.py"
spec = importlib.util.spec_from_file_location("chat_elo", path)
elo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elo)


def test_bt_recovers_elo():
    true = np.array([0.0, 100.0, -200.0])
    p = 1 / (1 + 10 ** ((true[None, :] - true[:, None]) / 400))  # p[i, j]: i beats j
    W = 2000 * p
    np.fill_diagonal(W, 0)
    assert np.abs(elo.bt(W) - true).max() < 2


def test_fit_resamples_games_not_pairs():
    # A wins every pair in 5 games and loses every pair in 5 others: the point is 0 and the interval must be wide,
    # which it would not be with pairs resampled independently
    recs = [dict(x="b", y="a", gid=f"g{g}", better="x" if g < 5 else "y") for g in range(10) for _ in range(50)]
    point, lo, hi, games, _ = elo.fit_pairs(["b", "a"], recs, B=500)
    assert games == 10 and abs(point[1]) < 1 and hi[1] - lo[1] > 200 and lo[1] < 0 < hi[1]


def test_ties_count_half():
    recs = [dict(x="b", y="a", gid=f"g{g}", better="tie") for g in range(50)]
    assert abs(elo.fit_pairs(["b", "a"], recs, B=50)[0][1]) < 1
