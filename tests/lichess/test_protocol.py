import threading
import time
import urllib.error

import chess
import pytest

from allie.lichess.bot import Bot, screen
from allie.lichess.client import Lichess
from allie.lichess.config import Challenge, Config
from allie.lichess.engine import Engine, Play
from allie.lichess.mock import MockLichess

STARTED = []


@pytest.fixture
def engine(tiny):
    e = Engine(tiny)
    yield e
    for bot, mock in STARTED:
        bot.stop()
        bot.join()
        mock.close()
    STARTED.clear()
    e.close()


def start(engine, mock=None, **play):
    mock = mock or MockLichess({"tok": "allie"})
    config = Config(max_games=4, play=Play(think_time=False, **play))
    bot = Bot(config, Lichess("tok", mock.url, wait=0.1), engine)
    threading.Thread(target=bot.run, daemon=True).start()
    STARTED.append((bot, mock))
    wait(lambda: bot.me is not None)
    return mock, bot


def wait(cond, timeout=60):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


def finished(mock, n=1):
    return len(mock.games) >= n and all(
        g.status != "started" for g in mock.games.values()
    )


def challenge(**kw):
    c = dict(variant=dict(key="standard"), speed="blitz", rated=True, challenger=dict(id="x"),
             timeControl=dict(type="clock", limit=180, increment=2))  # fmt: skip
    for k, v in kw.items():
        c[k] = v
    return c


def test_screen():
    rules = Challenge(min_base=60, max_base=3600, speeds=("blitz", "rapid"))
    assert screen(challenge(), rules, False) is None
    assert screen(challenge(variant=dict(key="chess960")), rules, False) == "standard"
    assert screen(challenge(speed="bullet"), rules, False) == "timeControl"
    assert (
        screen(challenge(timeControl=dict(type="unlimited")), rules, False)
        == "timeControl"
    )
    tc = lambda base, inc: dict(type="clock", limit=base, increment=inc)
    assert screen(challenge(timeControl=tc(30, 0)), rules, False) == "tooFast"
    assert screen(challenge(timeControl=tc(5400, 0)), rules, False) == "tooSlow"
    bot = dict(id="b", title="BOT")
    assert screen(challenge(challenger=bot), rules, False) == "noBot"
    assert screen(challenge(), Challenge(rated=False), False) == "casual"
    assert screen(challenge(), rules, True) == "later"


def test_plays_a_game_to_the_end(engine):
    mock, _ = start(engine)
    mock.challenge("random", "allie", 60, 1, color="white")
    mock.challenge("someone", "allie", 60, 0, variant="chess960")
    wait(lambda: finished(mock))
    g = next(iter(mock.games.values()))
    assert g.status in ("mate", "draw", "stalemate", "resign") and not mock.rejected
    assert len(g.board.move_stack) > 10
    assert mock.declined[0][1] == "standard"
    chess.Board().push_uci(g.board.move_stack[0].uci())


def test_reconnects_and_retries(engine):
    mock = MockLichess({"tok": "allie"})
    mock.drop_after = 3  # the first game stream closes after three states
    mock.fail["/api/bot/game/stream"] = [503]
    mock.fail["/api/bot/game/g"] = [429, 503]  # the first move: rate limited, server error
    mock, _ = start(engine, mock)
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: finished(mock))
    streams = [c for c in mock.calls if "/game/stream/" in c[2]]
    assert len(streams) >= 2 and not mock.rejected and not any(mock.fail.values())
    assert next(iter(mock.games.values())).status != "outoftime"


def test_clock_features(engine):
    mock, bot = start(engine)
    mock.challenge("random", "allie", 60, 1, color="white")
    wait(lambda: len(mock.games) == 1)
    g = next(iter(mock.games.values()))
    wait(lambda: len(g.board.move_stack) >= 16)
    game = bot.games[g.id].game
    known = [(c, h // 1000) for c, h in zip(game.clocks, g.history) if c is not None]
    assert len(known) >= 0.8 * len(game.clocks) and all(c == h for c, h in known)


def test_draw_offer_and_resign(engine):
    mock = MockLichess({"tok": "allie"})
    mock.house_offers = {6}
    mock, _ = start(engine, mock, draw_accept=1.0)
    mock.challenge("random", "allie", 60, 1, color="white")
    wait(lambda: finished(mock))
    g = next(iter(mock.games.values()))
    assert g.status == "draw"
    mock2, _ = start(engine, resign_loss=0.0, resign_moves=1)
    mock2.challenge("random", "allie", 60, 1, color="white")
    wait(lambda: finished(mock2))
    g2 = next(iter(mock2.games.values()))
    assert (
        g2.status == "resign"
        and g2.winner == "white"  # the challenger, who chose white
        and len(g2.board.move_stack) >= 20
    )


def test_rejects_bad_token():
    mock = MockLichess({"tok": "allie"})
    with pytest.raises(urllib.error.HTTPError):
        Lichess("wrong", mock.url).account()


def test_drain(engine):
    mock, bot = start(engine)
    mock.challenge("random", "allie", 60, 1)
    wait(lambda: len(bot.games) == 1)
    bot.drain()
    mock.challenge("other", "allie", 60, 1)
    wait(lambda: bot.stopped.is_set(), timeout=120)
    assert finished(mock) and mock.declined[-1][1] == "later"


def test_quota_counts_accepted_challenges(engine):
    mock = MockLichess({"tok": "allie"})
    mock.challenge("someone", "allie", 60, 1)
    mock.challenge("someone", "allie", 60, 1)  # queued before the first game's gameStart
    mock, _ = start(engine, mock)
    wait(lambda: len(mock.declined) == 1)
    assert len(mock.games) == 1 and mock.declined[0][1] == "later"


def test_engine_runs_calls_alone(tiny):
    e = Engine(tiny)
    assert e.run(lambda: 42) == 42
    e.close()
