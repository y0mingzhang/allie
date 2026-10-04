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
    for bot, mock, run in STARTED:
        bot.stop()
        bot.join()
        mock.close()
        run.join()
    STARTED.clear()
    e.close()


def start(engine, mock=None, max_games=4, **play):
    mock = mock or MockLichess({"tok": "allie"})
    play = dict(think_time=False, resign=False, draws=False) | play
    config = Config(max_games=max_games, play=Play(**play))
    bot = Bot(config, Lichess("tok", mock.url, wait=0.1, silence=3), engine)  # mock keepalives: 1 s
    run = threading.Thread(target=bot.run, daemon=True)
    run.start()
    STARTED.append((bot, mock, run))
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


def test_draw_offer_and_resign(engine, monkeypatch):
    from allie.lichess import behaviour

    monkeypatch.setattr(behaviour, "accept_draw", lambda wdl: True)
    mock = MockLichess({"tok": "allie"})
    mock.house_offers = {6}
    mock, _ = start(engine, mock, draws=True)
    mock.challenge("random", "allie", 60, 1, color="white")
    wait(lambda: finished(mock))
    g = next(iter(mock.games.values()))
    assert g.status == "draw"
    def on_turn_from_ply_2(wdl, ply, *args, on_turn=True):
        return on_turn and ply >= 2

    monkeypatch.setattr(behaviour, "resign", on_turn_from_ply_2)
    mock2, _ = start(engine, resign=True)
    mock2.challenge("random", "allie", 60, 1, color="white")
    wait(lambda: finished(mock2))
    g2 = next(iter(mock2.games.values()))
    # the bot (black) resigns instead of its second move
    assert g2.status == "resign" and g2.winner == "white" and len(g2.board.move_stack) == 3
    monkeypatch.setattr(behaviour, "resign", lambda wdl, ply, *args, on_turn=True: not on_turn)
    mock3, _ = start(engine, resign=True)
    mock3.challenge("random", "allie", 60, 1, color="white")
    wait(lambda: finished(mock3))
    g3 = next(iter(mock3.games.values()))
    # ... or right after its first move
    assert g3.status == "resign" and g3.winner == "white" and len(g3.board.move_stack) == 2


def test_rejects_bad_token():
    mock = MockLichess({"tok": "allie"})
    with pytest.raises(urllib.error.HTTPError):
        Lichess("wrong", mock.url).account()
    mock.close()


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


def test_restart_resumes_games(engine):
    mock, first = start(engine)
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: len(mock.games) == 1)
    g = next(iter(mock.games.values()))
    wait(lambda: len(g.board.move_stack) >= 10)
    first.stop()  # the process dies mid-game (preemption)
    first.join()
    _, second = start(engine, mock)
    wait(lambda: g.id in second.finished)
    assert not mock.rejected and g.status != "outoftime" and len(g.board.move_stack) > 12


def test_agreed_draw_frees_the_slot(engine, monkeypatch):
    """The bot offers a draw with its move, the opponent accepts: the game ends, its thread
    exits and the next challenge is accepted and played (live game y9S42Hkn ended this way)."""
    from allie.lichess import behaviour

    monkeypatch.setattr(behaviour, "offer_draw", lambda wdl, ply, *args: ply >= 6)
    mock = MockLichess({"tok": "allie"})
    mock.house_accepts = True
    mock, bot = start(engine, mock, max_games=1, draws=True)
    mock.challenge("random", "allie", 60, 1, color="white")
    wait(lambda: finished(mock))
    g = next(iter(mock.games.values()))
    assert g.status == "draw" and 6 <= len(g.board.move_stack) <= 8
    wait(lambda: g.id in bot.finished and not bot.games)
    mock.challenge("someone", "allie", 60, 1, color="white")
    wait(lambda: finished(mock, 2))
    assert not mock.declined and len(mock.games) == 2


@pytest.mark.parametrize("kind", ["event", "game"])
def test_stalled_stream_reconnects(engine, kind):
    """A stream that goes silent (no keepalives) is dropped after Lichess.silence seconds and
    reopened; challenges and moves then flow again."""
    mock = MockLichess({"tok": "allie"})
    mock.stall[kind] = 30
    mock, bot = start(engine, mock)
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: finished(mock), timeout=30)
    paths = [c[2] for c in mock.calls]
    stream = "/api/stream/event" if kind == "event" else "/api/bot/game/stream/"
    assert sum(stream in p for p in paths) >= 2 and not mock.rejected
    assert next(iter(mock.games.values())).status != "outoftime"


def test_game_survives_unexpected_errors(engine, monkeypatch, caplog):
    """An unexpected error in the game thread is logged and the game resumes from a fresh
    stream; a persistent one resigns the game rather than leaving it to time out."""
    from allie.lichess import bot as botmod

    real, failures = botmod.Match.on_state, [1]

    def flaky(self, s):
        if failures[0] and len(s["moves"].split()) >= 4:
            failures[0] -= 1
            raise RuntimeError("injected")
        return real(self, s)

    monkeypatch.setattr(botmod.Match, "on_state", flaky)
    monkeypatch.setattr(botmod, "ERRORS", 3)
    mock, bot = start(engine)
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: finished(mock))
    g = next(iter(mock.games.values()))
    assert g.status in ("mate", "draw", "stalemate") and len(g.board.move_stack) > 10
    failures[0] = 10**9  # from now on every state fails: the bot resigns its game
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: finished(mock, 2))
    g2 = list(mock.games.values())[1]
    assert g2.status == "resign" and g2.winner == "black"  # the challenger; the bot was white
    assert sum("error 1 of 3" in r.message for r in caplog.records) == 2
    assert any("giving up" in r.message for r in caplog.records)


def test_plays_through_a_full_disk(engine, tmp_path, monkeypatch):
    """The shared filesystem fills up mid-game (live 2026-10-04: EDQUOT from 08:17 to 12:41
    ET): every write there fails, the bot plays on, the log lines go to the node-local file,
    and the log notes the gap once the disk takes writes again."""
    import builtins
    import errno
    import logging
    import sys

    from allie.lichess import logs

    shared, local = tmp_path / "shared", tmp_path / "local.log"
    shared.mkdir()
    path = shared / "bot.log"
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    listener = logs.setup(str(path))
    listener.handlers[0].fallback = str(local)
    real, full = builtins.open, [False]

    def quota(file, mode="r", *args, **kwargs):
        if full[0] and str(file).startswith(str(shared)) and any(c in mode for c in "wa+"):
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", quota)
    try:
        mock, bot = start(engine)
        mock.challenge("random", "allie", 60, 1, color="black")
        mock.challenge("other", "allie", 60, 1, color="white")
        wait(lambda: len(mock.games) == 2)

        class Stale:  # the open handle fails too (swapped in, not closed under the writer)
            def write(self, text):
                raise OSError(errno.EDQUOT, "Disk quota exceeded")

            flush = close = lambda self: None

        full[0] = True
        listener.handlers[0].stream = Stale()
        wait(lambda: finished(mock, 2))
        assert not mock.rejected and all(g.status != "outoftime" for g in mock.games.values())
        wait(lambda: len(bot.finished) == 2 and not any(t.is_alive() for t in bot.threads))
        full[0] = False
        logging.getLogger("allie").info("disk back")
    finally:
        listener.stop()
        root.handlers[:] = handlers
        root.setLevel(level)
    text = path.read_text()
    assert "lines could not be written here" in text and text.rstrip().endswith("disk back")
    assert " over: " in local.read_text()  # the games' ends, logged locally meanwhile
