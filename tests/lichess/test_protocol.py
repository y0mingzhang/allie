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


def start(engine, mock=None, max_games=4, abort=30, searcher=None, **play):
    mock = mock or MockLichess({"tok": "allie"})
    play = dict(think_time=False, resign=False, draws=False) | play
    config = Config(max_games=max_games, abort=abort, play=Play(**play))
    bot = Bot(config, Lichess("tok", mock.url, wait=0.1, silence=3), engine, searcher)  # mock keepalives: 1 s
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
    return len(mock.games) >= n and all(g.status != "started" for g in mock.games.values())


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
    assert screen(challenge(timeControl=dict(type="unlimited")), rules, False) == "timeControl"
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

    monkeypatch.setattr(behaviour, "accept_draw", lambda wdl, ply: True)
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
    """Unexpected errors in the game thread are logged and the game resumes from a fresh
    stream. Scattered ones never add up (the count restarts once the bot moves); ERRORS in a
    row resign the game rather than leave it to time out."""
    from allie.lichess import bot as botmod

    real, scattered, every = botmod.Match.on_state, {2, 4, 6, 8}, [False]

    def flaky(self, s):
        n = len(s["moves"].split())
        if every[0] or n in scattered:  # the bot's turns: it is white
            scattered.discard(n)
            raise RuntimeError("injected")
        return real(self, s)

    monkeypatch.setattr(botmod.Match, "on_state", flaky)
    monkeypatch.setattr(botmod, "ERRORS", 3)
    mock, bot = start(engine)
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: finished(mock))
    g = next(iter(mock.games.values()))
    assert g.status in ("mate", "draw", "stalemate") and len(g.board.move_stack) > 10
    assert not scattered and sum("error 1 of 3" in r.message for r in caplog.records) == 4
    wait(lambda: not bot.games)  # else the same challenger may meet per_user = 1 ("later")
    every[0] = True  # from now on every state fails: the bot resigns its game
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: finished(mock, 2))
    g2 = list(mock.games.values())[1]
    assert g2.status == "resign" and g2.winner == "black"  # the challenger; the bot was white
    assert sum("giving up" in r.message for r in caplog.records) == 1


def test_failed_decision_plays_the_policy(engine, monkeypatch, caplog):
    """A decision that fails (here its search) plays a move sampled from the raw policy; one whose policy
    fails too plays a random legal move."""
    from allie.lichess import engine as enginemod

    class Panic(BaseException):  # as pyo3's PanicException
        pass

    def broken(game, n, *_):
        raise Panic("injected")

    said = lambda text: sum(text in r.getMessage() for r in caplog.records)
    mock, _ = start(engine, searcher=broken, mode="strongest", search=4)
    mock.challenge("random", "allie", 60, 1, color="black")
    wait(lambda: said("playing the policy") >= 3)
    assert not said("playing a random move")
    monkeypatch.setattr(enginemod.Game, "position", lambda self: 1 / 0)
    n = said("playing a random move")
    wait(lambda: said("playing a random move") >= n + 2)


def test_errors_on_the_opponents_turn_do_not_resign(engine, monkeypatch, caplog):
    """A persistent error while the opponent thinks: the bot's clock is not running, so it
    waits instead of resigning, and plays on once the opponent moves."""
    from allie.lichess import bot as botmod

    real = botmod.Match.on_state

    def flaky(self, s):
        if len(s["moves"].split()) == 5:  # black to move: the opponent, as the bot is white
            raise RuntimeError("injected")
        return real(self, s)

    def slow(board):
        if len(board.move_stack) == 5:
            time.sleep(8)
        return next(iter(board.legal_moves)).uci()

    monkeypatch.setattr(botmod.Match, "on_state", flaky)
    monkeypatch.setattr(botmod, "ERRORS", 2)
    mock, _ = start(engine, MockLichess({"tok": "allie"}, house=slow))
    mock.challenge("slow", "allie", 60, 1, color="black")
    wait(lambda: len(mock.games) == 1)
    g = next(iter(mock.games.values()))
    wait(lambda: len(g.board.move_stack) >= 8, timeout=60)
    assert g.status == "started"
    assert sum("injected" in r.exc_text for r in caplog.records if r.exc_text) >= 3
    assert not any("giving up" in r.message for r in caplog.records)


def thinking(monkeypatch, think, search=0.0):
    """Every decision takes `search` seconds and draws `think` seconds; no resignation or draw offer."""
    from dataclasses import replace

    from allie.lichess import engine as enginemod

    decide = enginemod.Game.decide

    def slow(self, *args):
        d = decide(self, *args)
        time.sleep(search)
        return replace(d, think=think, resign=False, offer_draw=False)

    monkeypatch.setattr(enginemod.Game, "decide", slow)


def game_of(mock, n=1):
    wait(lambda: len(mock.games) >= n)
    return list(mock.games.values())[n - 1]


@pytest.mark.parametrize(("search", "think"), [(0.6, 1.2), (0.9, 0.3)])
def test_think_time_counts_from_the_state(engine, monkeypatch, search, think):
    """The think-time wait starts when the state arrives: a search inside the sampled think
    time does not add to it (a move takes max(think, search), the human's pace), and a search
    longer than the think time adds no wait. Timed at the server, from the opponent's move to the bot's."""
    thinking(monkeypatch, think, search)
    mock, _ = start(engine, think_time=True)
    mock.challenge("random", "allie", 600, 0, color="black")  # the bot is white
    g = game_of(mock)
    wait(lambda: len(g.times) >= 9)
    took = [g.times[j] - g.times[j - 1] for j in (2, 4, 6, 8)]
    want = max(search, think)
    assert all(want <= t < want + 0.25 for t in took), took


@pytest.mark.parametrize("accept", [True, False])
def test_offer_during_the_think_is_answered_before_the_move(engine, monkeypatch, accept):
    """An offer made while the bot thinks is answered at once, before its move: accepted, the game ends with
    no move after it; declined, the bot moves without an offer of its own. (Live, the bot moved first and its
    late yes became its own offer: 26 draws, some at move 4.)"""
    from allie.lichess import behaviour

    thinking(monkeypatch, 1.5)
    monkeypatch.setattr(behaviour, "accept_draw", lambda wdl, ply: accept)
    mock, _ = start(engine, draws=True, think_time=True)
    mock.challenge("random", "allie", 600, 0, color="black")  # the bot is white
    g = game_of(mock)
    wait(lambda: len(g.board.move_stack) == 4)  # the bot's turn
    mock.offer_draw(g, "black")
    t = time.monotonic()
    answer = f"/draw/{'yes' if accept else 'no'}"
    wait(lambda: any(answer in c[2] for c in mock.calls), timeout=10)
    assert time.monotonic() - t < 1.0 and not mock.rejected
    if accept:
        wait(lambda: g.status != "started")
        assert g.status == "draw" and len(g.board.move_stack) == 4
    else:
        wait(lambda: len(g.board.move_stack) >= 6)
        assert g.status == "started" and not any("/draw/yes" in c[2] or "offeringDraw" in c[2] for c in mock.calls)


def test_game_that_ends_during_the_think(engine, monkeypatch):
    """The opponent resigns while the bot thinks: no move is sent after it, the slot frees at once and a
    rematch is accepted (live: 166 moves posted to ended games, 55 rematches declined "later")."""
    thinking(monkeypatch, 5.0)
    mock, bot = start(engine, think_time=True)
    mock.challenge("random", "allie", 600, 0, color="black")  # the bot is white
    g = game_of(mock)
    wait(lambda: len(g.board.move_stack) == 2, timeout=15)
    g.end("resign", "white")
    t = time.monotonic()
    wait(lambda: not bot.games, timeout=5)
    assert time.monotonic() - t < 1.5
    mock.challenge("random", "allie", 600, 0, color="black")
    game_of(mock, 2)
    assert not mock.declined and not mock.rejected
    assert sum("/move/" in c[2] for c in mock.calls if g.id in c[2]) == 1


@pytest.mark.parametrize("restart", [True, False])
def test_lost_move_is_resent(engine, monkeypatch, caplog, restart):
    """Lichess acknowledges the bot's move and loses it (live RtI5RtnK: in a restart; the bot waited 349 s
    believing it had moved and lost on time). With the stream dropped, the reconnect's state shows the bot to
    move; with the stream up, no state comes within LATE seconds and the bot resyncs. Either way it sends the
    same move again, without deciding anew."""
    from allie.lichess import bot as botmod

    monkeypatch.setattr(botmod, "LATE", 0.5)
    mock = MockLichess({"tok": "allie"})
    mock.lose = [restart]
    mock, _ = start(engine, mock)
    mock.challenge("random", "allie", 60, 1, color="black")  # the bot is white: its first move is lost
    wait(lambda: finished(mock))
    g = game_of(mock)
    moves = [c[2] for c in mock.calls if "/move/" in c[2]]
    assert moves[0] == moves[1] and moves[0].endswith(g.board.move_stack[0].uci()) and not mock.lose
    assert g.status != "outoftime" and len(g.board.move_stack) > 10 and not mock.rejected
    assert any("resyncing" in r.getMessage() for r in caplog.records) != restart


class Calls:
    """A client that records each call and succeeds."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args)) or {}


def match(engine, **play):
    from types import SimpleNamespace

    from allie.lichess.bot import Match

    play = dict(think_time=False, resign=False) | play
    bot = SimpleNamespace(config=Config(play=Play(**play)), client=Calls(), me="allie", engine=engine, search=None)
    return Match(bot, "g1"), bot.client.calls


def full(moves="", **kw):
    side = lambda u: dict(id=u, name=u, rating=1500)
    return dict(type="gameFull", id="g1", variant=dict(key="standard"), speed="blitz", rated=True,
                white=side("allie"), black=side("opp"), clock=dict(initial=180000, increment=2000),
                initialFen="startpos", state=state(moves, **kw))  # fmt: skip


def state(moves, **kw):
    return dict(type="gameState", moves=moves, wtime=170000, btime=170000, winc=2000, binc=2000, status="started") | kw


def test_offers_are_answered_only_before_the_bots_move(engine, monkeypatch):
    """A state with the opponent's offer that arrives after the bot's move at that position is stale (Lichess
    declined the offer when the bot moved; a yes would offer a draw): it is not answered. A live offer is, and
    an accepted one ends the bot's turn."""
    from allie.lichess import behaviour

    m, calls = match(engine)
    m.on_event(full())
    m.act()
    (_, _, first, offer), = calls  # fmt: skip
    assert offer is False
    m.on_event(state("", bdraw=True))  # sent before the bot's move reached Lichess
    m.act()
    assert [c[0] for c in calls] == ["move"]
    m.on_event(state(first))
    m.act()
    m.on_event(state(f"{first} e7e5", bdraw=True))  # a live offer, declined (too early)
    m.act()
    assert calls[1] == ("draw", "g1", False) and calls[2][0] == "move" and calls[2][3] is False
    monkeypatch.setattr(behaviour, "accept_draw", lambda wdl, ply: True)
    m.on_event(state(f"{first} e7e5 {calls[2][2]} b8c6", bdraw=True))
    m.act()
    m.act()
    assert calls[3] == ("draw", "g1", True) and len(calls) == 4  # accepted: no move after it


def test_resend_after_reconnect_and_resync(engine, monkeypatch):
    """After the bot's move, a reconnect's state with the bot still to move (its move was lost) resends the
    same move without deciding again; a stream that shows no new state LATE seconds after the move resyncs."""
    from allie.lichess import bot as botmod
    from allie.lichess import engine as enginemod

    decide, n = enginemod.Game.decide, [0]

    def counted(self, *args):
        n[0] += 1
        return decide(self, *args)

    monkeypatch.setattr(enginemod.Game, "decide", counted)
    monkeypatch.setattr(botmod, "LATE", 0.2)
    m, calls = match(engine)
    m.on_event(full())
    m.act()
    m.tick()
    time.sleep(0.3)
    with pytest.raises(ConnectionError, match="resyncing"):
        m.tick()
    m.on_event(full())  # the reconnect: still the bot's move
    m.act()
    assert [c[:3] for c in calls] == [calls[0][:3]] * 2 and n[0] == 1
    m.on_event(full(calls[0][2]))  # it arrived: no resend
    m.act()
    assert len(calls) == 2


def test_patience_and_chat(engine):
    """No first move from the opponent: the wait scales with the base time (30 s up to 5 minutes, 90 s from
    15), and starts again whenever they chat, up to 3 times as long."""
    from allie.lichess.bot import patience

    assert [patience(30, b) for b in (60, 180, 300, 600, 900, 3600)] == [30, 30, 30, 60, 90, 90]
    assert patience(0, 600) == 0
    m, calls = match(engine)
    m.on_event(full("e2e4"))  # the bot is white and moved: the opponent owes a first move
    m.patience, now = 1.0, time.monotonic()
    m.waiting, m.chatted = now - 1.5, now - 0.5
    m.tick()
    assert not calls
    m.chatted = now - 1.2
    m.tick()
    assert calls == [("abort", "g1")]
    m.waiting, m.chatted = time.monotonic() - 3.5, time.monotonic()
    m.tick()
    assert calls[-1] == ("abort", "g1") and len(calls) == 2


def test_rating_offset(engine):
    """play.offset moves the bot's header Elo off its rating (mirrored or fixed), the opponent's stays."""
    assert Play(rating="opponent", offset=-50).ratings(1500) == (1450, 1500)
    assert Play(rating="opponent").ratings(None) == (1500, 1500)
    assert Play(rating=1800, offset=-50).ratings(None) == (1750, 1800)
    m, _ = match(engine, rating="opponent", offset=-50)
    m.on_event(full())
    assert m.game.elo == (1450, 1500)


def test_refused_move_resyncs(engine, caplog):
    """Lichess refuses a move and the game goes on (a desync, not the opponent resigning
    meanwhile): the bot resyncs from a fresh stream and moves again instead of flagging."""
    mock = MockLichess({"tok": "allie"})
    mock.fail["/api/bot/game/g"] = [400]  # the bot's first move
    mock, _ = start(engine, mock)
    mock.challenge("random", "allie", 60, 1, color="black")  # the bot is white
    wait(lambda: finished(mock))
    g = next(iter(mock.games.values()))
    assert g.status != "outoftime" and len(g.board.move_stack) > 10 and not mock.fail["/api/bot/game/g"]
    assert sum("resyncing" in str(r.exc_info[1]) for r in caplog.records if r.exc_info) == 1


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


def test_slow_opponent_after_both_first_moves_is_not_aborted(engine):
    def slow(board):
        if len(board.move_stack) >= 2:
            time.sleep(3)
        return next(iter(board.legal_moves)).uci()

    mock, _ = start(engine, MockLichess({"tok": "allie"}, house=slow), abort=1)
    mock.challenge("slow", "allie", 60, 1, color="white")  # the bot is black
    wait(lambda: len(mock.games) == 1)
    g = next(iter(mock.games.values()))
    wait(lambda: len(g.board.move_stack) >= 4, timeout=30)
    assert g.status == "started" and not any("/abort" in c[2] for c in mock.calls)


@pytest.mark.parametrize("color", ["white", "black"])
def test_aborts_a_game_the_opponent_never_starts(engine, color):
    """The opponent never makes a first move (live game fG7tNbjC sat an hour after 1. e4, and
    held the drain): the bot aborts after config.abort seconds and the drain completes."""
    mock = MockLichess({"tok": "allie"}, house_delay=600)
    mock, bot = start(engine, mock, abort=2)
    mock.challenge("idle", "allie", 60, 1, color=color)  # the challenger's color
    wait(lambda: len(bot.games) == 1)
    bot.drain()
    wait(lambda: bot.stopped.is_set(), timeout=30)
    g = next(iter(mock.games.values()))
    assert g.status == "aborted" and len(g.board.move_stack) == (color == "black")
