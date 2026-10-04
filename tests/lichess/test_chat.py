import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import chess
import pytest

from allie.lichess import chat, llm
from allie.lichess.bot import Bot
from allie.lichess.chat import Chat, Chatter, clean, played, split
from allie.lichess.client import Lichess
from allie.lichess.config import Config
from allie.lichess.engine import Engine, Play
from allie.lichess.llm import Reply
from allie.lichess.mock import MockLichess


def wait(cond, timeout=60):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


@pytest.fixture(scope="module")
def engine(tiny):
    e = Engine(tiny)
    yield e
    e.close()


class Model:
    """A stand-in model: records each call, says `text` (or nothing), optionally after a
    gate is opened."""

    def __init__(self, text="Fair enough.", gated=False):
        self.calls, self.text, self.gate = [], text, threading.Event()
        if not gated:
            self.gate.set()

    def __call__(self, system, messages):
        self.calls.append((system, messages))
        self.gate.wait(10)
        out = json.dumps({"speak": bool(self.text), "text": self.text})
        return Reply(
            [{"type": "text", "text": out}], bool(self.text), self.text, 0.1, 0.2, 0.001
        )

    def turn(self, i=-1):
        return self.calls[i][1][-1]["content"]


class Game:
    """A match as the chat sees it, with a client that records posts and serves the chat."""

    def __init__(self, engine, opp="opp", white=False, rated=False, **cfg):
        self.posts, self.lines = [], []
        client = SimpleNamespace(
            chat=lambda gid, text, room: self.posts.append((room, text)),
            chat_lines=lambda gid: self.lines,
        )
        cfg = Chat(enabled=True, gap=0, poll=0.05, **cfg)
        config = SimpleNamespace(chat=cfg, play=Play())
        bot = SimpleNamespace(me="allie", config=config, client=client, engine=engine)
        self.match = SimpleNamespace(bot=bot, gid="g1", over=False)
        self.opp, self.white, self.rated = opp, white, rated

    def chatter(self, model, seed=0):
        self.c = Chatter(self.match, model, seed)
        return self.c

    def feed(self, *events):
        for e in events:
            self.c.put(e)
        self.c.q.join()

    def full(self, moves=""):
        player = lambda u: {"id": u, "name": u.capitalize(), "rating": 1500}
        w, b = ("allie", self.opp) if self.white else (self.opp, "allie")
        return {"type": "gameFull", "id": "g1", "variant": {"key": "standard"},
                "speed": "blitz", "rated": self.rated, "white": player(w),
                "black": player(b), "clock": {"initial": 180000, "increment": 2000},
                "initialFen": "startpos", "state": state(moves)}  # fmt: skip

    def say(self, text, who=None, room="player"):
        who = who or self.opp.capitalize()
        return {"type": "chatLine", "room": room, "username": who, "text": text}


def state(moves, status="started", **kw):
    clocks = {"wtime": 170000, "btime": 171000, "winc": 2000, "binc": 2000}
    return {"type": "gameState", "moves": moves, "status": status} | clocks | kw


def test_clean_and_played():
    assert clean('"Good game!"') == "Good game!" and clean("see https://x.org") == ""
    assert clean("May the best pawn win.</text>") == "May the best pawn win."
    long = "First sentence here, quite long. " * 3 + "x" * 50
    a, b = chat.parts(long)
    assert a.endswith("long.") and len(a) <= 140 and len(b) <= 140
    assert chat.parts("short") == ["short"] and chat.parts("") == []
    b = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3"):
        b.push_uci(u)
    moves = [m.uci() for m in b.move_stack]
    assert played("Nf3, the classic", moves) and played("nice game", moves)
    assert not played("try Nc6 here", moves)


def test_conversation(engine):
    """The hello, an answer per message with Allie's annotations and the position, an
    append-only conversation, and the system prompt cached for an hour."""
    m = Model("Hi! Enjoy the game.")
    g = Game(engine, p_moment=0)
    c = g.chatter(m)
    g.feed(g.full())
    assert g.posts == [("player", chat.HELLO)]
    g.feed(state("e2e4"), state("e2e4 e7e5"), g.say("hi gl"))
    assert len(m.calls) == 1 and g.posts[-1] == ("player", "Hi! Enjoy the game.")
    system, messages = m.calls[0]
    assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "CASUAL" in system[1]["text"] and "Opp (1500)" in system[1]["text"]
    turn = m.turn()
    assert "1. e4 them" in turn and "| Allie: " in turn and "| you " in turn
    assert "1... e5 you" in turn and 'Chat (player) Opp: "hi gl"' in turn
    assert "Now (ply 2): them to move." in turn and "Opening: King's Pawn Game" in turn
    assert "FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" in turn
    assert "the fixed greeting" in turn
    g.feed(state("e2e4 e7e5 g1f3"), g.say("wdyt about my next move?"))
    second = m.calls[1][1]
    assert second[0] == messages[0] and second[1]["role"] == "assistant"
    assert "2. Nf3 them" in m.turn() and "1. e4" not in m.turn()  # only what is new
    assert len(c.timings) == 2 and c.timings[0]["text"] == "Hi! Enjoy the game."


def test_moments_draw_resign(engine):
    """Unprompted calls at moments by probability; a declined draw offer; the end, with
    Allie's review; then answers after the game, by fetching the chat once its stream ends."""
    m = Model("")
    g = Game(engine, p_moment=0, p_draw=1, p_end=1, every=0, linger=5)
    g.chatter(m)
    moves = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6".split()
    g.feed(g.full(), *[state(" ".join(moves[: k + 1])) for k in range(len(moves))])
    assert m.calls == []  # p_moment 0: the updates wait
    them = "wdraw"
    g.feed(state(" ".join(moves), **{them: True}), state(" ".join(moves)))
    assert len(m.calls) == 1
    turn = m.turn()
    assert (
        "Event: they offer a draw." in turn and "you declined their draw offer" in turn
    )
    assert "1. e4 them" in turn  # the waiting updates
    g.feed(state(" ".join(moves + ["d2d3"]), status="resign", winner="black"))
    assert len(m.calls) == 2
    assert "you won by resignation, they resigned after 7 plies" in m.turn()
    assert "Review by Allie" in m.turn()
    m.text = "Sure: 4. d3 was the quiet turn."
    g.lines = [{"user": "Opp", "text": "how did I play?"}]
    g.feed(chat.END)
    wait(lambda: len(m.calls) == 3)
    wait(lambda: g.posts[-1] == ("player", "Sure: 4. d3 was the quiet turn."))
    assert 'Chat (player) Opp: "how did I play?"' in m.turn()
    time.sleep(0.3)
    assert len(m.calls) == 3  # each line once


def test_quiet_rated_and_close(engine):
    m = Model("Maybe Nc6 next?")
    g = Game(engine, opp="quiet", rated=True)
    c = g.chatter(m)
    g.feed(g.full(), state("e2e4"), g.say("blunder?"))
    assert len(m.calls) == 1 and "RATED" in m.calls[0][0][1]["text"]
    assert g.posts == [("player", chat.HELLO)]  # an unplayed move, rated: dropped
    m.text, m.gate = "Not telling!", threading.Event()
    g.c.put(g.say("and now?"))
    wait(lambda: len(m.calls) == 2)
    g.c.put(g.say("!quiet"))
    m.gate.set()
    g.c.q.join()
    assert g.posts[-1] == ("player", chat.QUIET) and len(g.posts) == 2
    g.feed(state("e2e4 e7e5"), g.say("hello?"))
    assert len(m.calls) == 2
    g4 = Game(engine, opp="early")
    g4.chatter(Model("x"))
    g4.c.put(g4.full())
    g4.c.put(g4.say("!quiet"))  # before the chat thread reads the game
    g4.c.q.join()
    assert g4.posts == [("player", chat.QUIET)]
    g2 = Game(engine, opp="quiet")
    g2.chatter(m)
    g2.feed(g2.full())
    assert g2.posts == []  # a rematch, still quiet
    g3 = Game(engine, opp="crash")
    c3 = g3.chatter(Model("x"))
    g3.feed(g3.full())
    c3.close()  # the game thread ended without the game ending
    g3.c.q.join()
    assert c3.done and c3.cancelled
    del c


def test_clocks_takeback_and_late(engine):
    """A reconnect's plies get both clocks as the bot's Game does; a takeback rewinds the
    chat's Game; an unprompted line is dropped if the game moved on while it was written."""
    from allie.lichess.engine import Game as Reference

    m = Model("What a move!", gated=True)
    g = Game(engine, p_moment=0, hello="")
    c = g.chatter(m)
    g.feed(g.full("e2e4 e7e5"))
    ref = Reference(engine, 1500, 1500, 180, 2, "blitz")
    ref.update(["e2e4", "e7e5"], 170, 171)
    assert c.game.clocks == ref.clocks == [170, 171]
    g.feed(state("e2e4 e7e5 g1f3"), state("e2e4"))  # a takeback of two plies
    assert c.game.moves == ["e2e4"] and c.sans == ["e4"] and max(c.views) == 1
    c.want = ("a test", "player", True, time.monotonic())
    c.put({"type": "opponentGone", "gone": False})
    wait(lambda: m.calls)
    for k in range(2, 6):
        c.put(state(" ".join("e2e4 e7e5 g1f3 b8c6 f1c4".split()[:k])))
    m.gate.set()
    c.q.join()
    assert g.posts == []  # late


def test_split():
    """The chat gets each event at once even while the game thread is busy; the game thread
    gets them all in order, then the stream's error."""
    got = []
    sink = SimpleNamespace(gid="g", done=False, put=got.append)

    def stream():
        yield {"type": "a"}
        yield None
        yield {"type": "b"}
        raise OSError("closed")

    events = split(stream(), sink)
    assert next(events) == {"type": "a"}
    wait(lambda: len(got) == 3)  # a, b and the end, while the game thread waits
    assert got == [{"type": "a"}, {"type": "b"}, chat.END]
    assert next(events) is None and next(events) == {"type": "b"}
    with pytest.raises(OSError):
        next(events)


def test_through_the_bot(engine, monkeypatch):
    """The bot on the mock server: the hello, an answer, the game's moves unchanged."""
    m = Model("Thanks, you too!")
    monkeypatch.setattr(chat, "model", lambda cfg: m)
    mock = MockLichess({"tok": "allie"}, house_delay=0.1, max_plies=16)
    cfg = Chat(enabled=True, gap=0, p_moment=0, p_end=0, linger=1, poll=0.1)
    config = Config(play=Play(think_time=False), chat=cfg)
    bot = Bot(config, Lichess("tok", mock.url, wait=0.1), engine)
    threading.Thread(target=bot.run, daemon=True).start()
    try:
        wait(lambda: bot.me is not None)
        gid = mock.challenge("x", "allie", color="white")
        wait(lambda: gid in mock.games and mock.games[gid].board.move_stack)
        g = mock.games[gid]
        g.say("x", "hi, have fun")
        wait(lambda: sum(e["username"] == "allie" for e in g.chat) >= 2)
        ours = [e["text"] for e in g.chat if e["username"] == "allie"]
        assert ours[:2] == [chat.HELLO, "Thanks, you too!"]
        wait(lambda: g.status != "started")
        g.say("x", "gg")  # after the game: fetched
        wait(lambda: sum(e["username"] == "allie" for e in g.chat) >= 3)
        assert not mock.rejected
    finally:
        bot.stop()
        mock.close()
        bot.join()


def test_ledger(tmp_path, caplog):
    t = [llm.datetime(2026, 10, 4, 23, tzinfo=llm.UTC)]
    path = tmp_path / "spend.json"
    ledger = lambda: llm.Ledger(str(path), 1.0, 1.5, now=lambda: t[0])
    a = ledger()
    assert a.allows() and a.add(0.6) == [0.6, 0.6] and a.allows()
    a.add(0.5)
    assert not a.allows() and not a.allows()
    assert sum("2026-10-04" in r.message for r in caplog.records) == 1
    b = ledger()  # a restart
    assert not b.allows()
    t[0] = llm.datetime(2026, 10, 5, 1, tzinfo=llm.UTC)
    assert b.allows() and b.add(0.4) == pytest.approx([0.4, 1.5]) and not b.allows()
    t[0] = llm.datetime(2026, 11, 1, tzinfo=llm.UTC)
    assert b.allows()
    path.write_text("{not json")  # corrupt: no paid calls, and the evidence stays
    c = ledger()
    assert not c.allows() and c.add(0.1) == [float("inf")] * 2
    assert path.read_text() == "{not json" and not c.allows()


class FakeAPI:
    """A local Messages API that streams: answers from a script of decisions (dicts), HTTP
    error codes, or ("sleep", seconds)."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append(
                    ({k.lower(): v for k, v in self.headers.items()}, body)
                )
                step = api.script.pop(0)
                stall = 0
                if isinstance(step, tuple) and step[0] == "sleep":
                    time.sleep(step[1])
                    step = {"speak": False, "text": ""}
                elif isinstance(step, tuple):  # ("stall", s): stall after message_start
                    stall, step = step[1], {"speak": False, "text": ""}
                if isinstance(step, int):
                    data = json.dumps(
                        {"type": "error", "error": {"type": "x", "message": "no"}}
                    )
                    self.send_response(step)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    return self.wfile.write(data.encode())
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                usage = {"input_tokens": 300, "output_tokens": 1, "cache_read_input_tokens": 700,
                         "cache_creation_input_tokens": 100,
                         "cache_creation": {"ephemeral_5m_input_tokens": 100,
                                            "ephemeral_1h_input_tokens": 0}}  # fmt: skip
                message = {"id": "msg_1", "type": "message", "role": "assistant",
                           "model": body["model"], "content": [], "stop_reason": None,
                           "stop_sequence": None, "usage": usage}  # fmt: skip
                text = json.dumps(step)
                for e in (
                    {"type": "message_start", "message": message},
                    {"type": "content_block_start", "index": 0,
                     "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0,
                     "delta": {"type": "text_delta", "text": text}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                     "stop_sequence": None}, "usage": {"output_tokens": 12}},
                    {"type": "message_stop"},
                ):  # fmt: skip
                    self.wfile.write(
                        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode()
                    )
                    self.wfile.flush()
                    time.sleep(stall)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_claude(monkeypatch, tmp_path):
    """The SDK path on a local streaming API: the request (structured output, effort,
    thinking, fallback, caching), the key file, cost into the ledger, timeouts, a bad key."""
    pytest.importorskip("anthropic")
    say = {"speak": True, "text": "Nice move!"}
    api = FakeAPI([say, ("sleep", 1.5), ("stall", 3), 401, say])
    monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    key, ledger = tmp_path / "key", tmp_path / "spend.json"
    key.write_text("sk-test-123\n")
    key.chmod(0o600)
    cfg = Chat(enabled=True, timeout=1.0, key_file=str(key), ledger=str(ledger))
    m = llm.make(cfg)
    system = [
        {"type": "text", "text": "persona", "cache_control": {"type": "ephemeral"}}
    ]
    r = m(system, [{"role": "user", "content": "update"}])
    assert r.speak and r.text == "Nice move!" and r.first <= r.seconds
    headers, body = api.requests[0]
    assert headers["x-api-key"] == "sk-test-123" and body["stream"] is True
    assert "server-side-fallback-2026-07-01" in headers["anthropic-beta"]
    assert body["model"] == "claude-sonnet-5-5" and body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "between_tools"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["schema"] == llm.DECISION
    assert body["cache_control"] == {"type": "ephemeral"} and body["system"] == system
    usd = (300 * 2 + 100 * 2.5 + 700 * 0.2 + 12 * 10) / 1e6
    assert r.usd == pytest.approx(usd)
    day = llm.datetime.now(llm.UTC).strftime("%Y-%m-%d")
    assert json.loads(ledger.read_text())[day] == pytest.approx(usd)
    start = time.monotonic()
    assert m(system, [{"role": "user", "content": "u"}]) is None  # timeout
    assert time.monotonic() - start < 1.4
    start = time.monotonic()
    assert (
        m(system, [{"role": "user", "content": "u"}]) is None
    )  # stalled after starting
    assert time.monotonic() - start < 1.4  # the total deadline
    partial = (
        300 * 2 + 100 * 2.5 + 700 * 0.2 + (1 + 300) * 10
    ) / 1e6  # + all it could write
    assert json.loads(ledger.read_text())[day] == pytest.approx(usd + partial)
    assert m(system, [{"role": "user", "content": "u"}]) is None  # 401: off
    assert m(system, [{"role": "user", "content": "u"}]) is None
    assert len(api.requests) == 4
    key.unlink()
    assert llm.make(cfg)(system, []) is None  # no key


def test_config(tmp_path):
    from allie.lichess.config import load

    p = tmp_path / "bot.toml"
    p.write_text(
        '[chat]\nenabled = true\nrooms = ["player", "spectator"]\np_moment = 0.2\n'
    )
    c = load(p)
    assert c.chat.enabled and "spectator" in c.chat.rooms and c.chat.p_moment == 0.2
    p.write_text("[chat]\nstockfish = 'x'\n")
    with pytest.raises(ValueError):
        load(p)
