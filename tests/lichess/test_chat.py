import threading
import time
from types import SimpleNamespace

import chess
import pytest
import torch

from allie.lichess import chat
from allie.lichess.bot import Bot
from allie.lichess.chat import Chat, Chatter, clean
from allie.lichess.client import Lichess
from allie.lichess.config import Config
from allie.lichess.engine import WDL, Engine, Play
from allie.lichess.mock import MockLichess
from allie.lichess.tokens import MOVE_ID


def wait(cond, timeout=60):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


class Recorder:
    """A stand-in model: records each prompt, answers with a canned line."""

    def __init__(self, reply="Nice one!", delay=0.0):
        self.prompts, self.reply, self.delay = [], reply, delay

    def __call__(self, system, prompt):
        assert "SKIP" in system
        self.prompts.append(prompt)
        time.sleep(self.delay)
        return self.reply


def test_clean():
    assert clean("SKIP") == clean("skip.") == clean("  ") == []
    assert clean("see https://example.com") == clean("www.x.org is fun") == []
    assert clean('"Good game!"') == ["Good game!"]
    assert clean("**Well** played") == ["Well played"]
    long = clean("word " * 60)[0]
    assert len(long) <= 140 and long.endswith("…")
    assert clean("one\n\ntwo\nthree", 2) == ["one", "two"]


def logits(probs, wdl):
    """A model output with these move probabilities and this (side to move's) win/draw/loss."""
    z = torch.full((2432,), -50.0)
    for u, p in probs.items():
        z[MOVE_ID[u]] = float(torch.tensor(p).log())
    z[WDL] = torch.tensor(wdl).log()
    return z


class Fake:
    """A Match and Game in the shape the chat reads, driven by hand."""

    def __init__(self, white=False, **cfg):
        self.posts = []
        client = SimpleNamespace(
            chat=lambda gid, text, room: self.posts.append((room, text))
        )
        config = SimpleNamespace(chat=Chat(enabled=True, gap=0, **cfg))
        self.bot = SimpleNamespace(config=config, client=client, me="allie")
        self.gid, self.white, self.opponent, self.over = "g1", white, "opp", False
        self.game = SimpleNamespace(moves=[], board=chess.Board(), logits=None, used=[], tokens=[],
                                    elo=(1500, 1500))  # fmt: skip
        self.full = dict(type="gameFull", white=dict(name="Opp", rating=1480),
                         black=dict(name="allie", rating=1500), speed="blitz", rated=True,
                         clock=dict(initial=180000, increment=2000))  # fmt: skip

    def play(self, c, moves, probs=None, wdl=(0.4, 0.2, 0.4), status="started", winner=None,
             first=False):  # fmt: skip
        """The game reaches `moves`; the model's view there: probs and wdl, for the side to
        move (probs: uniform over the legal moves by default)."""
        g = self.game
        g.moves, g.board = list(moves), chess.Board()
        for u in moves:
            g.board.push_uci(u)
        if probs is None:
            probs = {m.uci(): 1.0 for m in g.board.legal_moves}
        g.logits = logits({u: p / sum(probs.values()) for u, p in probs.items()}, wdl)
        state = dict(type="gameState", moves=" ".join(moves), wtime=170000, btime=171000,
                     status=status, winner=winner)  # fmt: skip
        c.seen(self.full | dict(state=state) if first else state)

    def say(self, c, text, who=None, room="player"):
        who = (
            who or self.opponent.capitalize()
        )  # a username; the opponent's id in lower case
        c.heard(dict(type="chatLine", username=who, room=room, text=text))


def test_compliment_and_privacy():
    """Black bot. White's rare move that Allie rates strong gets a compliment; a rare blunder
    gets nothing, and an answer during the game leaves it and our good prospects out."""
    f, llm = Fake(every=0), Recorder()
    c = Chatter(f, llm)
    f.play(c, [], {"e2e4": 0.89, "f2f3": 0.01, "d2d4": 0.1}, first=True)  # white's view
    wait(lambda: f.posts)
    assert f.posts[0] == ("player", chat.HELLO)
    f.play(c, ["f2f3"], wdl=(0.2, 0.2, 0.6))  # our view: f3 was strong (0.5 -> 0.3)
    f.play(c, ["f2f3", "e7e5"], {"g1h3": 0.97, "g2g4": 0.03})
    wait(lambda: len(f.posts) == 2)
    assert "rare and strong" in llm.prompts[0]
    assert (
        "1. f3: Allie gave it 1% for a 1480 player (it expected e4 most"
        in llm.prompts[0]
    )
    f.play(c, ["f2f3", "e7e5", "g2g4"], wdl=(0.8, 0.1, 0.1))  # a blunder: 0.5 -> 0.85
    f.say(c, "what's the best move here?")
    wait(lambda: len(f.posts) == 3)
    p = llm.prompts[-1]
    assert "  Opp: what's the best move here?" in p and "Answer your opponent's" in p
    assert "Their last move" not in p and "Allie's estimate" not in p
    f.play(c, ["f2f3", "e7e5", "g2g4", "d8h4"], status="mate", winner="black")
    wait(lambda: len(f.posts) == 4)
    p = llm.prompts[-1]
    assert (
        "you won by checkmate" in p and "Their last move 2. g4: Allie gave it 3%" in p
    )
    assert (
        "Your last move 2...Qh4#" in p and "Their most surprising move: 1. f3 (1%)" in p
    )
    assert len(llm.prompts) == 3  # no remark on the blunder


def test_quiet_and_limits():
    f, llm = Fake(white=True, replies=2), Recorder("Thanks, you too!")
    f.opponent = "quiet-opp"
    c = Chatter(f, llm)
    f.play(c, [], first=True)
    f.say(c, "hi")
    f.say(c, "hi", who="allie")  # our own echo
    f.say(c, "Takeback sent", who="lichess")
    f.say(c, "nice bot", who="someone", room="spectator")  # spectators: not in rooms
    wait(lambda: len(f.posts) == 2)
    assert f.posts == [("player", chat.HELLO), ("player", "Thanks, you too!")]
    assert len(llm.prompts) == 1
    f.say(c, "!quiet please")
    wait(lambda: len(f.posts) == 3)
    assert f.posts[-1] == ("player", chat.QUIET)
    f.say(c, "hello?")
    f.play(c, ["e2e4", "e7e5"], status="mate")
    time.sleep(0.3)
    assert len(f.posts) == 3 and len(llm.prompts) == 1
    c2 = Chatter(f, llm)  # a rematch: still quiet, no hello
    f.play(c2, [], first=True)
    time.sleep(0.2)
    assert len(f.posts) == 3


def test_burst_and_budget():
    """Messages that arrive while the model writes get one answer, to the last; replies stop
    at the budget."""
    f, llm = Fake(white=True, replies=3, hello=""), Recorder(delay=0.3)
    f.opponent = "burst-opp"
    c = Chatter(f, llm)
    f.play(c, [], first=True)
    f.say(c, "a")
    wait(lambda: llm.prompts)
    for t in ("b", "c", "d", "e"):
        f.say(c, t)
    wait(lambda: len(f.posts) == 2)
    time.sleep(0.5)
    assert len(llm.prompts) == 2 and "  Burst-opp: c\n" in llm.prompts[1]


def test_slow_model_never_delays_the_game():
    f, llm = Fake(white=True, every=0, hello=""), Recorder(delay=2)
    f.opponent = "slow-opp"
    c = Chatter(f, llm)
    f.play(c, [], first=True)
    start = time.monotonic()
    for _ in range(5):
        f.say(c, "hey")
    assert time.monotonic() - start < 0.5


@pytest.fixture
def engine(tiny):
    e = Engine(tiny)
    yield e
    e.close()


def test_chat_through_the_protocol(engine, monkeypatch):
    """The bot on the mock server: the hello, an answer to the opponent and the post-game
    message (remarks off: the tiny model's are random), and the moves stay legal."""
    llm = Recorder("Good luck to you too!")
    monkeypatch.setattr(chat, "model", lambda *a: llm)
    mock = MockLichess({"tok": "allie"}, house_delay=0.1, max_plies=24)
    config = Config(
        play=Play(think_time=False), chat=Chat(enabled=True, gap=0, remarks=0)
    )
    bot = Bot(config, Lichess("tok", mock.url, wait=0.1), engine)
    threading.Thread(target=bot.run, daemon=True).start()
    try:
        wait(lambda: bot.me is not None)
        gid = mock.challenge("x", "allie", color="white")
        wait(lambda: gid in mock.games and mock.games[gid].board.move_stack)
        g = mock.games[gid]
        g.say("x", "hi, have fun")
        wait(lambda: g.status != "started")
        wait(lambda: llm.prompts and "game is over" in llm.prompts[-1])
        wait(lambda: sum(e["username"] == "allie" for e in g.chat) == 3)
        ours = [e["text"] for e in g.chat if e["username"] == "allie"]
        assert ours[0] == chat.HELLO and ours[1:] == ["Good luck to you too!"] * 2
        assert any(
            "Answer your opponent's" in p and "x: hi, have fun" in p
            for p in llm.prompts
        )
        assert not mock.rejected
    finally:
        bot.stop()
        mock.close()
        bot.join()


def test_config_section(tmp_path):
    from allie.lichess.config import load

    p = tmp_path / "bot.toml"
    p.write_text('[chat]\nenabled = true\nrooms = ["player", "spectator"]\nevery = 6\n')
    c = load(p)
    assert c.chat.enabled and "spectator" in c.chat.rooms and c.chat.every == 6
    assert not load(p, ["chat.enabled=false"]).chat.enabled
    p.write_text("[chat]\nbogus = 1\n")
    with pytest.raises(ValueError):
        load(p)


class FakeAPI:
    """A local Messages API: records each request, answers from a script of
    ("text", stop_reason) or (HTTP status,) or ("sleep", seconds)."""

    def __init__(self, script):
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
                if step[0] == "sleep":
                    time.sleep(step[1])
                    step = ("late", "end_turn")
                if isinstance(step[0], int):
                    out, code = (
                        {"type": "error", "error": {"type": "x", "message": "no"}},
                        step[0],
                    )
                else:
                    usage = dict(input_tokens=300, output_tokens=12, cache_read_input_tokens=700,
                                 cache_creation_input_tokens=0)  # fmt: skip
                    out, code = dict(id="msg_1", type="message", role="assistant",
                                     model=body["model"], content=[dict(type="text", text=step[0])],
                                     stop_reason=step[1], stop_sequence=None, usage=usage), 200  # fmt: skip
                data = json.dumps(out).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_claude_requests(monkeypatch, tmp_path):
    """The SDK path against a local API: the request's shape, the key file, refusals, timeouts,
    and a bad key turning the model off."""
    pytest.importorskip("anthropic")
    api = FakeAPI([("Nice move!", "end_turn"), ("x", "refusal"), ("sleep", 1.5), (401,),
                   ("never", "end_turn")])  # fmt: skip
    monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "key"
    path.write_text("sk-test-123\n")
    path.chmod(0o600)
    cfg = Chat(enabled=True, timeout=1.0, key_file=str(path))
    m = chat.make(cfg)
    assert m("system text", "facts") == "Nice move!"
    headers, body = api.requests[0]
    assert headers["x-api-key"] == "sk-test-123"
    assert "server-side-fallback-2026-07-01" in headers["anthropic-beta"]
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 200
    assert body["thinking"] == {"type": "between_tools"}
    assert body["output_config"] == {"effort": "low"} and body["fallbacks"] == "default"
    assert body["system"] == [
        dict(type="text", text="system text", cache_control=dict(type="ephemeral"))
    ]
    assert body["messages"] == [dict(role="user", content="facts")]
    assert m("s", "p") == ""  # refusal
    start = time.monotonic()
    assert m("s", "p") == ""  # timeout
    assert time.monotonic() - start < 1.4
    assert m("s", "p") == ""  # 401: off
    assert m("s", "p") == "" and len(api.requests) == 4
    assert m.usage["calls"] == 2 and m.usage["cached"] == 1400
    path.unlink()
    assert chat.make(cfg)("s", "p") == ""  # no key: fixed lines only
