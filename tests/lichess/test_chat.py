import json
import re
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
            chat=lambda gid, text, room, **kw: self.posts.append((room, text)),
            chat_lines=lambda gid, **kw: self.lines,
        )
        cfg = Chat(enabled=True, gap=0, poll=0.05, **{"recent": 0} | cfg)
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
    assert chat.gg("GG wp") and chat.gg("ggs") and not chat.gg("eggs")
    for t in (
        "why did you say gg?",
        "not a good game",
        "gg ez lol",
        "ty",
        "ty thanks",
        "g",
    ):
        assert not chat.gg(t)
    assert chat.gg("well played") and chat.gg("nice game ty") and chat.gg("g g")
    start = time.monotonic()
    assert not chat.gg("g" * 5000)
    assert time.monotonic() - start < 0.05  # no backtracking
    b = chess.Board()
    for u in "e2e4 e7e5 g1f3 b8c6".split():
        b.push_uci(u)
    assert chat.recase("nf3 then bc4 or bb5, be ok", b) == "Nf3 then Bc4 or Bb5, be ok"
    b = chess.Board("7k/8/8/1B6/2p5/1P6/8/4K3 w - - 0 1")  # bxc4 and Bxc4 both legal
    assert chat.recase("bxc4 or Bxc4", b) == "bxc4 or Bxc4"
    long = "First sentence here, quite long. " * 3 + "x" * 50
    a, b = chat.parts(long)
    assert a.endswith("long.") and len(a) <= 140 and len(b) <= 140
    assert chat.parts("short") == ["short"] and chat.parts("") == []
    b = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3"):
        b.push_uci(u)
    moves = [m.uci() for m in b.move_stack]
    assert played("Nf3, the classic", moves) and played("nice game", moves)
    assert not played("try Nc6 here", moves) and not played("maybe Bb5", moves)
    assert played("my c6 pawn was loose", ["e2e4", "c7c6"])  # a square, not a move
    assert not played("Next I want to push d4.", moves)  # a legal push: a suggestion


def test_conversation(engine):
    """The hello, an answer per message with Allie's annotations and the position, an
    append-only conversation, and the system prompt cached for an hour."""
    m = Model("hi! enjoy the game.")
    g = Game(engine, p_moment=0)
    c = g.chatter(m)
    g.feed(g.full())
    assert g.posts == [("player", chat.HELLO)]
    g.feed(state("e2e4"), state("e2e4 e7e5"), g.say("hi gl"))
    assert len(m.calls) == 1 and g.posts[-1] == ("player", "hi! enjoy the game.")
    system, messages = m.calls[0]
    assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "CASUAL" in system[1]["text"] and "Opp (1500)" in system[1]["text"]
    turn = m.turn()
    assert "1. e4 them" in turn and "| you expected " in turn and "| feel " in turn
    assert (
        re.search(r"1\.\.\. e5 you \d+s \| feel", turn) is not None
        and 'Chat (player) Opp: "hi gl"' in turn
    )
    assert "Now (ply 2): them to move." in turn and "Opening: King's Pawn Game" in turn
    assert "FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" in turn
    assert "the fixed greeting" in turn
    g.feed(state("e2e4 e7e5 g1f3"), g.say("wdyt about my next move?"))
    second = m.calls[1][1]
    assert second[0] == messages[0] and second[1]["role"] == "assistant"
    assert "2. Nf3 them" in m.turn() and "1. e4" not in m.turn()  # only what is new
    assert len(c.timings) == 2 and c.timings[0]["text"] == "hi! enjoy the game."


def test_moments_draw_resign(engine):
    """Unprompted calls at moments by probability; a declined draw offer; the end, with
    Allie's review; then answers after the game, by fetching the chat once its stream ends."""
    m = Model("")
    g = Game(engine, p_moment=0, quiet_p_moment=0, p_draw=1, p_end=1, every=0, linger=5,
             min_plies=0)  # fmt: skip
    g.chatter(m)
    moves = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6".split()
    g.feed(g.full(), *[state(" ".join(moves[: k + 1])) for k in range(len(moves))])
    assert m.calls == []  # p_moment 0: the updates wait
    g.feed(g.say("hi"))  # they chat: the chatty rates
    them = "wdraw"
    g.feed(state(" ".join(moves), **{them: True}), state(" ".join(moves)))
    assert len(m.calls) == 2
    turn = m.turn()
    assert (
        "Event: they offer a draw." in turn and "you declined their draw offer" in turn
    )
    assert "1. e4 them" in m.turn(0)  # the waiting updates
    g.feed(state(" ".join(moves + ["d2d3"]), status="resign", winner="black"))
    assert len(m.calls) == 3
    assert "you won by resignation, they resigned after 7 plies" in m.turn()
    assert "Your sense of the game" in m.turn()
    m.text = "sure: 4. d3 was the quiet turn."
    g.lines = [{"user": "Opp", "text": "how did I play?"}]
    g.feed(chat.END)
    wait(lambda: len(m.calls) == 4)
    wait(lambda: g.posts[-1] == ("player", "sure: 4. d3 was the quiet turn."))
    assert 'Chat (player) Opp: "how did I play?"' in m.turn()
    time.sleep(0.3)
    assert len(m.calls) == 4  # each line once


def script(c, feel):
    """Allie's views from a script: feel(ply) -> our expected score; uniform moves."""

    def see(moves, w, b):
        c.game.update(
            moves, None if w is None else w / 1000, None if b is None else b / 1000
        )
        board = chess.Board()
        for u in moves:
            board.push_uci(u)
        legal = [m.uci() for m in board.legal_moves]
        e = feel(len(moves))
        c.views[len(moves)] = chat.View(
            dict.fromkeys(legal, 1 / len(legal)), (e, 0.0, 1 - e), 5.0
        )

    c.see = see


def test_reciprocity(engine, caplog):
    """While they don't chat, remarks only in the quiet slots (the opening, a plan or a
    compliment, the finish); their gg after the game gets the fixed reply only if nothing
    was said after the game. Once they chat, the chatty rates; after two of our lines
    without a reply, the quiet ones again. Their lines are logged."""
    caplog.set_level("INFO", logger="allie.lichess.chat")
    m = Model("Nice.")
    g = Game(engine, opp="shy", hello="", quiet_p_moment=1, every=0, p_end=1, linger=5,
             min_plies=4)  # fmt: skip
    c = g.chatter(m)
    script(c, lambda n: 0.5 if n < 9 else 0.3)  # their 5th move (ply 9) is strong
    moves = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6".split()
    g.feed(g.full())
    for k in range(1, len(moves) + 1):
        g.feed(state(" ".join(moves[:k])))
    assert c.kinds == ["opening", "compliment"]  # not the plan: the middle slot is used
    g.feed(state(" ".join(moves), status="resign", winner="black"))
    assert c.kinds[-1] == "finish" and len(m.calls) == 3
    g.feed(g.say("gg wp"))
    assert g.posts[-1] == ("player", "Nice.") and len(g.posts) == 3  # no second gg
    assert "game g1 chat (player) Shy: gg wp" in caplog.text
    q = Game(engine, opp="shy2", hello="", quiet_p_moment=0, p_end=0, min_plies=0)
    q.chatter(Model("x"))
    q.feed(q.full(), state("e2e4 e7e5", status="resign", winner="black"), q.say("gg"))
    assert q.posts == [
        ("player", chat.GG)
    ]  # nothing said after the game: the fixed reply
    d = Game(engine, opp="fader", hello="", p_moment=1, quiet_p_moment=0, every=0)
    c = d.chatter(Model("hey"))
    d.feed(d.full(), d.say("hi"))
    assert c.engaged()
    c.moment(time.monotonic(), "endgame")  # a kind with no quiet slot
    c.respond()
    assert c.unreplied == 2 and not c.engaged()  # two lines unanswered: quiet again
    c.moment(time.monotonic(), "scramble")
    assert c.want is None


def test_facts_and_kinds(engine):
    """Each move's line says what it did (captures, checks, material, a piece left hanging);
    the mistake remark comes once our bad move is punished, the endgame and clock remarks
    once each; no remark for a mere surprise."""
    m = Model("")
    g = Game(engine, opp="tac", white=True, hello="", p_moment=1, every=0, scramble=150)
    c = g.chatter(m)
    script(c, lambda n: 0.5 if n < 5 else 0.3)  # our 3rd move (ply 5) is a mistake
    g.feed(g.full(), g.say("hi"))  # they chat: the chatty rates
    moves = "e2e4 d7d5 e4d5 d8d5 d1h5 d5h5".split()  # 3. Qh5?? Qxh5
    for k in range(1, 6):
        g.feed(state(" ".join(moves[:k])))
    g.feed(state(" ".join(moves), wtime=140000))
    turns = "\n".join(m.turn(i) for i in range(len(m.calls)))
    assert "2. exd5 you" in turns and "takes a pawn | feel" in turns
    assert "material: you +1" in turns and "left hanging: d5" not in turns  # a trade
    assert "3. Qh5 you" in turns and "left hanging: Qh5" in turns
    assert "3... Qxh5 them" in turns and "takes a queen" in turns
    assert "material: you -9" in turns
    assert c.kinds == ["mistake"]  # the clock remark waits: one call at a time
    g.feed(state(" ".join(moves + ["b1c3"]), wtime=130000))
    g.feed(state(" ".join(moves + ["b1c3", "h5e5"]), wtime=130000))
    assert c.kinds[0] == "mistake" and c.kinds[-1] == "scramble"  # the opening between


def test_recent_survives_an_outage(engine, tmp_path, caplog, monkeypatch):
    """A full disk (or any filesystem error) on the recent lines, or their lock held: the
    chat carries on without them, logged once, and a game still gets its lines posted."""
    import fcntl

    path = tmp_path / "recent.json"
    r = chat.Recent(str(path), 5)
    r.add("one")
    assert r.read() == ["one"]
    with open(str(path) + ".lock", "a") as held:  # another process holds the lock
        fcntl.flock(held, fcntl.LOCK_EX)
        start = time.monotonic()
        r.add("two")
        assert time.monotonic() - start < 2 and r.read() == []  # gave up; left alone
    r.retry = 0.0
    assert r.read() == ["one"]
    real = open

    def full(file, *args, **kwargs):
        if str(file).startswith(str(path)):
            raise OSError(122, "Disk quota exceeded")
        return real(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", full)
    r.retry = 0.0
    r.add("three")
    r.add("four")  # inside the wait: not even tried
    assert r.read() == []
    assert (
        sum("recent lines" in x.message for x in caplog.records) == 2
    )  # lock, then quota
    g = Game(engine, opp="disk", hello="", quiet_p_moment=1, every=0, recent=5,
             recent_file=str(path))  # fmt: skip
    g.chatter(Model("Still talking."))
    g.feed(g.full())
    for k in range(1, 7):
        g.feed(state(" ".join("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6".split()[:k])))
    assert g.posts and g.posts[-1] == ("player", "Still talking.")
    path.write_text("{corrupt")
    monkeypatch.setattr("builtins.open", real)
    r.retry = 0.0
    assert r.read() == []  # corrupt: empty, and rewritten with the next line
    r.add("five")
    assert r.read() == ["five"]


def test_one_goodbye(engine):
    """A finish remark queued when their gg arrives in the same batch: the fixed reply
    goes out and the remark is dropped, so the bot says goodbye once."""
    m = Model("A line.")
    g = Game(
        engine, opp="bye", hello="", quiet_p_moment=1, p_end=1, min_plies=2, linger=5
    )
    c = g.chatter(m)
    g.feed(g.full(), state("e2e4"), state("e2e4 e7e5"))
    m.gate.clear()
    c.put(state("e2e4 e7e5 g1f3", status="resign", winner="white"))
    c.put(g.say("gg"))
    m.gate.set()
    c.q.join()
    goodbyes = [x for x in g.posts if x[1] in (chat.GG, "A line.")]
    assert len(goodbyes) == 1


def test_recent_keeps_lines_when_a_read_fails(tmp_path, monkeypatch):
    path = tmp_path / "recent.json"
    r = chat.Recent(str(path), 5)
    r.add("one")
    real = open

    def stale(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and "r" in mode:
            raise OSError(116, "Stale file handle")
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", stale)
    r.add("two")
    monkeypatch.setattr("builtins.open", real)
    assert json.loads(path.read_text()) == ["one"]  # not replaced by ["two"]


def test_traded_and_hanging():
    b = chess.Board("4k3/8/8/3p4/4K3/8/8/8 w - - 0 1")
    b.push_uci("e4d5")  # the king takes: never a trade square, never hanging
    assert chat.traded(b) is None and chat.hanging(b, chess.WHITE) == []
    b = chess.Board()
    for u in "e2e4 d7d5 e4d5".split():
        b.push_uci(u)
    assert chat.traded(b) == chess.D5 and chat.hanging(b, chess.WHITE, chess.D5) == []
    assert chat.hanging(b, chess.WHITE) == ["d5"]


def test_recent_lines(engine, tmp_path):
    """Unprompted lines are remembered across games and passed in as not to be reused."""
    path = str(tmp_path / "recent.json")
    g = Game(
        engine,
        opp="r1",
        hello="",
        quiet_p_moment=1,
        every=0,
        recent=3,
        recent_file=path,
    )
    g.chatter(Model("A line to remember."))
    g.feed(g.full())
    for k, _ in enumerate("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6".split()):
        g.feed(state(" ".join("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6".split()[: k + 1])))
    assert json.loads(open(path).read()) == ["A line to remember."]
    h = Game(engine, opp="r2", hello="", recent=3, recent_file=path)
    c = h.chatter(Model("x"))
    h.feed(h.full())
    assert "- A line to remember." in c.system[1]["text"]


def test_short_game_ends_quietly(engine):
    """A moment queued by the last moves doesn't outlive the end: a quiet opponent's game of
    two plies ends without a call."""
    m = Model("wow")
    g = Game(engine, opp="brief", hello="", quiet_p_moment=1, p_end=0, every=0)
    c = g.chatter(m)
    g.feed(g.full())
    c.ply = lambda t, j, *a: c.moment(t, "compliment")  # every move a moment
    g.feed(state("e2e4 e7e5", status="resign", winner="black"))
    assert m.calls == [] and g.posts == []


def test_header_and_stale(engine, monkeypatch):
    """The header gives the rating Allie imitates, not the account's; a reply that comes
    STALE seconds after its message is not posted."""
    m = Model("ok", gated=True)
    g = Game(engine, opp="newcomer", hello="")
    g.match.bot.config.play = Play(rating="opponent")
    c = g.chatter(m)
    full = g.full()
    full["black"]["rating"] = 1335  # the bot account's own rating
    full["white"]["rating"] = 759
    g.feed(full)
    header = c.system[1]["text"]
    assert "You play as a 759-rated human" in header and "1335" not in header
    monkeypatch.setattr(chat, "STALE", 0.1)
    c.put(g.say("hi"))
    wait(lambda: m.calls)
    time.sleep(0.2)
    m.gate.set()
    c.q.join()
    assert g.posts == []


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
    c.want = ("plan", "player", True, time.monotonic())
    c.put({"type": "opponentGone", "gone": False})
    wait(lambda: m.calls)
    for k in range(2, 6):
        c.put(state(" ".join("e2e4 e7e5 g1f3 b8c6 f1c4".split()[:k])))
    m.gate.set()
    c.q.join()
    assert g.posts == []  # late
    stopped = threading.Event()
    g.match.bot.stopped = stopped
    m.gate = threading.Event()
    c.put(g.say("still there?"))
    wait(lambda: len(m.calls) == 2)
    stopped.set()  # the bot stops while the reply is written
    m.gate.set()
    c.q.join()
    assert g.posts == []


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
    m = Model("thanks, you too!")
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
        assert ours[:2] == [chat.HELLO, "thanks, you too!"]
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
                elif (
                    isinstance(step, tuple) and step[0] == "slow"
                ):  # late headers, slow body
                    time.sleep(step[1])
                    stall, step = step[1] / 6, {"speak": True, "text": "Late reply"}
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

        class Quiet(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # a client that gave up (a timeout test): printing it at exit can abort

        self.server = Quiet(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_claude(monkeypatch, tmp_path):
    """The SDK path on a local streaming API: the request (structured output, effort,
    thinking, fallback, caching), the key file, cost into the ledger, timeouts, a bad key."""
    pytest.importorskip("anthropic")
    say = {"speak": True, "text": "Nice move!"}
    api = FakeAPI([say, ("sleep", 1.5), ("stall", 3), ("slow", 0.7), 401, say])
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
    before = json.loads(ledger.read_text())[day]
    start = time.monotonic()
    assert (
        m(system, [{"role": "user", "content": "u"}]) is None
    )  # over the total deadline
    assert time.monotonic() - start < 1.4
    assert json.loads(ledger.read_text())[day] > before  # still charged
    assert m(system, [{"role": "user", "content": "u"}]) is None  # 401: off
    assert m(system, [{"role": "user", "content": "u"}]) is None
    assert len(api.requests) == 5
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


def test_ledger_survives_an_outage(tmp_path, caplog, monkeypatch):
    """A filesystem outage silences paid calls for RETRY seconds, keeps the charges it could
    not record, and writes them once the file works again."""
    t = [llm.datetime(2026, 10, 4, 12, tzinfo=llm.UTC)]
    path = tmp_path / "spend.json"
    a = llm.Ledger(str(path), 1.0, 1.5, now=lambda: t[0])
    assert a.add(0.1) == pytest.approx([0.1, 0.1])
    real = open

    def broken(file, *args, **kwargs):
        if str(file).startswith(str(path)):
            raise OSError(122, "Disk quota exceeded")
        return real(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", broken)
    assert a.add(0.2) == [float("inf")] * 2 and not a.allows()
    t[0] += llm.timedelta(seconds=a.RETRY + 1)
    assert not a.allows()  # still failing: another wait
    monkeypatch.setattr("builtins.open", real)
    assert not a.allows()  # inside the new wait
    t[0] += llm.timedelta(seconds=a.RETRY + 1)
    assert a.allows()
    assert json.loads(path.read_text())["2026-10-04"] == pytest.approx(0.3)
    assert sum("no paid calls" in r.message for r in caplog.records) == 1
