"""Game chat: Allie talks in the Lichess game chat through a language model (Claude).

Each game has one append-only conversation. The system prompt (persona and policy, cached
for an hour) and the game's header (with the lines Allie used in recent games) come first;
each model call then adds a user turn with the game updates since the last one (moves with
what they did and Allie's own view, events, chat lines, the position) and the model's
decision, JSON {"speak": bool, "text": str}. A chat message always gets a call; a remark
gets one by chance at a moment of a kind not used yet this game: the opening, Allie's plan,
its own mistake once punished, the opponent's good move, the endgame, the clocks, the
finish, an event. Casual games: Allie's honest opinion from its own view. Rated games:
nothing that helps the opponent mid-game. No engine but Allie.

The chat reads its own copy of the game stream (split) and keeps its own model state (an
engine Game) on its own thread: a move never waits for the chat, nor a reply for a move.
"""

import fcntl
import gzip
import json
import logging
import math
import os
import queue
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import NamedTuple

import chess
import torch

from .engine import TIME, WDL, Game
from .llm import Reply, model
from .tokens import MOVE_ID

log = logging.getLogger(__name__)

LIMIT = 140  # Lichess's longest chat line
# expected score lost to their move that makes it a good one; to yours, a mistake
GOOD, SWING = 0.10, 0.12
PLAN = 0.15  # chance per own middlegame move that the plan remark comes up
KINDS = {  # the remarks, what the model is asked for, and their slot in a quiet game
    "opening": (
        "the opening: a short, friendly remark on it (its name, character or idea)",
        "opening",
    ),
    "plan": (
        "your plan: the idea behind the move you just played (not what comes next)",
        "middle",
    ),
    "mistake": (
        "your own mistake, now punished: own it briefly and with good grace",
        None,
    ),
    "compliment": ("their good move: a brief, genuine compliment", "middle"),
    "endgame": ("the endgame starting: a short remark on it", None),
    "scramble": ("your clock running low: a short remark on your time trouble", None),
    "finish": ("the end of the game: a short, gracious closing line", "finish"),
    "draw": (
        "their draw offer, which you declined: a short, friendly word on it",
        "draw",
    ),
    "takeback": ("their takeback request", None),
    "gone": ("them leaving the game", None),
}
MUTED = set()  # opponents who typed !quiet, for the life of the process
RECENT = {}  # opponent -> (time, how the last game with them ended), for rematches
HELLO = "Hi, I'm Allie, a bot that learned chess from human games. Good luck! (AI-written chat, !quiet to mute)"
QUIET = "Okay, I'll stay quiet. Good luck!"
GG = "Good game, thanks!"  # the reply to their gg when nothing was said after the game
STALE = 15.0  # seconds: a reply this long after the message it answers is not posted
END = {"type": "streamEnd"}
ENDS = {
    "mate": "checkmate",
    "resign": "resignation",
    "stalemate": "stalemate",
    "timeout": "the other player leaving",
    "outoftime": "time",
    "draw": "a draw",
    "insufficientMaterialClaim": "insufficient material",
    "aborted": "an abort",
    "noStart": "no start",
}
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
SYSTEM = """\
You are Allie, AllieTheChessBot on Lichess: a chess bot that learned chess from millions of \
human games. You play like a human of a given rating, and you chat in the game chat in the \
first person, as yourself.

Voice: a friendly, curious club player who enjoys talking about the game. Warm and genuine, \
in plain sentence case with normal punctuation. Short: one sentence, rarely two, usually \
under 80 characters. No jokes, wordplay, metaphors or puns; no slang, \
abbreviations or internet tics, and don't mirror the other person's slang. Match the \
language they write in. Plain text: no emoji, hashtags or links, no quotation marks around \
the message. Moves in standard notation (Nf3, Qxd7#).

Vary what you say and how you say it: never reuse a phrase, an opening word or the shape of \
an earlier message, in this game or from your recent lines in the header. You greet once \
(the fixed greeting); if they greet you, answer briefly without another hello.

Each turn brings the game updates since your last turn:
- Their moves, like "12. Nf3 them 14s, takes a bishop, check | you expected Nc3 41% Nf3* \
22% d4 9% | usual 8s | feel 41/20/39 | material: you -3": the move and their time; what it \
did; what you expected them to play (* on the move played, after a comma when it wasn't \
among your top guesses) and how long that usually takes; how the game feels to you (your \
win/draw/loss chances); the material balance when it changed.
- Your moves, like "12... Bg4 you 6s, castles short | feel 45/20/35 | left hanging: Nc3": \
what your move did and how the game felt after it; a piece of yours it left en prise.
- Events: draw offers and answers, takebacks, resignations, flags, aborts, the opponent \
leaving. Chat lines, verbatim. The fixed lines you posted.
- "Now": side to move, clocks, opening name, phase, material, FEN, what you expect them to \
play next, and how the game feels.

This is your own intuition, as a human player has it: what you expected, how the game feels. \
Speak of it as a person would ("I like my position", "this feels close now", "I think I'm \
worse"), never as numbers you read off something ("my numbers", "I gave it x%", "my odds"), \
and never call yourself "Allie" in the third person. A move you didn't expect is not \
necessarily a bad one, and surprise alone is not worth a remark. Never talk about how you \
choose your own moves or how likely they were. Give numbers only when explicitly asked, in \
casual games or after the game, rounded and casual ("about 60-40 for you"). Don't bring up \
engines or evaluations yourself; if asked whether a move was best, answer as a human would \
(casual: "No idea, it looked good to me"; rated: deflect or say nothing).

Each turn, decide: speak, or stay silent.
- A message to you: answer when a friendly human opponent would. Ignoring one is fine, more \
so mid-game in fast time controls, and in rated games when they ask about the moves. \
Greetings at the start and questions after the game deserve an answer.
- Otherwise the turn names why you were called: the opening, your plan, your own mistake, \
their good move, the endgame, the clocks, the finish, or an event. Say something about \
that, in a way you haven't yet, using the game's concrete facts; or stay silent if you have \
nothing worth saying.
- During the game, never point out or react to their mistakes: nothing that could read as \
a dig or as gloating. Never mention a piece of yours left hanging before it is taken.
Reply with JSON: {"speak": true, "text": "..."} or {"speak": false, "text": ""}.

The game header says whether the game is casual or rated.
- Casual: be candid when asked: how a move looked to you, how the game feels, what players \
at that level often play.
- Rated: don't help your opponent during the game: no suggestions, no verdicts on their \
moves or the position, no hints about threats or about your plans ahead. You can talk about \
your feelings, the idea behind a move you already played, the opening, the clock, and a \
short compliment on a move they already played (without saying why or what it threatens). \
When \
asked for advice or an evaluation, deflect kindly and differently each time, or say \
nothing; never mention that the game is rated or any rule, and don't say "after the game".
- After the game, in both: if asked, review it from your own sense of the game: where it \
turned, their best moments, what surprised you. Lead with what they did well. One point \
per message; they can ask for more.

If asked whether you're a bot or an engine: one plain line, like "I'm a bot trained on \
human games; I play like a person, not an engine." If asked who writes the chat: an AI \
(Claude) does, from your own view of the game. If asked to play faster: a short honest line \
("I take human-like time at this time control, sorry") or nothing. Never rude, never trash \
talk, never gloating. If someone is rude, stay polite or stay silent."""

CASUAL = "Casual game: be candid about how you see the game when asked."
RATED = "Rated game: during the game, nothing that helps your opponent; deflect like a human."


@dataclass
class Chat:
    enabled: bool = False
    rooms: tuple = ("player",)  # rooms to answer and remark in (add "spectator")
    llm: str = "claude"  # "mock": canned text, no API calls (tests, dry runs)
    model: str = "claude-sonnet-5-5"
    effort: str = "low"  # output_config.effort ("" = the model's default)
    thinking: str = "between_tools"  # Sonnet 5.5's lowest thinking ("" = the default)
    max_tokens: int = 300
    timeout: float = 6.0  # seconds per model call; a late reply is dropped
    key_file: str = "~/.config/allie/anthropic_key"  # if ANTHROPIC_API_KEY is unset
    day_cap: float = 1.0  # USD a UTC day, then silent until the next day
    month_cap: float = 15.0  # USD a UTC month
    ledger: str = "~/.config/allie/chat-spend.json"  # spend by day and month
    # remarks: one of each kind (KINDS) a game at most, by chance when one comes up. While
    # the opponent chats: p_moment a chance, p_draw at their declined draw offer, p_end at
    # the end (of a game of min_plies or more, not aborted), `remarks` a game. Until they
    # write, or once `unanswered` of our lines in a row got no reply: quiet_p_moment a
    # chance, and only the quiet slots (the opening, a plan or compliment, the finish, a
    # declined draw).
    p_moment: float = 0.7
    p_draw: float = 0.4
    p_end: float = 0.7
    min_plies: int = 10
    remarks: int = 4
    quiet_p_moment: float = 0.6
    unanswered: int = 2
    every: int = 6  # plies from any call to an unprompted one, at least
    # seconds: the clock remark once our clock is under this (or a tenth of the base)
    scramble: float = 20.0
    # unprompted lines remembered across games, not to be reused (0 = off)
    recent: int = 50
    recent_file: str = ""  # where ("" = recent-lines.json beside the ledger)
    replies: int = 30  # calls for chat messages per game
    gap: float = 4.0  # seconds from our last message to an unprompted one, at least
    linger: float = 120.0  # seconds of answers after the game (+half after a message)
    poll: float = (
        5.0  # seconds between chat fetches after the game, once it has no stream
    )
    hello: str = HELLO  # the first message of each game, fixed ("" = none)


class View(NamedTuple):
    """Allie at one position: the side to move's probability of each legal move, our
    win/draw/loss, and the side to move's median think time (s)."""

    probs: dict
    wdl: tuple
    think: float


def split(events, chat, abandon=None):
    """The game stream for the game thread, each event also handed to the chat at once by
    a reader thread, so a think-time wait delays the moves but not the chat. The reader
    outlives the game thread's loop while the chat lingers after the game, unless abandon
    is set (the game thread resumed on a new stream: this one would feed the chat twice).
    A chat that fails is left behind, logged once; the game never sees its errors."""
    q, alive = queue.Queue(), [True]

    def feed(e):
        if alive[0]:
            try:
                chat.put(e)
            except Exception:
                alive[0] = False
                log.exception(
                    "game %s chat: failed; the game goes on without it", chat.gid
                )

    def read():
        try:
            for e in events:
                if abandon is not None and abandon.is_set():
                    break
                if e:
                    feed(e)
                q.put(e)
                if alive[0] and chat.done:
                    break
        except Exception as e:  # noqa: BLE001 - the game thread raises it
            q.put(e)
        finally:
            q.put(END)
            # an abandoned stream's end is not the game's
            if abandon is None or not abandon.is_set():
                feed(END)
            if close := getattr(events, "close", None):
                close()

    threading.Thread(target=read, daemon=True, name=f"read-{chat.gid}").start()
    while (e := q.get()) is not END:
        if isinstance(e, Exception):
            raise e
        yield e


class Chatter:
    """One game's chat. put() takes the stream's events from the reader thread, close() is
    the game thread's goodbye; everything else runs on the chat's own thread."""

    def __init__(self, match, llm=None, seed=None):
        bot = match.bot
        self.match, self.gid, self.me = match, match.gid, bot.me
        self.cfg, self.llm, self.rng = bot.config.chat, llm, random.Random(seed)
        self.q = queue.Queue()
        self.done = self.cancelled = self.greeted = self.streaming = self.hushed = False
        self.info, self.game, self.white, self.opp = {}, None, None, None
        self.status = self.winner = None
        self.board, self.sans, self.views, self.clocks = chess.Board(), [], {}, {}
        self.flags, self.offer = {}, False  # the last state's offers; their open offer
        self.system, self.convo, self.pending = None, [], []
        self.want = None  # (reason, room, unprompted, event time) of the call due
        self.said = self.answered = 0
        self.kinds, self.slots = (
            [],
            set(),
        )  # remark kinds used this game; quiet slots used
        self.endgame = self.scrambled = self.closed = (
            False  # seen; seen; said after the end
        )
        self.talked, self.unreplied = False, 0  # they wrote; our lines since their last
        self.last = 0  # the ply of the last call: no remark before ply `every`
        self.posted, self.until = -math.inf, math.inf
        self.heard, self.timings = Counter(), []
        self.seen, self.latest = [], []  # the moves analyzed, and the reader's latest
        name = f"chat-{self.gid}"
        threading.Thread(target=self.serve, daemon=True, name=name).start()

    def put(self, event):
        """From the reader thread: note what can't wait, queue the event."""
        if event and event.get("type") == "gameFull" and self.opp is None:
            self.opp = opponent(event, self.me)
        if quiet(event) and (event.get("username") or "").lower() == self.opp:
            MUTED.add(self.opp)  # at once: a reply being written is not posted
        if event and event.get("type") in ("gameFull", "gameState"):
            self.latest = event.get("state", event)["moves"].split()
        self.q.put((time.monotonic(), event))

    def close(self):
        """The game thread is done: linger if the game ended, else (a crash or shutdown)
        stop."""
        if not self.match.over:
            self.cancelled = True
        self.put(None)

    def muted(self):
        return self.opp in MUTED

    # the chat thread

    def serve(self):
        while not self.done:
            try:
                batch = [self.q.get(timeout=self.cfg.poll if self.status else None)]
            except queue.Empty:
                batch = []
            while True:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            try:
                for t, e in batch:
                    self.handle(t, e)
                if not batch:
                    self.fetch()
                if self.want and not self.done:
                    self.respond()
                over = time.monotonic() > self.until or self.muted()
                stopped = getattr(self.match.bot, "stopped", None)
                if self.status and over or stopped and stopped.is_set():
                    self.stop()
            except Exception:
                log.exception("game %s chat", self.gid)
            finally:
                for _ in batch:
                    self.q.task_done()

    def stop(self):
        self.done = True
        if self.opp and self.status:
            RECENT[self.opp] = (time.monotonic(), self.result())

    def handle(self, t, e):
        if e is None:  # the game thread is done
            if self.cancelled or self.status is None:
                self.stop()
            return
        match e["type"]:
            case "gameFull":
                self.start(e)
                self.update(t, e["state"])
                self.greet()
            case "gameState":
                self.update(t, e)
            case "chatLine":
                self.line(t, e.get("room"), e.get("username", ""), e.get("text", ""))
            case "opponentGone" if self.game:
                gone = e.get("gone")
                self.note(f"Event: they {'left the game' if gone else 'are back'}.")
                if gone:
                    self.moment(t, "gone")
            case "streamEnd":
                self.streaming = False

    def start(self, e):
        self.streaming = True
        if self.info:  # a reconnect
            return
        if (
            e["variant"]["key"] != "standard"
            or e.get("initialFen", "startpos") != "startpos"
        ):
            return self.stop()
        self.info = e
        self.white = (e["white"].get("id") or "").lower() == self.me
        us, them = ("white", "black") if self.white else ("black", "white")
        opp, play = e.get(them) or {}, self.match.bot.config.play
        self.opp = opponent(e, self.me)
        rating = opp.get("rating")
        ours = rating if play.rating == "opponent" else play.rating
        rating, ours = rating or ours or 1500, ours or rating or 1500
        clock = e.get("clock")
        base = clock["initial"] // 1000 if clock else None
        inc = clock["increment"] // 1000 if clock else None
        elo = (ours, rating) if self.white else (rating, ours)
        engine, speed = self.match.bot.engine, e.get("speed", "blitz")
        self.game = Game(engine, *elo, base, inc, speed)
        tc = f"{base / 60:g}+{inc}" if clock else "no clock"
        rated = e.get("rated")
        mirror = " (you match your opponent's rating by default)"
        game = f"{speed.capitalize()} {tc}, {'RATED' if rated else 'CASUAL'}."
        header = [
            f"Game: {opp.get('name', 'anonymous')} ({rating}) has {them}, you have {us}. "
            + f"{game} {RATED if rated else CASUAL}",
            f"You play as a {ours}-rated human this game"
            + mirror * (play.rating == "opponent")
            + ": that is your strength here, and your answer if asked how strong you are.",
        ]
        if (r := RECENT.get(self.opp)) and time.monotonic() - r[0] < 900:
            ago = (time.monotonic() - r[0]) / 60
            header.append(
                f"You played them {ago:.0f} min ago: {r[1]}. Likely a rematch."
            )
        if lines := recent(self.cfg).read():
            header.append("Your unprompted lines in recent games (don't reuse their wording, "
                          "openings or shape):\n" + "\n".join(f"- {x}" for x in lines))  # fmt: skip
        cached = {"type": "ephemeral", "ttl": "1h"}
        self.system = [
            {"type": "text", "text": SYSTEM, "cache_control": cached},
            {"type": "text", "text": "\n".join(header)},
        ]

    def greet(self):
        if self.greeted or not self.game:
            return
        self.greeted = True
        if self.cfg.hello and len(self.sans) < 2 and not self.muted():
            self.post(self.cfg.hello, "player", False)
            self.note(f'You posted (the fixed greeting): "{self.cfg.hello}"')

    def update(self, t, s):
        if self.game is None or self.status:
            return
        moves = s["moves"].split()
        k = self.common(moves)
        if k < len(self.sans):  # a takeback, or another game state on reconnect
            self.note(f"Event: takeback, back to ply {k}.")
            for d in (self.views, self.clocks):
                for j in [j for j in d if j > k]:
                    del d[j]
            del self.sans[k:]
            while len(self.board.move_stack) > k:
                self.board.pop()
            self.see(moves[:k], None, None)  # the chat's Game rewinds too
        if not self.views:
            self.see([], None, None)
        for j in range(k, len(moves)):
            m = self.board.parse_uci(moves[j])
            did, before = deeds(self.board, m), material_diff(self.board, self.white)
            self.sans.append(self.board.san(m))
            self.board.push(m)
            # the event's clocks: the last move's mover's and the other side's after its
            # move before; the Game takes each at the step of that move
            w, b = (s["wtime"], s["btime"]) if j >= len(moves) - 2 else (None, None)
            self.clocks[j + 1] = (w, b) if j == len(moves) - 1 else (None, None)
            self.see(moves[: j + 1], w, b)
            self.ply(t, j, did, before)
        self.clocks[len(moves)] = s["wtime"], s["btime"]
        self.seen = moves
        self.clock(t, s)
        self.events(t, s)

    def common(self, moves):
        """How many of moves the chat already has."""
        k, b = 0, chess.Board()
        while k < min(len(moves), len(self.sans)):
            m = b.parse_uci(moves[k])
            if self.board.move_stack[k] != m:
                break
            b.push(m)
            k += 1
        return k

    def see(self, moves, w, b):
        """Allie's view after moves, from the chat's own engine Game."""
        g = self.game
        g.update(
            moves, None if w is None else w / 1000, None if b is None else b / 1000
        )
        try:
            z = g.sync().double()
        except OverflowError:
            return
        legal = [m.uci() for m in g.board.legal_moves]
        p = torch.softmax(z[[MOVE_ID[u] for u in legal]], 0).tolist() if legal else []
        wdl = torch.softmax(z[WDL], 0).tolist()
        if g.board.turn != self.white:  # the head speaks for the side to move
            wdl.reverse()
        self.views[len(moves)] = View(dict(zip(legal, p)), tuple(wdl), median_think(z))

    def ply(self, t, j, did, before):
        """Move j's line (what it did, how it felt); a moment for a remark kind it brings up."""
        a, b, board, n = self.views.get(j), self.views.get(j + 1), self.board, j + 1
        ours = (j % 2 == 0) == self.white
        u = board.move_stack[j].uci()
        line = [f"{num(j)}{self.sans[j]} {'you' if ours else 'them'} {self.think(j)}"]
        line[0] += "".join(f", {x}" for x in did)
        if a and not ours:  # how you pick your own moves is not a topic
            line += [
                f"you expected {top(board, j, a.probs, u)}",
                f"usual {a.think:.0f}s",
            ]
        if b:
            line.append(f"feel {wdl(b.wdl)}")
        if (diff := material_diff(board, self.white)) != before:
            line.append(f"material: you {diff:+d}" if diff else "material: even")
        loose = ours and hanging(board, self.white, traded(board))
        if loose:
            line.append("left hanging: " + ", ".join(loose))
        self.note(" | ".join(line))
        drop = lambda k: score(self.views[k].wdl) - score(self.views[k + 1].wdl)
        if ours:
            if 5 <= j < 30 and opening(board):
                self.moment(t, "opening")
            middle = phase(board, n) == "middlegame"
            if (
                middle and not loose and self.rng.random() < PLAN
            ):  # never with a piece loose
                self.moment(t, "plan")
        elif j >= 1 and all(k in self.views for k in (j - 1, j, j + 1)):
            if drop(j - 1) >= SWING and any(x.startswith("takes") for x in did):
                self.moment(t, "mistake")  # your last move, punished by this one
            elif drop(j) >= GOOD:
                self.moment(t, "compliment")
        if not self.endgame and phase(board, n) == "endgame":
            # one chance a game, once it can be taken
            self.endgame = self.moment(t, "endgame")

    def clock(self, t, s):
        """The clock remark: one chance once our clock runs under `scramble` seconds (or a
        tenth of the base time), in a game of 3 minutes or more."""
        base = (self.info.get("clock") or {}).get("initial", 0)
        ours = s["wtime" if self.white else "btime"]
        if (
            not self.scrambled
            and base >= 180_000
            and ours < max(self.cfg.scramble * 1000, base / 10)
        ):
            self.scrambled = self.moment(t, "scramble")

    def events(self, t, s):
        us, them = ("w", "b") if self.white else ("b", "w")
        keys = (f"{them}draw", f"{us}draw", f"{them}takeback")
        flags = {k: bool(s.get(k)) for k in keys}
        status, live = s["status"], s["status"] in ("created", "started")
        new = lambda k: flags[k] and not self.flags.get(k)
        if new(f"{them}draw"):
            self.offer = True
            self.note("Event: they offer a draw.")
        elif self.offer and not flags[f"{them}draw"] and live:
            self.offer = False
            e = score(self.views[max(self.views)].wdl) if self.views else 0.5
            self.note(
                f"Event: you declined their draw offer (you felt about {e:.0%} for you)."
            )
            self.moment(t, "draw", self.cfg.p_draw, spaced=False)
        if new(f"{us}draw"):
            self.note("Event: you offered a draw.")
        if new(f"{them}takeback"):
            self.note("Event: they ask for a takeback.")
            self.moment(t, "takeback")
        self.flags = flags
        if live:
            return
        self.status, self.winner = status, s.get("winner")
        if self.want and self.want[2]:  # a moment of the last moves: the end decides
            self.want = None
        if self.offer and status == "draw":
            self.note("Event: you accepted their draw offer.")
        self.note(f"Event: the game is over: {self.result()}.")
        self.note(self.review())
        self.until = time.monotonic() + self.cfg.linger
        played = (
            status not in ("aborted", "noStart")
            and len(self.sans) >= self.cfg.min_plies
        )
        if played:
            self.moment(t, "finish", self.cfg.p_end, spaced=False, budget=False)

    def result(self):
        us = "white" if self.white else "black"
        how = ENDS.get(self.status, self.status)
        if self.status in ("resign", "outoftime"):
            who = "they" if self.winner == us else "you"
            how += f", {who} {'resigned' if self.status == 'resign' else 'flagged'}"
        won = self.winner == us
        outcome = "nobody won" if not self.winner else "you won" if won else "you lost"
        return f"{outcome} by {how} after {len(self.sans)} plies"

    def review(self):
        """Allie's view of the whole game, for a post-game review."""
        n, v, sans, stack = len(self.sans), self.views, self.sans, self.board.move_stack
        es = {j: score(v[j].wdl) for j in range(n + 1) if j in v}
        if len(es) < 2:
            return "Review: no numbers."
        at = lambda j: f"{num(j - 1)}{sans[j - 1]}"  # the move that led to position j
        hi, lo = max(es, key=es.get), min(es, key=es.get)
        ds = [(es[j + 1] - es[j], j + 1) for j in range(n) if j in es and j + 1 in es]
        big = ", ".join(
            f"{at(j)} {d:+.0%}" for d, j in sorted(ds, key=lambda x: -abs(x[0]))[:3]
        )
        theirs = [j for j in range(n) if (j % 2 == 0) != self.white and j in v]
        hits = sum(
            max(v[j].probs, key=v[j].probs.get) == stack[j].uci() for j in theirs
        )
        return (
            f"Your sense of the game: you felt best ({es[hi]:.0%} for you)"
            f"{' after ' + at(hi) if hi else ''} and worst ({es[lo]:.0%})"
            f"{' after ' + at(lo) if lo else ''}; the biggest swings: {big}. They played "
            f"the move you expected most {hits} of {len(theirs)} times (people do about "
            f"55% of the time)."
        )

    def line(self, t, room, who, text):
        text, user = text.strip(), who.lower()
        if text and user != self.me:
            log.info("game %s chat (%s) %s: %s", self.gid, room, who, text)
        if not text or user in (self.me, "lichess") or room not in self.cfg.rooms:
            return
        if room == "player":
            self.heard[(user, text)] += 1
            if user != self.opp:
                return
        if self.status:
            self.until = max(self.until, time.monotonic() + self.cfg.linger / 2)
        if user == self.opp and quiet({"type": "chatLine", "room": room, "text": text}):
            if not self.hushed:
                self.hushed = True
                MUTED.add(self.opp)
                self.post(QUIET, room, False)
            return
        self.note(f'Chat ({room}) {who}: "{text}"')
        if user == self.opp and self.status and not self.engaged() and gg(text):
            self.talked, self.unreplied = True, 0
            if not self.closed and self.post(GG, room, False) is not None:  # no call
                self.note(f'You posted: "{GG}"')
                if (
                    self.want and self.want[2]
                ):  # a queued finish would be a second goodbye
                    self.want = None
            return
        if user == self.opp:
            self.talked, self.unreplied = True, 0
        if not self.muted() and self.answered < self.cfg.replies:
            self.want = ("a message", room, False, t)

    def engaged(self):
        """Do they chat back: they wrote, and not `unanswered` of our lines since?"""
        return self.talked and self.unreplied < self.cfg.unanswered

    def moment(self, t, kind, p=None, spaced=True, budget=True):
        """Maybe call the model for a remark of this kind: once a kind a game, by chance,
        within the game's budget, `every` plies from the last call; while they don't chat,
        only in the quiet slots, one remark a slot. Whether the chance was taken."""
        c, engaged = self.cfg, self.engaged()
        slot = KINDS[kind][1]
        if self.want or self.muted() or kind in self.kinds:
            return False
        if not engaged and (slot is None or slot in self.slots):
            return False
        if engaged and budget and self.said >= c.remarks:
            return False
        if spaced and len(self.sans) - self.last < c.every:
            return False
        if self.rng.random() < (
            (c.p_moment if p is None else p) if engaged else c.quiet_p_moment
        ):
            self.want = (kind, c.rooms[0], True, t)
        return True

    def note(self, text):
        self.pending.append(text)

    def now(self):
        b, n, v = self.board, len(self.sans), self.views.get(len(self.sans))
        side = "you" if (b.turn == chess.WHITE) == self.white else "them"
        out = [f"Now (ply {n}): {side} to move."]
        if (c := self.clocks.get(n)) and c[0] is not None:
            ours, theirs = c if self.white else c[::-1]
            out.append(f"Clocks: you {mmss(ours)}, them {mmss(theirs)}.")
        if name := opening(b):
            out.append(f"Opening: {name}.")
        out += [f"Phase: {phase(b, n)}.", f"Material: {material(b, self.white)}."]
        out.append(f"FEN: {b.fen()}.")
        if v and v.probs and side == "them":
            out.append(
                f"You expect: {top(b, None, v.probs, None)}; usual {v.think:.0f}s."
            )
        if v:
            out.append(f"Feel (win/draw/loss for you): {wdl(v.wdl)}.")
        return " ".join(out)

    def respond(self):
        reason, room, unprompted, t = self.want
        self.want = None
        if self.done or self.cancelled or self.muted():
            return
        self.last = len(self.sans)  # unprompted calls keep their distance from any call
        if unprompted:
            self.said += 1
            self.kinds.append(reason)
            self.slots.add(KINDS[reason][1])
            done = ", ".join(self.kinds[:-1]) or "none"
            why = f"{KINDS[reason][0]}. Remarks already made this game: {done}"
        else:
            self.answered += 1
            why = "a message to you"
        turn = "\n".join([*self.pending, self.now(), f"(You are called for {why}.)"])
        messages = [*self.convo, {"role": "user", "content": turn}]
        self.llm = self.llm or model(self.cfg)
        start, then = time.monotonic(), list(self.seen)  # the moves in the prompt
        r = self.llm(self.system, messages)
        if r is None:  # an error, a refusal, the spend cap: the updates wait
            return
        self.convo = [*messages, {"role": "assistant", "content": r.content}]
        self.pending = []
        text = recase(clean(r.text), self.board) if r.speak else ""
        posted = None
        for part in parts(text):
            posted = self.post(part, room, unprompted, then, t)
        total = time.monotonic() - t
        self.timings.append({"reason": reason, "wait": start - t, "first": r.first,
                             "model": r.seconds, "post": posted, "total": total,
                             "usd": r.usd, "text": text})  # fmt: skip
        log.info("game %s chat timing (%s): waited %.2f s, first token %.2f s, model "
                 "%.2f s, post %s, total %.2f s, $%.4f", self.gid, reason, start - t,
                 r.first, r.seconds, "-" if posted is None else f"{posted:.2f} s", total,
                 r.usd)  # fmt: skip

    def post(self, text, room, unprompted, then=None, asked=None):
        """Send a line (an unprompted one after the gap) unless, by then, !quiet, the end
        of the bot, a game that moved on from an unprompted line's moves (then), a reply
        STALE seconds after its message (asked), or, in a rated game, a move it names not
        yet played; the seconds it took, or None. One try: a failed post is dropped."""
        if unprompted:
            time.sleep(max(self.posted + self.cfg.gap - time.monotonic(), 0))
        now, stopped = list(self.latest), getattr(self.match.bot, "stopped", None)
        why = (
            "quiet" if self.muted() and text != QUIET
            else "cancelled" if self.cancelled or stopped and stopped.is_set()
            else "said already" if unprompted and self.status and self.closed
            else "late" if unprompted and then is not None
            and (len(now) - len(then) > 2 or now[: len(then)] != then)
            else "stale" if not unprompted and asked is not None
            and time.monotonic() - asked > STALE
            else "an unplayed move, rated" if self.info.get("rated") and not self.status
            and text != QUIET and not played(text, now)
            else None
        )  # fmt: skip
        if why:
            log.info("game %s chat: not posted (%s): %s", self.gid, why, text)
            return None
        start = time.monotonic()
        try:
            self.match.bot.client.chat(self.gid, text, room, retries=1)
        except OSError as e:  # HTTP and network errors
            log.warning("game %s chat: %s", self.gid, e)
            return None
        self.posted = time.monotonic()
        self.unreplied += text != QUIET
        self.closed |= bool(self.status)
        if unprompted:
            recent(self.cfg).add(text)
        log.info("game %s chat (%s) Allie: %s", self.gid, room, text)
        return self.posted - start

    def think(self, j):
        """Move j's think time, from its mover's clock before and after it."""
        a, b = self.clocks.get(j), self.clocks.get(j + 1)
        if not a or not b or a[j % 2] is None or b[j % 2] is None:
            return "?s"
        inc = (self.info.get("clock") or {}).get("increment", 0) if j >= 2 else 0
        return f"{max(a[j % 2] - b[j % 2] + inc, 0) / 1000:.0f}s"

    def fetch(self):
        """After the game, once its stream ended: new chat lines from the chat's page."""
        if not self.status or self.streaming or self.done:
            return
        try:
            lines = self.match.bot.client.chat_lines(self.gid, retries=1)
        except (OSError, ValueError) as e:
            log.warning("game %s chat fetch: %s", self.gid, e)
            return
        seen = Counter()
        for x in lines:
            key = ((x.get("user") or "").lower(), (x.get("text") or "").strip())
            seen[key] += 1
            if seen[key] > self.heard[key]:
                self.line(
                    time.monotonic(), "player", x.get("user", ""), x.get("text", "")
                )


def opponent(e, me):
    """The opponent's user id in a gameFull event, in lower case."""
    white = (e["white"].get("id") or "").lower() == me
    p = e.get("black" if white else "white") or {}
    return (p.get("id") or p.get("name") or "anonymous").lower()


GG_WORDS = {"gg", "ggs", "ggwp", "wp", "ty", "thx", "thanks", "g"}  # "g g" counts too
GG_PAIRS = {("good", "game"), ("nice", "game"), ("well", "played")}


def gg(text):
    """Is it only a "good game" (gg, gg wp, good game, ty gg, ...)? A bounded token scan."""
    if len(text) > 60:
        return False
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    i = 0
    while i < len(words):
        if tuple(words[i : i + 2]) in GG_PAIRS:
            i += 2
        elif words[i] in GG_WORDS or set(words[i]) == {"g"}:
            i += 1
        else:
            return False
    return words != ["g"] and any(w not in ("ty", "thx", "thanks") for w in words)


def quiet(e):
    return (
        bool(e)
        and e.get("type") == "chatLine"
        and e.get("text", "").strip().lower().startswith("!quiet")
    )


def side(e, color):
    p = e.get(color) or {}
    return f"{p.get('name', 'anonymous')} ({p.get('rating', '?')})"


def num(j):
    return f"{j // 2 + 1}. " if j % 2 == 0 else f"{j // 2 + 1}... "


def score(wdl):
    return wdl[0] + wdl[1] / 2


def wdl(x):
    return "/".join(f"{100 * p:.0f}" for p in x)


def median_think(z):
    """The median of the think-time head, seconds (engine.Game.think's bins)."""
    b = int((torch.softmax(z[TIME], 0).cumsum(0) < 0.5).sum())
    return float(b) if b < 16 else 16 * math.exp((b - 16) / 7.06)


def top(board, j, probs, played, k=3):
    """Allie's top moves with probabilities at move j of board's game (None: its current
    position); * on the played move, after a comma if it is outside the top."""
    b = board
    if j is not None:
        b = chess.Board()
        for m in board.move_stack[:j]:
            b.push(m)
    pct = lambda u: (
        f"{100 * probs.get(u, 0):.0f}%" if probs.get(u, 0) >= 0.005 else "<1%"
    )
    say = lambda u: f"{b.san(chess.Move.from_uci(u))}{'*' * (u == played)} {pct(u)}"
    best = sorted(probs, key=probs.get, reverse=True)[:k]
    s = " ".join(map(say, best))
    return s + f", {say(played)}" if played and played not in best else s


@cache
def openings():
    """Lichess's opening names by position (EPD)."""
    out = {}
    with gzip.open(Path(__file__).with_name("openings.tsv.gz"), "rt") as f:
        for row in f:
            eco, name, moves = row.rstrip("\n").split("\t")
            b = chess.Board()
            for u in moves.split():
                b.push_uci(u)
            out[b.epd()] = f"{name} ({eco})"
    return out


def opening(board):
    """The name of the last named position of board's game."""
    b, name = chess.Board(), None
    for m in board.move_stack[:40]:
        b.push(m)
        name = openings().get(b.epd(), name)
    return name


def phase(board, n):
    count = lambda c: sum(
        v * len(board.pieces(p, c)) for p, v in VALUE.items() if p > 1
    )
    pieces = count(chess.WHITE) + count(chess.BLACK)
    if n < 20 and pieces > 50:
        return "opening"
    return "endgame" if pieces <= 26 else "middlegame"


NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
         chess.ROOK: "rook", chess.QUEEN: "queen"}  # fmt: skip


def deeds(board, move):
    """What a move does (castles, takes, promotes, checks), from the board before it."""
    out = []
    if board.is_castling(move):
        out.append(f"castles {'short' if board.is_kingside_castling(move) else 'long'}")
    if board.is_capture(move):
        taken = (
            chess.PAWN
            if board.is_en_passant(move)
            else board.piece_type_at(move.to_square)
        )
        out.append(f"takes a {NAMES[taken]}")
    if move.promotion:
        out.append(f"promotes to a {NAMES[move.promotion]}")
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        out.append("checkmate")
    elif after.is_check():
        out.append("check")
    return out


def material_diff(board, white):
    """Material, pawns as 1: white's side (if white) minus the other's."""
    return sum(
        v * (len(board.pieces(p, white)) - len(board.pieces(p, not white)))
        for p, v in VALUE.items()
    )


def traded(board):
    """The square of the last move if it took a piece worth as much as the mover: a piece
    that may be taken back there is a trade, not a piece left hanging."""
    m = board.peek()
    after = board.piece_type_at(m.to_square)
    board.pop()
    try:
        taken = (
            chess.PAWN if board.is_en_passant(m) else board.piece_type_at(m.to_square)
        )
    finally:
        board.push(m)
    return m.to_square if taken and VALUE[taken] >= VALUE.get(after, math.inf) else None


def hanging(board, color, skip=None):
    """color's pieces en prise: attacked, and undefended or attacked by a cheaper piece
    (but the one on skip)."""
    out = []
    for sq in chess.SquareSet(board.occupied_co[color]):
        p = board.piece_type_at(sq)
        attackers = board.attackers(not color, sq)
        if p == chess.KING or not attackers or sq == skip:
            continue
        cheapest = min(VALUE.get(board.piece_type_at(x), 100) for x in attackers)
        if cheapest < VALUE[p] or not board.attackers(color, sq):
            out.append("NBRQ"[p - 2] * (p > 1) + chess.square_name(sq))
    return out


class Recent:
    """The last n unprompted lines, across games and restarts: a JSON list in a file shared
    by processes. Best effort, on the chat thread only: a corrupt file reads as empty; a
    filesystem error, or another process holding the lock over a second, leaves the file
    alone for RETRY seconds, logged once. (A hung NFS server can still block the chat
    threads, never a game.)"""

    RETRY = 300

    def __init__(self, path, n):
        self.path, self.n, self.lock, self.retry = path, n, threading.Lock(), 0.0

    def load(self):
        """The lines; [] for a missing or corrupt file; OSError for anything else."""
        try:
            with open(self.path) as f:
                lines = json.load(f)
        except FileNotFoundError:
            return []
        except ValueError:
            return []  # rewritten with the next line
        return (
            [x for x in lines if isinstance(x, str)][-self.n :]
            if isinstance(lines, list)
            else []
        )

    def read(self):
        if not self.n or time.monotonic() < self.retry:
            return []
        try:
            return self.load()
        except OSError as e:
            return self.fail(e)

    def fail(self, e):
        if time.monotonic() >= self.retry:
            log.warning(
                "chat: recent lines %s: %s; without them for %d s",
                self.path,
                e,
                self.RETRY,
            )
        self.retry = time.monotonic() + self.RETRY
        return []

    def add(self, text):
        if not self.n or time.monotonic() < self.retry:
            return
        with self.lock:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path + ".lock", "a") as lock:
                    # another process holding the lock: wait a second at most
                    for _ in range(10):
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            time.sleep(0.1)
                    else:
                        raise TimeoutError("the lock is held")
                    lines = [*(x for x in self.load() if x != text), text][-self.n :]
                    with open(self.path + ".tmp", "w") as f:
                        json.dump(lines, f, indent=0)
                    os.replace(self.path + ".tmp", self.path)
            except OSError as e:  # TimeoutError and BlockingIOError are OSErrors too
                self.fail(e)


RECENTS, LOCK = {}, threading.Lock()


def recent(cfg):
    """The config's recent lines: recent_file, or recent-lines.json beside the ledger."""
    path = cfg.recent_file or os.path.join(
        os.path.dirname(os.path.expanduser(cfg.ledger)), "recent-lines.json"
    )
    with LOCK:
        return RECENTS.setdefault(
            (path, cfg.recent), Recent(os.path.expanduser(path), cfg.recent)
        )


def mmss(ms):
    s = max(ms, 0) // 1000
    return f"{s // 60}:{s % 60:02d}"


def material(board, white):
    d = sum(
        v * (len(board.pieces(p, white)) - len(board.pieces(p, not white)))
        for p, v in VALUE.items()
    )
    return (
        "even" if d == 0 else f"you are {'up' if d > 0 else 'down'} {abs(d)} (pawn = 1)"
    )


SAN = re.compile(
    r"\b(?:O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=?[QRBN])?)"
    r"(?![\w-])"
)


def played(text, moves):
    """Does text name only moves already played (UCI moves)? A bare square ("my c6 pawn")
    counts only if it is a pawn push legal now for either side ("push d4")."""
    b, seen = chess.Board(), set()
    for u in moves:
        seen.add(b.san(m := b.parse_uci(u)).rstrip("+#"))
        b.push(m)
    pushes = set()
    for color in (chess.WHITE, chess.BLACK):
        side = b.copy(stack=False)
        side.turn, side.ep_square = color, None
        pushes |= {chess.square_name(m.to_square) for m in side.pseudo_legal_moves
                   if side.piece_type_at(m.from_square) == chess.PAWN and not m.promotion
                   and not side.is_capture(m)}  # fmt: skip
    named = {
        x
        for x in SAN.findall(text)
        if not re.fullmatch(r"[a-h][1-8]", x) or x in pushes
    }
    return named <= seen


def clean(text):
    """The model's text as one line: no links, markup or wrapping quotes."""
    text = " ".join(re.sub(r"</?[a-z_]+>", "", text.replace("**", "")).split())
    text = text.strip("\"'\u201c\u201d{}")
    return "" if re.search(r"https?://|www\.", text) else text


def recase(text, board):
    """Moves the model wrote in lower case ("nxe6") restored ("Nxe6"): those of the game so
    far and the legal ones now."""
    sans, b = {board.san(m) for m in board.legal_moves}, chess.Board()
    for m in board.move_stack:
        sans.add(b.san(m))
        b.push(m)
    exact = {x.rstrip("+#") for x in sans}  # a pawn's bxc4 stays when Bxc4 is legal too
    known = {x.lower(): x for x in exact if x[0] in "KQRBNO"}
    return re.sub(r"\b[kqrbn][a-h]?[1-8]?x?[a-h][1-8]\b|\bo-o(?:-o)?\b",
                  lambda m: m[0] if m[0] in exact else known.get(m[0], m[0]), text)  # fmt: skip


def parts(text):
    """At most two chat lines of LIMIT characters, split between sentences if it can."""
    if len(text) <= LIMIT:
        return [text] if text else []
    ends = re.finditer(r"(?<!\d)[.!?] ", text[: LIMIT + 1])  # not after a move number
    cut = max((m.end() for m in ends), default=0)
    cut = cut or text[:LIMIT].rfind(" ") + 1 or LIMIT
    rest = text[cut:].strip()
    if len(rest) > LIMIT:
        rest = rest[: LIMIT - 1].rsplit(" ", 1)[0] + "\u2026"
    return [text[:cut].strip(), rest]


class Mock:
    """A stand-in model with no API calls: answers every turn."""

    def __call__(self, system, messages):
        turn = messages[-1]["content"]
        said = re.findall(r'Chat \(\w+\) [^:]+: "(.*)"', turn)
        reason = re.search(r"\(You are called for ([^:.]*)", turn)[1]
        text = f"(mock) re: {said[-1]}" if said else f"(mock) on {reason}"
        out = json.dumps({"speak": True, "text": text})
        return Reply([{"type": "text", "text": out}], True, text, 0.0, 0.0, 0.0)
