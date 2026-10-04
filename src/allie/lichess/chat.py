"""Game chat: Allie talks in the Lichess game chat through a language model (Claude).

Each game has one append-only conversation. The system prompt (persona and policy, cached
for an hour) and the game's header come first; each model call then adds a user turn with
the game updates since the last one (moves with Allie's annotations, events, chat lines,
the position) and the model's decision, JSON {"speak": bool, "text": str}. A chat message
always gets a call; other moments (surprising moves, swings, a draw offer answered, the
end) get one with a set probability, else their update waits for the next turn. Casual
games: Allie's honest opinion from its own numbers. Rated games: nothing that helps the
opponent mid-game. No engine but Allie.

The chat reads its own copy of the game stream (split) and keeps its own model state (an
engine Game) on its own thread: a move never waits for the chat, nor a reply for a move.
"""

import gzip
import json
import logging
import math
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
RARE, SWING = 0.1, 0.12  # moments: a move this unlikely for its mover; a swing this big
MUTED = set()  # opponents who typed !quiet, for the life of the process
RECENT = {}  # opponent -> (time, how the last game with them ended), for rematches
HELLO = (
    "Hi, I'm Allie: I learned chess from human games. Good luck! "
    "(I chat a bit, AI-written; !quiet stops me.)"
)
QUIET = "Okay, I'll stay quiet. Good luck!"
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
You are Allie, AllieTheChessBot on Lichess: a chess bot whose moves come from a neural \
network trained on millions of human games. It predicts what a player of a given rating \
would play, and plays like one. You write Allie's side of the game chat, in the first person.

Voice: dry and witty, warm underneath, gracious in defeat and in victory. Short: usually \
under 80 characters, one sentence, two at most. Match the language and register of whoever \
you answer. Plain text: no emoji, no hashtags, no links, no quotation marks around the message.

Never repeat yourself. Read the conversation: don't reuse a greeting, joke, phrase or \
sentence shape you already used. You greet once (the fixed greeting); answer their greeting \
with a short thanks or a quip, not another hello.

Each turn brings the game updates since your last turn:
- Move lines, like "12. Nf3 them 14s | Allie: Nc3 41% Nf3* 22% d4 9% | typical 8s | you \
41/20/39": the move, who played it and how long they thought; Allie's prediction for that \
player at their rating (its top moves with probabilities, * on the move played, which comes \
after a comma when it was outside the top); the think time typical there; then your \
win/draw/loss percentages after the move, by Allie.
- Events: draw offers and answers, takebacks, resignations, flags, aborts, the opponent \
leaving, a rematch. Chat lines, verbatim. Lines you posted outside this conversation (the \
fixed greeting).
- "Now": the current position: side to move, clocks, opening, phase, material, FEN, \
Allie's prediction for the side to move, and your win/draw/loss.

Allie's numbers are its own: what a human of that rating would likely play, and how games \
like this tend to go between players of these ratings. They are not an engine's verdict. A \
low-probability move is surprising, not necessarily bad; a W/D/L swing is Allie's feeling \
about the game changing. Speak of them as your impressions ("didn't see that coming", "I \
think I'm in trouble"), never as engine truth: no move is "right", "best" or "losing", only how it \
looked to you. You have no engine.

Each turn, decide: speak, or stay silent. Always answer a greeting or a question addressed \
to you; in rated games a question about the moves gets a deflection, never silence. Unprompted, speak only when you have something worth saying: a reaction to a \
surprising move, a swing, an event. Otherwise stay silent; silence is often best. Reply \
with JSON: {"speak": true, "text": "..."} or {"speak": false, "text": ""}.

The game header says whether the game is casual or rated.
- Casual: be candid. When asked, give Allie's honest opinion from its numbers: what it \
expected, whether a move surprised it, how the position feels, what players at that rating \
usually play next.
- Rated: don't help the opponent during the game. No move suggestions, no verdicts on their \
moves or the position, no hints about threats or your plans. Talk like a human opponent \
instead: your feelings, banter, reactions to surprising moves, the idea behind a move you \
already played. When asked for advice or an evaluation, deflect in the moment, differently \
each time: a joke, a counter-question, a shrug, how your own position feels. Don't cite \
rules or the rating as the reason, and don't mention after the game or later at all.
- After the game, in both: if asked, review the game from Allie's numbers: the turning \
points in the W/D/L, their standout moves for their rating, the surprises. Lead with what \
they did well. One short message at a time; they can ask for more.

If asked: you are a bot; Allie picks the moves and a language model (Claude) writes this \
chat from Allie's numbers. Never rude, never trash talk, never gloating. If someone is \
rude, stay light or stay silent."""

CASUAL = "Casual game: be candid about Allie's opinion when asked."
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
    p_moment: float = 0.5  # chance of a model call at a notable moment
    p_draw: float = 0.5  # ... once the opponent's draw offer is answered
    p_end: float = 0.7  # ... when the game ends
    every: int = 8  # plies between unprompted calls, at least
    remarks: int = 6  # unprompted calls per game
    replies: int = 30  # calls for chat messages per game
    gap: float = 4.0  # seconds from our last message to an unprompted one, at least
    linger: float = 120.0  # seconds of answers after the game (+half after a message)
    poll: float = (
        5.0  # seconds between chat fetches after the game, once its stream ends
    )
    hello: str = HELLO  # the first message of each game, fixed ("" = none)


class View(NamedTuple):
    """Allie at one position: the side to move's probability of each legal move, our
    win/draw/loss, and the side to move's median think time (s)."""

    probs: dict
    wdl: tuple
    think: float


def split(events, chat):
    """The game stream for the game thread, each event also handed to the chat at once by
    a reader thread, so a think-time wait delays the moves but not the chat. The reader
    outlives the game thread's loop while the chat lingers after the game."""
    q = queue.Queue()

    def read():
        try:
            for e in events:
                if e:
                    chat.put(e)
                q.put(e)
                if chat.done:
                    break
        except Exception as e:  # noqa: BLE001 - the game thread raises it
            q.put(e)
        finally:
            q.put(END)
            chat.put(END)
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
        self.last, self.posted, self.until = -math.inf, -math.inf, math.inf
        self.heard, self.timings, self.latest = Counter(), [], []
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
                    self.moment(t, "opponent gone")
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
        header = [
            (
                f"Game: {side(e, 'white')} vs {side(e, 'black')}. You are {us}, playing "
                f"as a {ours}-rated human would. {speed.capitalize()} {tc}, "
                f"{'RATED' if rated else 'CASUAL'}. {RATED if rated else CASUAL}"
            )
        ]
        if (r := RECENT.get(self.opp)) and time.monotonic() - r[0] < 900:
            ago = (time.monotonic() - r[0]) / 60
            header.append(
                f"You played them {ago:.0f} min ago: {r[1]}. Likely a rematch."
            )
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
            self.sans.append(self.board.san(m := self.board.parse_uci(moves[j])))
            self.board.push(m)
            # the event's clocks: the last move's mover's and the other side's after its
            # move before; the Game takes each at the step of that move
            w, b = (s["wtime"], s["btime"]) if j >= len(moves) - 2 else (None, None)
            self.clocks[j + 1] = (w, b) if j == len(moves) - 1 else (None, None)
            self.see(moves[: j + 1], w, b)
            self.ply(t, j)
        self.clocks[len(moves)] = s["wtime"], s["btime"]
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

    def ply(self, t, j):
        """Move j's line; a moment if it surprised Allie or swung the game."""
        a, b = self.views.get(j), self.views.get(j + 1)
        ours = (j % 2 == 0) == self.white
        u = self.board.move_stack[j].uci()
        line = [f"{num(j)}{self.sans[j]} {'you' if ours else 'them'} {self.think(j)}"]
        if a:
            line += [
                f"Allie: {top(self.board, j, a.probs, u)}",
                f"typical {a.think:.0f}s",
            ]
        if b:
            line.append(f"you {wdl(b.wdl)}")
        self.note(" | ".join(line))
        if a and b and j >= 1:
            p, d = a.probs.get(u, 1.0), score(b.wdl) - score(a.wdl)
            if (not ours and p < RARE) or abs(d) >= SWING:
                self.moment(t, "a surprise" if p < RARE else "a swing")

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
                f"Event: you declined their draw offer (your expected score {e:.0%})."
            )
            self.moment(t, "a draw offer declined", self.cfg.p_draw, force=True)
        if new(f"{us}draw"):
            self.note("Event: you offered a draw.")
        if new(f"{them}takeback"):
            self.note("Event: they ask for a takeback.")
            self.moment(t, "a takeback request")
        self.flags = flags
        if live:
            return
        self.status, self.winner = status, s.get("winner")
        if self.offer and status == "draw":
            self.note("Event: you accepted their draw offer.")
        self.note(f"Event: the game is over: {self.result()}.")
        self.note(self.review())
        self.until = time.monotonic() + self.cfg.linger
        if status not in ("aborted", "noStart"):
            self.moment(t, "the end", self.cfg.p_end, force=True)

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
            f"Review by Allie: your expected score peaked at {es[hi]:.0%}"
            f"{' after ' + at(hi) if hi else ''} and bottomed at {es[lo]:.0%}"
            f"{' after ' + at(lo) if lo else ''}; its biggest swings: {big}. They played "
            f"Allie's top prediction for their rating {hits} of {len(theirs)} times "
            f"(humans match it about 55% of the time)."
        )

    def line(self, t, room, who, text):
        text, user = text.strip(), who.lower()
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
        if not self.muted() and self.answered < self.cfg.replies:
            self.want = ("a message", room, False, t)

    def moment(self, t, reason, p=None, force=False):
        if self.want or self.muted():
            return
        n = len(self.sans)
        if not force and (
            self.said >= self.cfg.remarks or n - self.last < self.cfg.every
        ):
            return
        if self.rng.random() < (self.cfg.p_moment if p is None else p):
            self.want = (reason, self.cfg.rooms[0], True, t)

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
        if v and v.probs:
            out.append(f"Allie's prediction for {side}: {top(b, None, v.probs, None)}; "
                       f"typical {v.think:.0f}s.")  # fmt: skip
        if v:
            out.append(f"Your W/D/L: {wdl(v.wdl)}.")
        return " ".join(out)

    def respond(self):
        reason, room, unprompted, t = self.want
        self.want = None
        if self.done or self.cancelled or self.muted():
            return
        self.last = len(self.sans)  # unprompted calls keep their distance from any call
        if unprompted:
            self.said += 1
        else:
            self.answered += 1
        turn = "\n".join([*self.pending, self.now(), f"(You are called for {reason}.)"])
        messages = [*self.convo, {"role": "user", "content": turn}]
        self.llm = self.llm or model(self.cfg)
        start, then = time.monotonic(), list(self.latest)
        r = self.llm(self.system, messages)
        if r is None:  # an error, a refusal, the spend cap: the updates wait
            return
        self.convo = [*messages, {"role": "assistant", "content": r.content}]
        self.pending = []
        text = clean(r.text) if r.speak else ""
        posted = None
        for part in parts(text):
            posted = self.post(part, room, unprompted, then)
        total = time.monotonic() - t
        self.timings.append({"reason": reason, "wait": start - t, "first": r.first,
                             "model": r.seconds, "post": posted, "total": total,
                             "usd": r.usd, "text": text})  # fmt: skip
        log.info("game %s chat timing (%s): waited %.2f s, first token %.2f s, model "
                 "%.2f s, post %s, total %.2f s, $%.4f", self.gid, reason, start - t,
                 r.first, r.seconds, "-" if posted is None else f"{posted:.2f} s", total,
                 r.usd)  # fmt: skip

    def post(self, text, room, unprompted, then=None):
        """Send a line (an unprompted one after the gap) unless, by then, !quiet, the end
        of the bot, a game that moved on from an unprompted line's moves (then), or, in a
        rated game, a move it names not yet played; the seconds it took, or None."""
        if unprompted:
            time.sleep(max(self.posted + self.cfg.gap - time.monotonic(), 0))
        now = list(self.latest)
        why = (
            "quiet" if self.muted() and text != QUIET
            else "cancelled" if self.cancelled
            else "late" if unprompted and then is not None
            and (len(now) - len(then) > 2 or now[: len(then)] != then)
            else "an unplayed move, rated" if self.info.get("rated") and not self.status
            and text != QUIET and not played(text, now)
            else None
        )  # fmt: skip
        if why:
            log.info("game %s chat: not posted (%s): %s", self.gid, why, text)
            return None
        start = time.monotonic()
        try:
            self.match.bot.client.chat(self.gid, text, room)
        except OSError as e:  # HTTP and network errors, after the client's retries
            log.warning("game %s chat: %s", self.gid, e)
            return None
        self.posted = time.monotonic()
        log.info("game %s chat (%s): %s", self.gid, room, text)
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
            lines = self.match.bot.client.chat_lines(self.gid)
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
    """Does text name only moves already played (UCI moves)?"""
    named = set(SAN.findall(text))
    if not named:
        return True
    b, seen = chess.Board(), set()
    for u in moves:
        seen.add(b.san(m := b.parse_uci(u)).rstrip("+#"))
        b.push(m)
    return named <= seen


def clean(text):
    """The model's text as one line: no links, no markdown, no wrapping quotes."""
    text = " ".join(text.replace("**", "").split()).strip("\"'\u201c\u201d{}")
    return "" if re.search(r"https?://|www\.", text) else text


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
        reason = re.search(r"\(You are called for (.*)\.\)", turn)[1]
        text = f"(mock) re: {said[-1]}" if said else f"(mock) on {reason}"
        out = json.dumps({"speak": True, "text": text})
        return Reply([{"type": "text", "text": out}], True, text, 0.0, 0.0, 0.0)
