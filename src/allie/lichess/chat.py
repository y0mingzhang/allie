"""Game chat: short remarks a language model (Claude) writes from Allie's own numbers, at
notable moments only, and answers to what the opponent writes. Off unless [chat] enabled.

Fair play: during the game the model sees only what Allie itself thinks, minus anything that
would help the opponent (their mistakes, Allie's good prospects). Stockfish runs only after the
game. A move never waits for the chat: each game's chat has its own thread.
"""

import functools
import logging
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import NamedTuple

import chess
import torch

from .engine import WDL
from .tokens import MOVE_ID

log = logging.getLogger(__name__)

RARE = 0.08  # a move this unlikely at the mover's rating is a surprise
GOOD = 0.12  # a move that shifts the expected score this much is strong (or a mistake)
DROP = 0.25  # an expected score this far below its high since the last remark: pressure
LIMIT = 140  # Lichess's longest chat line
MUTED = set()  # opponents who typed !quiet, for the life of the process
HELLO = (
    "Hi, I'm Allie, a bot that learned chess from human games. Good luck! "
    "I chat a little (AI-written); type !quiet to stop me."
)
QUIET = "Okay, I'll stay quiet. Good luck!"
ENDS = {
    "mate": "checkmate",
    "resign": "resignation",
    "stalemate": "stalemate",
    "timeout": "the other player leaving",
    "outoftime": "time",
    "draw": "agreement or rule",
    "insufficientMaterialClaim": "insufficient material",
}
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

SYSTEM = """\
You are Allie (Lichess account AllieTheChessBot), a chess bot: a neural network that learned \
from millions of human games to predict what a player of a given rating would play, and plays \
like one. You write Allie's messages in the game chat, in the first person.

Rules:
- Reply with one chat message of at most 120 characters (after the game: at most two lines of \
up to 120 characters each). Plain text: no markdown, no emoji, no quotation marks around it, \
no links.
- Be warm, modest and good-humoured. Never rude, sarcastic, boastful or condescending; no \
trash talk, no gloating.
- Use only the facts given. Don't invent moves, tactics, threats, evaluations or opening \
names. Write moves as the facts write them.
- While the game is on, never help either side: don't suggest moves or plans, don't point out \
mistakes or threats, don't reveal what you intend. If asked for advice or an evaluation, offer \
to look at it after the game.
- After the game you may discuss the key moments in the facts, including Stockfish's.
- If asked: you are a bot. Allie picks the moves; a language model (Claude) writes this chat \
from Allie's numbers.
- Answer in the language of the message you answer.
- If there is nothing good to say, or the message needs no answer, reply with exactly SKIP.

The facts you get:
- A move's probability is how often Allie expects a player of that rating to choose it: a low \
one means a surprising move, not a bad one.
- Allie's win/draw/loss estimate is from your side, for players of these ratings. It is a \
human-level hunch, not an engine verdict: speak of it as a feeling ("I think I'm in trouble").
- Think times come from the clocks. Stockfish facts appear only after the game.
- The facts may leave things out on purpose during the game; don't guess at what is missing.

Your voice, in examples (don't copy them):
- After a rare, strong move: "Oh, I didn't see that coming. Nice move!"
- When the game turns against you: "You're making this hard for me. Well played so far."
- To "hi, gl": "Hi! Good luck to you too, have fun."
- To "what should I play here?": "No hints mid-game, sorry! Happy to look at it with you after."
- To "are you a bot?": "I am! Allie picks my moves, and a language model writes my chat from \
Allie's numbers."
- After a loss: "Thanks for the game, you earned that one! 23. Qh5 swung it: Stockfish liked \
23. Nf3 better."
- After a win: "gg, thanks! That was close for a while: I thought you had me after 18...Rxe4."\
"""


@dataclass
class Chat:
    enabled: bool = False
    rooms: tuple = ("player",)  # remarks go here; "spectator" also answers spectators
    llm: str = "claude"  # "mock": canned text, no API calls (tests, dry runs)
    model: str = "claude-sonnet-5-5"
    effort: str = "low"  # output_config.effort ("" = the model's default)
    thinking: str = "between_tools"  # Sonnet 5.5's lowest setting ("" = the default)
    max_tokens: int = 200
    timeout: float = 3.0  # seconds per model call; on a timeout the message is dropped
    key_file: str = "~/.config/allie/anthropic_key"  # if ANTHROPIC_API_KEY is unset
    every: int = 10  # plies between two unprompted remarks, at least
    remarks: int = 4  # unprompted remarks per game (the hello and post-game aside)
    replies: int = 8  # answers to chat messages per game
    gap: float = 4.0  # seconds between two of our messages, at least
    hello: str = HELLO  # the first message of each game ("" = none)
    stockfish: str = ""  # binary for the post-game message's analysis; never mid-game


class Job(NamedTuple):
    kind: str  # hello, quiet: text is the message; remark, reply, end: a prompt
    ply: int
    rooms: tuple
    text: object  # a string, or for "end" a function that returns the prompt
    lines: int = 1


def safe(f):
    """The game thread's entry points: the chat is best-effort and never raises into a game."""

    @functools.wraps(f)
    def g(self, *args):
        try:
            f(self, *args)
        except Exception:
            log.exception("game %s chat", self.match.gid)

    return g


class Chatter:
    """One game's chat. The game thread calls seen() after each gameFull or gameState and
    heard() for each chatLine; both only read the game and queue work for the chat thread."""

    def __init__(self, match, llm=None):
        self.match, self.cfg, self.llm = match, match.bot.config.chat, llm
        self.views, self.clocks = {}, {}  # by ply: Allie's view, both clocks (ms)
        self.lines, self.info = [], {}  # the chat so far; the gameFull event
        self.ref, self.last, self.said, self.answered = None, -math.inf, 0, 0
        self.greeted = self.done = False
        self.posted = 0.0
        self.jobs = queue.Queue()
        name = f"chat-{match.gid}"
        threading.Thread(target=self.serve, daemon=True, name=name).start()

    # the game thread

    @safe
    def seen(self, event):
        m, g = self.match, self.match.game
        if self.done:
            return
        if event["type"] == "gameFull":
            self.info = event
        if g is None:  # aborted at the start
            return self.close() if m.over else None
        s = event.get("state", event)
        moves = s["moves"].split()  # at the end, the game may lack the last move
        n = len(moves)
        for k in [k for k in self.views if k > n]:  # a takeback
            del self.views[k]
        if g.logits is not None and len(g.used) == len(g.tokens) and len(g.moves) == n:
            self.views.setdefault(n, view(g, m.white))
        self.clocks[n] = s["wtime"], s["btime"]
        if s["status"] not in ("created", "started"):
            if s["status"] in ENDS and n >= 2 and not self.muted():
                end = functools.partial(
                    self.ending, moves, s["status"], s.get("winner")
                )
                self.jobs.put(Job("end", n, self.cfg.rooms, end, 2))
            return self.close()
        if not self.greeted:
            self.greeted = True
            if self.cfg.hello and n < 2 and not self.muted():
                self.jobs.put(Job("hello", n, ("player",), self.cfg.hello))
        elif n >= 2 and (n % 2 == 1) == m.white:  # we just moved
            self.remark(moves)

    @safe
    def heard(self, event):
        who, room = event.get("username", ""), event.get("room")
        text, them = event.get("text", "").strip(), self.match.opponent
        if not text or who.lower() in (self.match.bot.me, "lichess"):
            return
        if room == "player" and who.lower() != them:
            return
        if room == "spectator" and "spectator" not in self.cfg.rooms:
            return
        self.lines.append((who, text))
        if who.lower() == them and text.lower().startswith("!quiet"):
            if not self.muted():
                MUTED.add(them)
                self.jobs.put(Job("quiet", 0, (room,), QUIET))
            return
        g = self.match.game
        if self.done or g is None or self.muted() or self.answered >= self.cfg.replies:
            return
        self.answered += 1
        moves = list(g.moves)
        lines = self.facts(moves, private=True) + ["Recent chat:"]
        lines += [f"  {w}: {t}" for w, t in self.lines[-8:]]
        whom = "your opponent" if who.lower() == them else f"the spectator {who}"
        lines.append(f"Answer {whom}'s last message, or SKIP.")
        self.jobs.put(Job("reply", len(moves), (room,), "\n".join(lines)))

    def remark(self, moves):
        """After our move at ply n - 1: a word on their move at n - 2, or on the game's turn."""
        n = len(moves)
        a, b = self.views.get(n - 2), self.views.get(n - 1)
        if not (a and b):
            return
        e0, e1 = score(a[1]), score(b[1])
        self.ref = max(e0, e0 if self.ref is None else self.ref)
        if a[0].get(moves[n - 2], 1.0) < RARE and e1 <= e0 - GOOD:
            task = "Their last move was rare and strong: compliment it, or SKIP."
        elif e1 <= self.ref - DROP and e1 < 0.45:
            task = "Your chances dropped: say gracefully that they're pressing you, or SKIP."
        else:
            return
        if self.said >= self.cfg.remarks or n - self.last < self.cfg.every:
            return
        self.said, self.last, self.ref = self.said + 1, n, e1
        prompt = "\n".join(self.facts(moves, private=True) + [task])
        self.jobs.put(Job("remark", n, self.cfg.rooms, prompt))

    def close(self):
        self.done = True
        self.jobs.put(None)

    def muted(self):
        return self.match.opponent in MUTED

    # facts for the model

    def facts(self, moves, private):
        """The game so far from our side. private (during the game): without what would help
        the opponent: their mistakes and our good prospects."""
        m, g, views, n = self.match, self.match.game, self.views, len(moves)
        us, them = ("white", "black") if m.white else ("black", "white")
        opp = self.info.get(them) or {}
        board = board_at(moves, n)
        name, rating = opp.get("name", "anonymous"), opp.get("rating", "?")
        out = [
            f"You are Allie, playing {us} as a {g.elo[us == 'black']}-rated player would.",
            f"Opponent: {name} ({them}, rated {rating}). {kind(self.info)}.",
            f"Moves: {movetext(moves)}",
            f"FEN: {board.fen()}",
            f"Material: {material(board, m.white)}",
        ]
        if n in self.clocks:
            w, b = self.clocks[n]
            out.append(
                f"Clocks: you {mmss(w if m.white else b)}, them {mmss(b if m.white else w)}"
            )
        k = n - ((n % 2 == 0) != m.white)  # the last position with us to move
        if k in views and (not private or score(views[k][1]) <= 0.55):
            w, d, l = views[k][1]
            out.append(
                f"Allie's estimate for you when you were last to move: win {w:.0%}, "
                f"draw {d:.0%}, loss {l:.0%}."
            )
        j = k - 1  # their last move
        if j >= 0 and j in views and k in views:
            probs, d = views[j][0], score(views[k][1]) - score(views[j][1])
            if not private or d < 0.05:
                out.append(
                    f"Their last move {san(moves, j)}: Allie gave it "
                    f"{pct(probs.get(moves[j], 1.0))} for a {rating} player "
                    f"({top(moves, j, probs)}); they thought {self.think(j)}. {effect(d)}"
                )
        if k < n and k in views:
            out.append(
                f"Your last move {san(moves, k)}: Allie gave it "
                f"{pct(views[k][0].get(moves[k], 1.0))}; you thought {self.think(k)}."
            )
        return out

    def ending(self, moves, status, winner):
        """The post-game prompt: the result, Allie's view of the whole game, Stockfish's."""
        m, views = self.match, self.views
        us = "white" if m.white else "black"
        out = self.facts(moves, private=False)
        result = "a draw" if not winner else "you won" if winner == us else "you lost"
        out.append(f"The game is over: {result} by {ENDS[status]}.")
        es = {k: score(v[1]) for k, v in views.items() if k > 0}
        if es:
            hi, lo = max(es, key=es.get), min(es, key=es.get)
            out.append(
                f"Allie's expected score for you peaked at {es[hi]:.0%} after "
                f"{san(moves, hi - 1)} and bottomed at {es[lo]:.0%} after {san(moves, lo - 1)}."
            )
        theirs = [j for j in views if j < len(moves) and (j % 2 == 0) != m.white]
        if theirs:
            p = lambda j: views[j][0].get(moves[j], 1.0)
            hits = sum(
                max(views[j][0], key=views[j][0].get) == moves[j] for j in theirs
            )
            j = min(theirs, key=p)
            out.append(
                f"They played Allie's top prediction {hits} of {len(theirs)} times (Allie "
                f"matches about 55-60% of human moves). Their most surprising move: "
                f"{san(moves, j)} ({pct(p(j))})."
            )
        if self.cfg.stockfish:
            try:
                out += stockfish(self.cfg.stockfish, moves, m.white)
            except Exception as e:  # noqa: BLE001 - analysis is optional
                log.warning("game %s: stockfish: %s", m.gid, e)
        out.append(
            "The game just ended. Write a friendly post-game message: thank them, and mention "
            "one interesting fact from above. At most two short lines."
        )
        return "\n".join(out)

    def think(self, j):
        """Move j's think time, from its mover's clock before and after it."""
        a, b = self.clocks.get(j), self.clocks.get(j + 1)
        if a is None or b is None:
            return "an unknown time"
        inc = (self.info.get("clock") or {}).get("increment", 0)
        return f"{max(a[j % 2] - b[j % 2] + inc, 0) / 1000:.0f} s"

    # the chat thread

    def serve(self):
        while True:
            jobs = [self.jobs.get()]
            while not self.jobs.empty():
                jobs.append(self.jobs.get_nowait())
            replies = [i for i, j in enumerate(jobs) if j and j.kind == "reply"]
            for i, j in enumerate(jobs):
                if j is None:
                    return
                if j.kind == "reply" and i != replies[-1]:  # answer a burst's last line
                    continue
                try:
                    self.run(j)
                except Exception:
                    log.exception("game %s chat %s", self.match.gid, j.kind)

    def run(self, j):
        if j.kind in ("hello", "quiet"):
            lines = [j.text]
        elif self.muted():
            return
        else:
            self.llm = self.llm or model(self.cfg)
            prompt = j.text() if callable(j.text) else j.text
            lines = clean(self.llm(SYSTEM, prompt), j.lines)
            if j.kind == "remark" and len(self.match.game.moves) - j.ply > 2:
                log.info(
                    "game %s chat: the remark at ply %d came late",
                    self.match.gid,
                    j.ply,
                )
                return
        for line in lines:
            for room in j.rooms:
                self.post(line, room)

    def post(self, text, room):
        time.sleep(max(self.posted + self.cfg.gap - time.monotonic(), 0))
        self.match.bot.client.chat(self.match.gid, text, room)
        self.posted = time.monotonic()
        self.lines.append(("you", text))
        log.info("game %s chat (%s): %s", self.match.gid, room, text)


def view(g, white):
    """Allie's view of the game's last position: each legal move's probability, and our
    (win, draw, loss)."""
    z = g.logits.double()
    legal = [m.uci() for m in g.board.legal_moves]
    p = torch.softmax(z[[MOVE_ID[u] for u in legal]], 0).tolist() if legal else []
    wdl = torch.softmax(z[WDL], 0).tolist()
    if g.board.turn != white:  # the head speaks for the side to move
        wdl.reverse()
    return dict(zip(legal, p)), wdl


def score(wdl):
    return wdl[0] + wdl[1] / 2


def pct(p):
    return "under 1%" if p < 0.005 else f"{p:.0%}"


def board_at(moves, j):
    b = chess.Board()
    for u in moves[:j]:
        b.push_uci(u)
    return b


def san(moves, j, k=None):
    """Moves j..k-1 as "12. Nf3 Nf6 13. ..." (one move by default)."""
    line = [chess.Move.from_uci(u) for u in moves[j : j + 1 if k is None else k]]
    return board_at(moves, j).variation_san(line)


def movetext(moves, last=30):
    j = max(len(moves) - last, 0)
    return ("... " if j else "") + san(moves, j, len(moves)) if moves else "(none yet)"


def top(moves, j, probs):
    best = max(probs, key=probs.get)
    if best == moves[j]:
        return "its top prediction"
    b = board_at(moves, j)
    return f"it expected {b.san(chess.Move.from_uci(best))} most, {pct(probs[best])}"


def effect(d):
    if abs(d) < 0.05:
        return "Allie thinks it changed little."
    who = "you" if d > 0 else "them"
    return f"Allie thinks it helped {who}: your expected score moved {d:+.0%}."


def kind(info):
    c = info.get("clock")
    tc = f"{c['initial'] / 60000:g}+{c['increment'] // 1000}" if c else "no clock"
    rated = "rated" if info.get("rated") else "casual"
    return f"{info.get('speed', 'a').capitalize()} game, {tc}, {rated}"


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


def clean(text, lines=1):
    """The model's reply as at most `lines` chat lines; none for SKIP or a link."""
    text = text.strip()
    if (
        not text
        or text.upper().startswith("SKIP")
        or re.search(r"https?://|www\.", text)
    ):
        return []
    out = []
    for s in text.splitlines():
        s = " ".join(s.replace("**", "").split()).strip("\"'“”")
        if s:
            out.append(s if len(s) <= LIMIT else s[: LIMIT - 1].rsplit(" ", 1)[0] + "…")
    return out[:lines]


def stockfish(path, moves, white, nodes=30_000):
    """Each side's costliest move by Stockfish's expected score, and its best move there."""
    import chess.engine

    with chess.engine.SimpleEngine.popen_uci(path) as sf:
        sf.configure({"Threads": 1, "Hash": 16})
        board, infos = chess.Board(), []
        for u in [None, *moves]:
            if u:
                board.push_uci(u)
            infos.append(sf.analyse(board, chess.engine.Limit(nodes=nodes)))
    e = [i["score"].white().wdl(model="lichess").expectation() for i in infos]
    drop = lambda j: (e[j] - e[j + 1]) * (1 if j % 2 == 0 else -1)  # for move j's mover
    out = []
    for who, mover in (("your", white), ("their", not white)):
        js = [j for j in range(len(moves)) if (j % 2 == 0) == mover]
        j = max(js, key=drop, default=None)
        if j is None or drop(j) < 0.1:
            out.append(f"Stockfish: no clear mistake on {who} side.")
            continue
        pv = infos[j].get("pv", [])[:3]
        best = f"; it preferred {board_at(moves, j).variation_san(pv)}" if pv else ""
        out.append(
            f"Stockfish: {who} costliest move was {san(moves, j)} (eval {cp(infos[j])} "
            f"-> {cp(infos[j + 1])}, white's view){best}."
        )
    return out


def cp(info):
    s = info["score"].white()
    return f"mate in {s.mate()}" if s.is_mate() else f"{s.score() / 100:+.1f}"


def mock(system, prompt):
    return "(mock) " + prompt.rsplit("\n", 1)[-1]


MODELS, LOCK = {}, threading.Lock()


def model(cfg):
    """The model of a [chat] config, shared by the games: a function (system, prompt) -> text
    that never raises."""
    with LOCK:
        if repr(cfg) not in MODELS:
            MODELS[repr(cfg)] = make(cfg)
        return MODELS[repr(cfg)]


def make(cfg):
    if cfg.llm == "mock":
        return mock
    silent = lambda system, prompt: ""
    if not (k := key(cfg.key_file)):
        log.error("chat: no ANTHROPIC_API_KEY or %s: fixed lines only", cfg.key_file)
        return silent
    try:
        return Claude(cfg, k)
    except ImportError:
        log.error("chat: no anthropic package (allie[chat]): fixed lines only")
        return silent


def key(path):
    """The Anthropic API key: ANTHROPIC_API_KEY, else the first line of path (chmod 600)."""
    if k := os.environ.get("ANTHROPIC_API_KEY"):
        return k
    path = os.path.expanduser(path)
    try:
        with open(path) as f:
            k = f.readline().strip()
    except OSError:
        return None
    if os.stat(path).st_mode & 0o077:
        log.warning("chat: %s is readable by others; chmod 600 it", path)
    return k or None


class Claude:
    """Claude through the Anthropic SDK. Errors drop one message: a bad key, model or request
    turns the model off, a rate limit pauses it for a minute. Keeps token totals in usage."""

    def __init__(self, cfg, key):
        import anthropic

        self.sdk, self.cfg, self.until = anthropic, cfg, 0.0
        self.client = anthropic.Anthropic(
            api_key=key, timeout=cfg.timeout, max_retries=0
        )
        self.usage = {
            k: 0 for k in ("calls", "input", "cached", "written", "output", "seconds")
        }
        self.extra = {}
        if cfg.effort:
            self.extra["output_config"] = {"effort": cfg.effort}
        if cfg.thinking:
            self.extra["thinking"] = {"type": cfg.thinking}

    def __call__(self, system, prompt):
        a, start = self.sdk, time.monotonic()
        if start < self.until:
            return ""
        try:
            r = self.client.beta.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **self.extra,
            )
        except (a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError,
                a.BadRequestError) as e:  # fmt: skip
            log.error("chat model off: %s", e)
            self.until = math.inf
            return ""
        except a.RateLimitError:
            log.warning("chat model rate limited; pausing a minute")
            self.until = start + 60
            return ""
        except a.APIError as e:  # timeouts, connection and server errors
            log.warning("chat model: %s", e)
            return ""
        u, t = r.usage, time.monotonic() - start
        cached, written = (
            u.cache_read_input_tokens or 0,
            u.cache_creation_input_tokens or 0,
        )
        for k, v in zip(
            self.usage, (1, u.input_tokens, cached, written, u.output_tokens, t)
        ):
            self.usage[k] += v
        log.info("chat model %s: %d in + %d cached + %d written, %d out, %.2f s, %s", r.model,
                 u.input_tokens, cached, written, u.output_tokens, t, r.stop_reason)  # fmt: skip
        if r.stop_reason in ("refusal", "max_tokens"):
            return ""
        return "".join(b.text for b in r.content if b.type == "text")
