"""The bot: accept challenges by the config's rules, play every game in its own thread."""

import http.client
import json
import logging
import queue
import random
import threading
import time
import urllib.error
from dataclasses import replace

from .chat import END, Chatter
from .engine import Decision, Game

log = logging.getLogger(__name__)
NETWORK = (OSError, urllib.error.URLError, json.JSONDecodeError, http.client.HTTPException)


def screen(c, rules, busy):
    """None to accept the challenge, else Lichess's decline reason."""
    tc = c.get("timeControl", {})
    title = (c.get("challenger") or {}).get("title")
    if c["variant"]["key"] != "standard":
        return "standard"
    if tc.get("type") != "clock" or c["speed"] not in rules.speeds:
        return "timeControl"
    if tc["limit"] < rules.min_base or tc["increment"] < rules.min_increment:
        return "tooFast"
    if tc["limit"] > rules.max_base or tc["increment"] > rules.max_increment:
        return "tooSlow"
    if c["rated"] and not rules.rated:
        return "casual"
    if not c["rated"] and not rules.casual:
        return "rated"
    if title == "BOT" and not rules.bots:
        return "noBot"
    if title != "BOT" and not rules.humans:
        return "onlyBot"
    if busy:
        return "later"
    return None


ERRORS = 5  # unexpected errors in one game, with no move of the bot's between, before it resigns
LATE = 3.0  # seconds within which Lichess's state shows the bot's move, draw acceptance or resignation


def patience(abort, base):
    """Seconds to wait for the opponent's first move: `abort` up to a 5-minute base, 3x from 15 minutes (live,
    opponents of slow games were often still there, chatting, when the bot aborted at 30 s)."""
    return abort * min(max((base or 0) / 300, 1.0), 3.0)


class Feed:
    """A game stream read on its own thread: the game thread takes its events with a timeout, so a think-time
    wait never stops it reading, and the chat gets each event at once. The reader outlives the game thread's
    loop while the chat lingers after the game, unless abandoned (the game thread resumed on a new stream:
    this one would feed the chat twice). A chat that fails is left behind, logged once."""

    def __init__(self, events, chat=None, name="game"):
        self.q, self.chat, self.abandoned = queue.Queue(), chat, threading.Event()
        threading.Thread(target=self.read, args=(events,), daemon=True, name=f"read-{name}").start()

    def feed(self, e):
        if self.chat is not None:
            try:
                self.chat.put(e)
            except Exception:
                log.exception("game %s chat: failed; the game goes on without it", self.chat.gid)
                self.chat = None

    def read(self, events):
        try:
            for e in events:
                if self.abandoned.is_set():
                    break
                if e:
                    self.feed(e)
                self.q.put(e)
                if self.chat is not None and self.chat.done:
                    break
        except Exception as e:  # noqa: BLE001 - the game thread raises it
            self.q.put(e)
        finally:
            self.q.put(END)
            if not self.abandoned.is_set():  # an abandoned stream's end is not the game's
                self.feed(END)
            if close := getattr(events, "close", None):
                close()

    def get(self, timeout=None):
        """The next event: None for a keepalive or after `timeout` seconds, END once the stream ended; the
        stream's error is raised."""
        try:
            e = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(e, Exception):
            raise e
        return e

    def empty(self):
        return self.q.empty()

    def abandon(self):
        self.abandoned.set()


class Bot:
    def __init__(self, config, client, engine, search=None):
        self.config, self.client, self.engine, self.search = config, client, engine, search
        self.games, self.finished, self.lock = {}, {}, threading.Lock()
        self.pending = {}  # accepted challenge id -> (challenger, time accepted)
        self.threads = []
        self.me = None
        self.stopped, self.draining = threading.Event(), False

    def run(self):
        """Serve the event stream until stop(), reconnecting with backoff."""
        self.me = self.client.account()["id"]
        log.info("playing as %s", self.me)
        backoff = 1
        while not self.stopped.is_set():
            try:
                for event in self.client.stream("/api/stream/event"):
                    if event:
                        self.on_event(event)
                        backoff = 1
                    if self.draining:
                        with self.lock:
                            if not self.opponents():
                                self.stop()
                    if self.stopped.is_set():
                        break
            except NETWORK as e:
                log.warning("event stream: %s; reconnecting in %d s", e, backoff)
                self.stopped.wait(backoff)
                backoff = min(2 * backoff, 60)
            except Exception:  # a bug or a local failure: logged, and the bot goes on
                log.exception("event stream: unexpected error; reconnecting in %d s", backoff)
                self.stopped.wait(backoff)
                backoff = min(2 * backoff, 60)

    def stop(self):
        self.stopped.set()

    def join(self):
        """After stop(): wait for the game threads."""
        while alive := [t for t in self.threads if t.is_alive()]:
            for t in alive:
                t.join()

    def drain(self):
        """Decline new challenges; run() returns once the current games end."""
        log.info("draining: no new games")
        self.draining = True

    def opponents(self):
        """Under the lock: the opponents of current games (not yet reported finished) and of accepted
        challenges whose game has not started yet (a challenge's game has its id; reservations last a minute)."""
        now = time.monotonic()
        self.pending = {c: v for c, v in self.pending.items() if now - v[1] < 60}
        return [m.opponent for m in self.games.values() if not m.ended] + [u for u, _ in self.pending.values()]

    def on_event(self, event):
        match event["type"]:
            case "challenge":
                c = event["challenge"]
                who = c["challenger"]["id"]
                if who == self.me:
                    return
                rules = self.config.challenge
                with self.lock:
                    them = self.opponents()
                    busy = self.draining or len(them) >= self.config.max_games
                    reason = screen(c, rules, busy or them.count(who) >= rules.per_user)
                    if not reason:
                        self.pending[c["id"]] = (who, time.monotonic())
                log.info("challenge %s from %s: %s", c["id"], who, reason or "accept")
                try:
                    if reason:
                        self.client.decline(c["id"], reason)
                    else:
                        self.client.accept(c["id"])
                except urllib.error.HTTPError as e:  # withdrawn meanwhile
                    log.warning("challenge %s: %s", c["id"], e)
                    with self.lock:
                        self.pending.pop(c["id"], None)
            case "challengeCanceled":
                with self.lock:
                    self.pending.pop(event["challenge"]["id"], None)
            case "gameStart":
                g = event["game"]
                gid = g.get("gameId") or g["id"]
                with self.lock:
                    self.pending.pop(gid, None)
                    if gid in self.games or self.stopped.is_set():
                        return
                    match = self.games[gid] = Match(self, gid)
                    match.opponent = (g.get("opponent") or {}).get("id")
                    t = threading.Thread(target=self.play, args=(match,), name=gid, daemon=True)
                    self.threads.append(t)
                t.start()
            case "gameFinish":  # ahead of the game's own stream: a rematch challenge right after is not "later"
                g = event["game"]
                with self.lock:
                    if m := self.games.get(g.get("gameId") or g["id"]):
                        m.ended = True

    def play(self, match):
        """Serve one game's stream until it ends. Network errors reconnect; any other error
        is logged and the game resumes from a fresh stream. After ERRORS of them with no move of
        the bot's between (a persistent bug), the bot resigns on its own turn rather than leave
        the game to time out; on the opponent's turn its clock is not running, so it waits."""
        gid, backoff, errors, since, feed = match.gid, 1, 0, match.sends, None
        try:
            while not match.over and not self.stopped.is_set():
                try:
                    feed = Feed(self.client.stream(f"/api/bot/game/stream/{gid}"), match.chat, gid)
                    while not match.over and not self.stopped.is_set():
                        if (event := feed.get(match.wait())) is END:
                            raise ConnectionError("game stream closed")
                        if event:
                            match.on_event(event)
                            backoff = 1
                        if feed.empty():  # act on the latest state only
                            match.act()
                        match.tick()
                except NETWORK as e:
                    if feed:  # a failed move POST, too: the stream may still be up
                        feed.abandon()
                    log.warning("game %s stream: %s; retry in %d s", gid, e, backoff)
                    self.stopped.wait(backoff)
                    backoff = min(2 * backoff, 5 if match.to_move() else 30)  # its clock may be running
                except Exception:
                    if feed:
                        feed.abandon()
                    if match.sends != since:  # the game went on since the last error
                        errors, since = 0, match.sends
                    errors = min(errors + 1, ERRORS)  # the opponent's turn holds it at ERRORS
                    log.exception("game %s: error %d of %d; resuming", gid, errors, ERRORS)
                    if errors >= ERRORS and match.to_move():
                        log.error("game %s: giving up; resigning", gid)
                        try:
                            match.call(self.client.resign, gid)
                        except Exception:  # noqa: BLE001 - the game ends on time anyway
                            log.exception("game %s: resign failed", gid)
                        break
                    self.stopped.wait(min(backoff, 5))  # the bot's clock may be running
                    backoff = min(2 * backoff, 5)
        finally:
            if feed and not match.chat:
                feed.abandon()
            with self.lock:
                self.games.pop(gid, None)
                self.finished[gid] = match.stats
            match.close()


class Match:
    """One game's protocol state: color, the latest state, the move decided and when it is due."""

    def __init__(self, bot, gid):
        self.bot, self.gid = bot, gid
        self.game, self.white, self.over, self.ended, self.opponent = None, None, False, False, None
        # plies: the bot's last move sent (or draw accepted, resignation sent), offer answered, resignation weighed
        self.moved = self.answered = self.conceded = -1
        self.state, self.since, self.ply, self.sends = None, 0.0, 0, 0  # since: when the position arrived
        self.due = None  # (when, Decision, moves): the move decided at this position, kept to resend
        self.sent = None  # (when, ply, refused): the last action, until a state moves past its ply
        self.claim, self.waiting, self.chatted = None, None, 0.0  # waiting: since when the opponent owes a first move
        self.patience = bot.config.abort
        self.stats = []
        self.chat = Chatter(self) if bot.config.chat.enabled else None

    def on_event(self, event):
        if "moves" in (s := event.get("state", event)):  # recorded before anything can fail
            self.ply = len(s["moves"].split())
        match event["type"]:
            case "gameFull":
                self.on_full(event)
            case "gameState":
                self.on_state(event)
            case "opponentGone":
                self.on_gone(event)
            case "chatLine" if event.get("room") == "player":
                if (event.get("username") or "").lower() not in (self.bot.me, "lichess"):
                    self.chatted = time.monotonic()

    def on_full(self, event):
        play = self.bot.config.play
        if event["variant"]["key"] != "standard" or event.get("initialFen", "startpos") != "startpos":
            log.warning("game %s: not standard chess from the start; aborting", self.gid)
            self.call(self.bot.client.abort, self.gid)
            self.over = True
            return
        self.white = event["white"].get("id") == self.bot.me
        side = event["black" if self.white else "white"]
        self.opponent = side.get("id")
        ours, them = play.ratings(side.get("rating"))
        clock = event.get("clock")
        base = clock["initial"] // 1000 if clock else None
        inc = clock["increment"] // 1000 if clock else None
        if self.game is None:
            elo = (ours, them) if self.white else (them, ours)
            self.game = Game(self.bot.engine, *elo, base, inc, event.get("speed", "blitz"))
            self.patience = patience(self.bot.config.abort, base)
            title = f" {t}" if (t := side.get("title")) else ""
            log.info("game %s: %s as %s, Elo %s vs %s%s, %s+%s, %s", self.gid, play.mode,
                     "white" if self.white else "black", ours, them, title, base, inc,
                     "rated" if event.get("rated") else "casual")  # fmt: skip
            if self.bot.config.greeting and not event["state"]["moves"]:
                self.call(self.bot.client.chat, self.gid, self.bot.config.greeting)
        # the server's state is the truth: on the bot's turn, its last move did not arrive (live RtI5RtnK lost
        # on time with 327 s left after a move posted into a Lichess restart)
        self.moved, self.sent = -1, None
        self.on_state(event["state"])

    def on_state(self, s):
        if s["status"] not in ("created", "started"):
            log.info("game %s over: %s, winner %s", self.gid, s["status"], s.get("winner"))
            self.over = True
            return
        moves = s["moves"].split()
        if self.state is None or self.state["moves"] != s["moves"]:
            self.since = time.monotonic()
        if self.due and self.due[2] != s["moves"]:
            self.due = None
        self.state = s
        first = len(moves) < 2 and (len(moves) % 2 == 0) != self.white
        self.waiting = (self.waiting or time.monotonic()) if first else None
        self.game.update(moves, s["wtime"] / 1000, s["btime"] / 1000)

    def act(self):
        """On the latest state (no event waiting): weigh resigning after the bot's own move; on its turn answer
        the opponent's draw offer, decide the move, and send it once its think time is up. An offer is answered
        only before the bot moves: after a move, Lichess has declined it, and a yes would offer a draw."""
        s = self.state
        if self.over or self.ended or s is None:
            return
        game, play, n = self.game, self.bot.config.play, len(self.game.moves)
        clock = (s["wtime"] if self.white else s["btime"]) / 1000
        if (n % 2 == 0) != self.white:  # a human might resign right after a move that turns out badly
            if self.conceded < n:
                self.conceded = n
                if game.concede(play, clock, self.white):
                    log.info("game %s ply %d: resigning after its move", self.gid, n)
                    self.call(self.bot.client.resign, self.gid)
            return
        if self.moved == n:
            return
        if s.get("bdraw" if self.white else "wdraw") and self.answered != n:
            self.answered = n
            accept = game.accept_draw(play, self.white)
            log.info("game %s ply %d: draw offer %s", self.gid, n, "accepted" if accept else "declined")
            self.call(self.bot.client.draw, self.gid, accept)
            if accept:
                self.due, self.moved, self.sent = None, n, (time.monotonic(), n, False)
                return
        if self.due is None:  # then back to the stream: events that came during the search go first
            self.decide(s, clock)
            return
        when, d, _ = self.due
        if time.monotonic() < when:
            return
        if self.answered == n and d.offer_draw:  # it declined theirs: no offer of its own
            d = replace(d, offer_draw=False)
        ok = self.call(self.bot.client.move, self.gid, d.move, d.offer_draw) is not None
        self.moved, self.sent = n, (time.monotonic(), n, not ok)
        if ok:
            self.sends += 1
        else:  # mostly the opponent resigned meanwhile; else tick() resyncs
            self.due = None

    def decide(self, s, clock):
        """The bot's move here, due its think time after the position arrived; or its resignation."""
        game, play, n = self.game, self.bot.config.play, len(self.game.moves)
        start = time.monotonic()
        try:
            d = game.decide(play, self.bot.search, clock)
        except BaseException:  # a Rust panic too (pyo3's PanicException): the policy, else a random move, loudly
            log.exception("game %s: decision failed; playing the policy", self.gid)
            try:
                d = game.decide(replace(play, mode="human", temperature=1.0), None, clock)
            except BaseException:
                log.exception("game %s: the policy failed too; playing a random move", self.gid)
                legal = list(game.board.legal_moves)
                d = Decision(random.choice(legal).uci(), 0.0, (0, 1, 0), 0.0)
        spent = time.monotonic() - start
        self.stats.append(spent)
        searched = (
            f" search {d.sims} {1000 * d.searched:.0f} ms{' stopped' if d.stopped else ' cut' if d.cut else ''}"
            if d.sims
            else ""
        )
        log.info("game %s ply %d: %s p=%.2f wdl=%.2f/%.2f/%.2f %.0f ms think %.1f s%s%s", self.gid, n, d.move,
                 d.probability, *d.wdl, 1000 * spent, d.think, searched, "; resigning" * d.resign)  # fmt: skip
        if d.resign:
            self.call(self.bot.client.resign, self.gid)
            self.moved, self.sent = n, (time.monotonic(), n, False)
            return
        self.due = (self.since + d.think, d, s["moves"])

    def wait(self):
        """Seconds the game thread may wait for an event: until the move is due, at most one (tick())."""
        if self.due is None or self.moved == self.ply:
            return 1.0
        return min(max(self.due[0] - time.monotonic(), 0.005), 1.0)

    def to_move(self):
        """Whether the bot's clock runs, by Lichess's last state (unknown color: assume so)."""
        return self.white is None or (self.ply % 2 == 0) == self.white

    def tick(self):
        """Resync a game whose state shows no move LATE seconds after the bot's action: refused, the board must
        disagree; accepted, Lichess lost it. Abort a game whose opponent has not made a first move in its
        patience (longer while they chat): Lichess can leave it open for an hour, holding a slot and a drain."""
        now = time.monotonic()
        if self.sent and not self.over and self.ply == self.sent[1] and now - self.sent[0] > LATE:
            refused, self.sent = self.sent[2], None
            if refused:
                raise RuntimeError("move refused and the game goes on; resyncing")
            raise ConnectionError(f"no new state {LATE:g} s after the bot's move; resyncing")
        if self.waiting and self.patience and not self.over:
            if now - max(self.waiting, self.chatted) > self.patience or now - self.waiting > 3 * self.patience:
                log.info("game %s: no first move from the opponent in %.0f s; aborting", self.gid,
                         now - self.waiting)  # fmt: skip
                self.waiting = self.chatted = now  # if the abort fails, retry a period later
                self.call(self.bot.client.abort, self.gid)

    def on_gone(self, event):
        if self.claim:
            self.claim.cancel()
        if event.get("gone") and "claimWinInSeconds" in event:
            self.claim = threading.Timer(
                event["claimWinInSeconds"] + 1,
                self.call,
                (self.bot.client.claim_victory, self.gid),
            )
            self.claim.daemon = True
            self.claim.start()

    def call(self, fn, *args):
        try:
            return fn(*args)
        except urllib.error.HTTPError as e:  # e.g. the game ended meanwhile
            log.warning("game %s: %s %s", self.gid, fn.__name__, e)

    def close(self):
        if self.claim:
            self.claim.cancel()
        if self.chat:
            self.chat.close()
        if self.stats:
            ms = sorted(1000 * s for s in self.stats)
            log.info("game %s: %d moves, decision ms median %.0f max %.0f", self.gid, len(ms),
                     ms[len(ms) // 2], ms[-1])  # fmt: skip
