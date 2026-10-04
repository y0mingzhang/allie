"""The bot: accept challenges by the config's rules, play every game in its own thread."""

import logging
import random
import threading
import time
import urllib.error

from .engine import Decision, Game

log = logging.getLogger(__name__)


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


class Bot:
    def __init__(self, config, client, engine, search=None):
        self.config, self.client, self.engine, self.search = (
            config,
            client,
            engine,
            search,
        )
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
            except (OSError, urllib.error.URLError, ValueError) as e:
                log.warning("event stream: %s; reconnecting in %d s", e, backoff)
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
        """Under the lock: the opponents of current games and of accepted challenges whose
        game has not started yet (a challenge's game has its id; reservations last a minute)."""
        now = time.monotonic()
        self.pending = {c: v for c, v in self.pending.items() if now - v[1] < 60}
        return [m.opponent for m in self.games.values()] + [u for u, _ in self.pending.values()]

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

    def play(self, match):
        gid, backoff = match.gid, 1
        try:
            while not match.over and not self.stopped.is_set():
                try:
                    for event in self.client.stream(f"/api/bot/game/stream/{gid}"):
                        if event:
                            match.on_event(event)
                            backoff = 1
                        if match.over or self.stopped.is_set():
                            break
                    else:
                        if not match.over and not self.stopped.is_set():
                            raise ConnectionError("game stream closed")
                except (OSError, urllib.error.URLError, ValueError) as e:
                    log.warning("game %s stream: %s; retry in %d s", gid, e, backoff)
                    self.stopped.wait(backoff)
                    backoff = min(2 * backoff, 30)
        except Exception:
            log.exception("game %s failed", gid)
            raise
        finally:
            with self.lock:
                self.games.pop(gid, None)
                self.finished[gid] = match.stats
            match.close()


class Match:
    """One game's protocol state: color, answered offers, the move already sent."""

    def __init__(self, bot, gid):
        self.bot, self.gid = bot, gid
        self.game, self.white, self.over, self.opponent = None, None, False, None
        self.moved = self.answered = -1
        self.claim = None
        self.stats = []

    def on_event(self, event):
        match event["type"]:
            case "gameFull":
                self.on_full(event)
            case "gameState":
                self.on_state(event)
            case "opponentGone":
                self.on_gone(event)

    def on_full(self, event):
        play = self.bot.config.play
        if event["variant"]["key"] != "standard" or event.get("initialFen", "startpos") != "startpos":
            log.warning("game %s: not standard chess from the start; aborting", self.gid)
            self.call(self.bot.client.abort, self.gid)
            self.over = True
            return
        self.white = event["white"].get("id") == self.bot.me
        self.opponent = event["black" if self.white else "white"].get("id")
        them = event["black" if self.white else "white"].get("rating")
        ours = them if play.rating == "opponent" else play.rating
        them = them or ours or 1500
        ours = ours or them
        clock = event.get("clock")
        base = clock["initial"] // 1000 if clock else None
        inc = clock["increment"] // 1000 if clock else None
        if self.game is None:
            elo = (ours, them) if self.white else (them, ours)
            self.game = Game(
                self.bot.engine, *elo, base, inc, event.get("speed", "blitz")
            )
            log.info("game %s: %s as %s, Elo %s vs %s, %s+%s", self.gid, play.mode,
                     "white" if self.white else "black", ours, them, base, inc)  # fmt: skip
            if self.bot.config.greeting and not event["state"]["moves"]:
                self.call(self.bot.client.chat, self.gid, self.bot.config.greeting)
        self.on_state(event["state"])

    def on_state(self, s):
        start = time.monotonic()
        if s["status"] not in ("created", "started"):
            log.info(
                "game %s over: %s, winner %s", self.gid, s["status"], s.get("winner")
            )
            self.over = True
            return
        moves = s["moves"].split()
        game, play = self.game, self.bot.config.play
        game.update(moves, s["wtime"] / 1000, s["btime"] / 1000)
        if s.get("bdraw" if self.white else "wdraw") and self.answered != len(moves):
            self.answered = len(moves)
            self.call(self.bot.client.draw, self.gid, game.accept_draw(play, self.white))
        if (len(moves) % 2 == 0) != self.white:
            game.sync()  # the bot's own move joins the cache while the opponent thinks
            return
        if self.moved == len(moves):
            return
        clock = (s["wtime"] if self.white else s["btime"]) / 1000
        try:
            d = game.decide(play, self.bot.search, clock)
        except Exception:  # keep the game going on a random legal move, loudly
            log.exception("game %s: decision failed; playing a random move", self.gid)
            legal = list(game.board.legal_moves)
            d = Decision(random.choice(legal).uci(), 0.0, (0, 1, 0), 0.0)
        spent = time.monotonic() - start
        self.stats.append(spent)
        log.info("game %s ply %d: %s p=%.2f wdl=%.2f/%.2f/%.2f %.0f ms think %.1f s", self.gid,
                 len(moves), d.move, d.probability, *d.wdl, 1000 * spent, d.think)  # fmt: skip
        if d.resign:
            self.call(self.bot.client.resign, self.gid)
            return
        if d.think > spent:
            self.bot.stopped.wait(d.think - spent)
        if self.call(self.bot.client.move, self.gid, d.move, d.offer_draw) is not None:
            self.moved = len(moves)

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
        if self.stats:
            ms = sorted(1000 * s for s in self.stats)
            log.info("game %s: %d moves, decision ms median %.0f max %.0f", self.gid, len(ms),
                     ms[len(ms) // 2], ms[-1])  # fmt: skip
