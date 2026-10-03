"""A local stand-in for lichess.org's Bot API, for tests and offline self-play.

Users are identified by token. Games run real clocks (each side's first move is free, increment
after every move), check every move with python-chess, and end by mate, stalemate, automatic
draws (repetition, 50 moves, material), flag, resignation, agreed draw or max_plies. Users that
have no token are played by the server ("house" players: random movers by default).
"""

import json
import random
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue

import chess


def speed(base, inc):
    t = base + 40 * inc
    for limit, name in (
        (29, "ultraBullet"),
        (179, "bullet"),
        (479, "blitz"),
        (1499, "rapid"),
    ):
        if t < limit:
            return name
    return "classical"


class MockGame:
    def __init__(self, gid, white, black, base, inc, rated, ratings, max_plies):
        self.id, self.white, self.black, self.max_plies = gid, white, black, max_plies
        self.base, self.inc, self.rated, self.ratings = base, inc, rated, ratings
        self.board = chess.Board()
        self.clock = [base * 1000.0, base * 1000.0]
        self.turn_start = time.monotonic()
        self.status, self.winner = "started", None
        self.draw = {"white": False, "black": False}
        self.version = 0
        self.history = []  # each move's mover's clock after it, ms
        self.changed = threading.Condition()
        self.chat = []

    def color(self, user):
        return (
            "white" if user == self.white else "black" if user == self.black else None
        )

    def state(self):
        s = dict(
            type="gameState",
            moves=" ".join(m.uci() for m in self.board.move_stack),
            wtime=int(self.clock[0]),
            btime=int(self.clock[1]),
            winc=self.inc * 1000,
            binc=self.inc * 1000,
            status=self.status,
        )
        s |= {f"{c[0]}draw": True for c in ("white", "black") if self.draw[c]}  # true only
        if self.winner:
            s["winner"] = self.winner
        return s

    def full(self):
        side = lambda u, r: dict(id=u, name=u, rating=r, title=None)
        return dict(
            type="gameFull",
            id=self.id,
            variant=dict(key="standard"),
            speed=speed(self.base, self.inc),
            rated=self.rated,
            clock=dict(initial=self.base * 1000, increment=self.inc * 1000),
            white=side(self.white, self.ratings[0]),
            black=side(self.black, self.ratings[1]),
            initialFen="startpos",
            state=self.state(),
        )

    def publish(self):
        with self.changed:
            self.version += 1
            self.changed.notify_all()

    def end(self, status, winner=None):
        self.status, self.winner = status, winner
        self.publish()

    def move(self, user, uci, offer=False):
        """None if played, else the error. Moving declines the opponent's draw offer; offer:
        offer one with the move."""
        if self.status != "started":
            return "game over"
        side = 0 if self.board.turn == chess.WHITE else 1
        if self.color(user) != ("white", "black")[side]:
            return "not your turn"
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return "bad move"
        if move not in self.board.legal_moves:
            return f"illegal move {uci}"
        now = time.monotonic()
        if len(self.board.move_stack) >= 2:
            self.clock[side] -= 1000 * (now - self.turn_start)
            if self.clock[side] < 0:
                self.clock[side] = 0
                self.end("outoftime", ("black", "white")[side])
                return None
            self.clock[side] += 1000 * self.inc
        self.turn_start = now
        self.board.push(move)
        self.history.append(int(self.clock[side]))
        color = ("white", "black")[side]
        self.draw = {color: offer, ("black", "white")[side]: False}
        out = self.board.outcome(claim_draw=True)
        if out:
            if out.termination == chess.Termination.CHECKMATE:
                self.end("mate", "white" if out.winner else "black")
            elif out.termination == chess.Termination.STALEMATE:
                self.end("stalemate")
            else:
                self.end("draw")
        elif len(self.board.move_stack) >= self.max_plies:
            self.end("draw")
        else:
            self.publish()
        return None


class MockLichess:
    def __init__(self, tokens, house=None, house_delay=0.05, max_plies=400):
        self.tokens = dict(tokens)  # token -> user id
        self.house_offers = set()  # plies at which house players offer a draw with their move
        self.house = house or (
            lambda board: random.choice(list(board.legal_moves)).uci()
        )
        self.house_delay, self.max_plies = house_delay, max_plies
        self.events = {u: Queue() for u in self.tokens.values()}
        self.games, self.challenges, self.declined, self.rejected = {}, {}, [], []
        self.calls, self.fail = (
            [],
            {},
        )  # fail: path prefix -> [HTTP codes to return first]
        self.drop_after = (
            None  # close each user's next game stream after this many events
        )
        self.next_id, self.lock = 0, threading.Lock()
        self.closing = threading.Event()
        handler = type("Handler", (Handler,), dict(mock=self))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.closing.set()
        for g in self.games.values():
            g.publish()
        self.server.shutdown()

    def uid(self, prefix):
        with self.lock:
            self.next_id += 1
            return f"{prefix}{self.next_id:07d}"

    def challenge(self, challenger, dest, base=180, inc=2, rated=False, color="random",
                  variant="standard", ratings=(1500, 1500), title=None, kind="clock"):  # fmt: skip
        cid = self.uid("g")  # as on Lichess, the game keeps the challenge's id
        tc = (
            dict(type=kind, limit=base, increment=inc)
            if kind == "clock"
            else dict(type=kind)
        )
        self.challenges[cid] = c = dict(
            id=cid,
            status="created",
            challenger=dict(
                id=challenger, name=challenger, rating=ratings[0], title=title
            ),
            destUser=dict(id=dest, name=dest, rating=ratings[1]),
            variant=dict(key=variant),
            rated=rated,
            speed=speed(base, inc) if kind == "clock" else kind,
            timeControl=tc,
            color=color,
            ratings=ratings,
        )
        self.events[dest].put(dict(type="challenge", challenge=c))
        return cid

    def accept(self, cid):
        c = self.challenges[cid]
        a, b = c["challenger"]["id"], c["destUser"]["id"]
        flip = c["color"] == "black" or c["color"] == "random" and random.random() < 0.5
        white, black = (b, a) if flip else (a, b)
        r = c["ratings"][::-1] if flip else c["ratings"]
        tc = c["timeControl"]
        limit, inc = tc["limit"], tc["increment"]
        g = MockGame(cid, white, black, limit, inc, c["rated"], r, self.max_plies)
        self.games[g.id] = g
        for u in (white, black):
            if u in self.events:
                self.events[u].put(
                    dict(type="gameStart", game=dict(gameId=g.id, id=g.id))
                )
            else:
                threading.Thread(
                    target=self.house_player, args=(g, u), daemon=True
                ).start()
        return g

    def house_player(self, g, user):
        seen = -1
        while g.status == "started" and not self.closing.is_set():
            with g.changed:
                if g.version == seen:
                    g.changed.wait(1)
                seen = g.version
            white = g.board.turn == chess.WHITE
            if g.status == "started" and g.color(user) == (
                "white" if white else "black"
            ):
                time.sleep(self.house_delay)
                offer = len(g.board.move_stack) in self.house_offers
                g.move(user, self.house(g.board.copy()), offer)

    def offer_draw(self, g, color):
        g.draw[color] = True
        g.publish()


class Handler(BaseHTTPRequestHandler):
    mock: MockLichess
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def user(self):
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        return self.mock.tokens.get(token)

    def reply(self, code, body=None):
        data = json.dumps(body if body is not None else {"ok": code == 200}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def route(self, method):
        m, path = self.mock, urllib.parse.urlparse(self.path)
        user = self.user()
        m.calls.append((user, method, self.path))
        if user is None:
            return self.reply(401, {"error": "No such token"})
        for prefix, codes in m.fail.items():
            if path.path.startswith(prefix) and codes:
                return self.reply(codes.pop(0), {"error": "injected"})
        parts = path.path.strip("/").split("/")
        match method, parts:
            case "GET", ["api", "account"]:
                return self.reply(200, dict(id=user, username=user, title="BOT"))
            case "GET", ["api", "stream", "event"]:
                return self.stream_events(user)
            case "GET", ["api", "bot", "game", "stream", gid]:
                return self.stream_game(user, m.games.get(gid))
            case "POST", ["api", "challenge", cid, "accept"]:
                m.accept(cid)
                return self.reply(200)
            case "POST", ["api", "challenge", cid, "decline"]:
                n = int(self.headers.get("Content-Length") or 0)
                form = urllib.parse.parse_qs(self.rfile.read(n).decode())
                m.declined.append((cid, form.get("reason", ["generic"])[0]))
                return self.reply(200)
            case "POST", ["api", "bot", "game", gid, *action]:
                return self.game_action(user, m.games.get(gid), action, path.query)
        return self.reply(404, {"error": "not found"})

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def line(self, obj=None):
        self.wfile.write((json.dumps(obj) if obj is not None else "").encode() + b"\n")
        self.wfile.flush()

    def open_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

    def stream_events(self, user):
        self.open_stream()
        q = self.mock.events[user]
        try:
            while not self.mock.closing.is_set():
                try:
                    self.line(q.get(timeout=1))
                except Empty:
                    self.line()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_game(self, user, g):
        if g is None:
            return self.reply(404, {"error": "no such game"})
        self.open_stream()
        drop, sent = self.mock.drop_after, 0
        self.mock.drop_after = None
        try:
            with g.changed:
                seen = g.version
            self.line(g.full())
            while g.status == "started" and not self.mock.closing.is_set():
                with g.changed:
                    if g.version == seen:
                        g.changed.wait(1)
                    changed, seen = g.version != seen, g.version
                if changed:
                    self.line(g.state())
                    sent += 1
                    if drop is not None and sent >= drop:
                        return
                else:
                    self.line()
            self.line(g.state())
        except (BrokenPipeError, ConnectionResetError):
            pass

    def game_action(self, user, g, action, query):
        if g is None or g.color(user) is None:
            return self.reply(404, {"error": "no such game"})
        color = g.color(user)
        match action:
            case ["move", uci]:
                err = g.move(user, uci, "offeringDraw=true" in query)
                if err:
                    self.mock.rejected.append((g.id, user, uci, err))
                return self.reply(400, {"error": err}) if err else self.reply(200)
            case ["resign"]:
                g.end("resign", "black" if color == "white" else "white")
            case ["abort"]:
                g.end("aborted")
            case ["draw", answer]:
                other = "black" if color == "white" else "white"
                if answer == "yes" and g.draw[other]:
                    g.end("draw")
                elif answer == "yes":
                    self.mock.offer_draw(g, color)
                else:
                    g.draw[other] = False
                    g.publish()
            case ["chat"]:
                n = int(self.headers.get("Content-Length") or 0)
                g.chat.append(urllib.parse.parse_qs(self.rfile.read(n).decode()))
            case ["claim-victory"]:
                pass
            case _:
                return self.reply(404, {"error": "unknown action"})
        return self.reply(200)
