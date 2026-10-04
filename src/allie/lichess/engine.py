"""Move choice for live games: one resident model, one inference thread batching all games."""

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from time import monotonic

import chess
import numpy as np
import torch

from . import behaviour, calibration
from .behaviour import FORMATS
from .model import Cache, step
from .tokens import CONTEXT, HEADER, MOVE_ID, START, advance, features, fill, header

TIME, WDL = slice(2350, 2413), slice(2413, 2416)


@dataclass
class Play:
    mode: str = "human"  # a key of MODES: human samples the policy; strongest: argmax (or search);
    # calibrated: plays at the strength of humans of its rating (calibration.py)
    rating: int | str = 1500  # the bot's header Elo, or "opponent" to mirror it
    temperature: float = 1.0  # human mode
    search: int = 0  # strongest mode: coverage-search simulations (0 = policy argmax)
    think_time: bool = True  # wait a think time drawn from the model's think-time head
    resign: bool = True  # resign as often and as late as humans of the bot's rating do
    draws: bool = True  # offer and accept draws as humans do (otherwise: never)
    lag: float = 0.3  # seconds Lichess charges beyond the bot's own think time (measured live)


class Engine:
    """Owns the model; game threads call extend(), steps() and run(), which the inference thread
    serves, batching concurrent extend() and steps() calls into one forward of up to max_items."""

    def __init__(self, model, max_batch=32, max_items=64):
        behaviour.check()
        self.model, self.max_batch, self.max_items = model, max_batch, max_items
        self.requests = queue.SimpleQueue()
        self.forwards = self.tokens = self.widest = 0  # widest: most items in one forward
        self.thread = threading.Thread(target=self._loop, daemon=True, name="inference")
        self.thread.start()

    def close(self):
        self.requests.put(None)
        self.thread.join()

    def extend(self, cache, ids, feats, boards):
        """Append tokens to a game's cache; logits at the last one."""
        return self._call((cache, ids, feats, boards))

    def steps(self, items):
        """extend() for several caches at once (a search's nodes); logits [len(items), vocab]."""
        return self._call(list(items))

    def run(self, fn):
        """fn() on the inference thread, alone."""
        return self._call(fn)

    def _call(self, request):
        f = Future()
        self.requests.put((request, f))
        return f.result()

    def _loop(self):
        while (first := self.requests.get()) is not None:
            batch, items = [first], self._size(first[0])
            while len(batch) < self.max_batch and items < self.max_items:
                try:
                    batch.append(r := self.requests.get_nowait())
                except queue.Empty:
                    break
                if r is None:  # close(): serve what came before it, then stop
                    self.requests.put(None)
                    batch.pop()
                    break
                items += self._size(r[0])
            groups = [(r if isinstance(r, list) else [r], f, isinstance(r, list))
                      for r, f in batch if not callable(r)]  # fmt: skip
            if groups:
                self._forward(groups)
            for r, f in batch:
                if callable(r):
                    try:
                        f.set_result(r())
                    except Exception as e:  # noqa: BLE001
                        f.set_exception(e)

    def _forward(self, groups):
        """One forward for every group's items; if it fails, each group alone, so one bad request
        fails only its own caller."""
        flat = [it for g, _, _ in groups for it in g]
        try:
            logits = step(self.model, flat).cpu()
        except Exception as e:  # noqa: BLE001
            if len(groups) > 1:
                for g in groups:
                    self._forward([g])
                return
            groups[0][1].set_exception(e)
            return
        self.forwards += 1
        self.tokens += sum(len(it[1]) for it in flat)
        self.widest = max(self.widest, len(flat))
        lo = 0
        for g, f, many in groups:
            f.set_result(logits[lo : lo + len(g)] if many else logits[lo])
            lo += len(g)

    @staticmethod
    def _size(r):
        return len(r) if isinstance(r, list) else 0 if r is None or callable(r) else 1


@dataclass
class Decision:
    move: str
    think: float  # seconds to wait before playing, already capped by the clock
    wdl: tuple  # the mover's (bot's) win / draw / loss probabilities
    probability: float  # of the chosen move
    resign: bool = False
    offer_draw: bool = False


class Game:
    """One game's tokens, clocks and cache. Clocks: each move's mover's clock after it."""

    def __init__(self, engine, white, black, base, increment, speed="blitz", seed=None):
        self.engine, self.base, self.inc = engine, base, increment
        self.speed, self.elo = speed, (white, black)
        self.tokens = header(base, increment, white, black)
        self.boards = [START] * HEADER
        self.moves, self.clocks = [], []
        self.board = chess.Board()
        self.cache = Cache(engine.model)
        self.logits, self.used = None, []  # the logits at the cache's end; its (token, features)
        self.rng = np.random.default_rng(seed)

    def update(self, moves, wtime=None, btime=None):
        """The server's move list and clocks (seconds) after its last move."""
        common = 0
        while (
            common < min(len(moves), len(self.moves))
            and moves[common] == self.moves[common]
        ):
            common += 1
        if common < len(self.moves):  # takeback or a different game state: rewind
            del self.moves[common:], self.clocks[common:]
            del self.tokens[HEADER + common :], self.boards[HEADER + common :]
            self.board = chess.Board()
            for m in self.moves:
                self.board.push_uci(m)
            self.cache.truncate(HEADER + common)
            self.logits = None
        for j in range(common, len(moves)):
            move = self.board.parse_uci(moves[j])
            self.board.push(move)
            token = MOVE_ID[move.uci()]
            self.moves.append(moves[j])
            self.clocks.append(None)
            self.tokens.append(token)
            self.boards.append(advance(self.boards[-1], token))
        # the event's clocks are the last move's mover's and the other side's after its move
        # before; a new move's clock comes only from the first event that reports it
        for j in (len(moves) - 1, len(moves) - 2):
            t = wtime if j % 2 == 0 else btime
            if j >= common and t is not None:
                self.clocks[j] = int(t)

    def sync(self):
        """Bring the cache up to the last known token; the logits there. Clocks learnt later
        can revise earlier positions' features (gaps are filled), so the cache is kept only
        up to the first position whose token or features changed."""
        if len(self.tokens) > CONTEXT:
            raise OverflowError("game longer than the model's context")
        now = list(zip(self.tokens, map(tuple, self.features())))
        n, top = 0, min(self.cache.n, len(self.used))
        while n < top and self.used[n] == now[n]:
            n += 1
        if n == len(now) and self.logits is not None:
            return self.logits
        n = min(n, len(now) - 1)  # at least the last token, for its logits
        self.cache.truncate(n)
        self.logits = self.engine.extend(
            self.cache,
            torch.tensor(self.tokens[n:]),
            torch.tensor([f for _, f in now[n:]], dtype=torch.float32),
            torch.tensor(np.frombuffer(b"".join(self.boards[n:]), np.uint8).reshape(-1, 68)),
        )
        self.used = now
        return self.logits

    def features(self):
        """Clock features of every token."""
        clocks = fill(self.clocks, self.base)
        f = lambda p: features(p - HEADER + 1, self.base, self.inc, clocks)
        return [f(p) if p >= HEADER - 1 else [-1] * 3 for p in range(len(self.tokens))]

    def decide(self, play, search=None, clock=None):
        """The bot's move at the current position. clock: its time left in seconds."""
        try:
            return MODES[play.mode](self, play, search, clock)
        except OverflowError:  # past 1,014 plies: a random legal move
            legal = [m.uci() for m in self.board.legal_moves]
            return Decision(str(self.rng.choice(legal)), 0.0, (0, 1, 0), 1 / len(legal))

    def position(self):
        """The legal moves (UCI), their probabilities, the side to move's win / draw / loss
        probabilities and its think-time distribution (63 bins)."""
        z = self.sync().double()
        legal = [m.uci() for m in self.board.legal_moves]
        p = torch.softmax(z[[MOVE_ID[m] for m in legal]], 0).numpy()
        return legal, p, tuple(torch.softmax(z[WDL], 0).tolist()), torch.softmax(z[TIME], 0).numpy()

    def concede(self, play, clock, white):
        """After the bot's own move (the opponent to move): resign now, as a human would on
        seeing the new position? clock: the bot's time left."""
        if not play.resign or len(self.moves) < 2:
            return False
        try:
            w, d, loss = torch.softmax(self.sync().double()[WDL], 0).tolist()
        except OverflowError:
            return False
        ply, me = len(self.moves), 0 if white else 1
        if ply % 2 == me:  # the bot is to move: not after its own move
            return False
        x = ((loss, d, w), ply, self.elo[me], FORMATS.get(self.speed, 1), clock, self.base)
        return behaviour.resign(*x, self.rng, on_turn=False)

    def behave(self, play, clock, move, probability, wdl, time):
        """A Decision for this move: think time, resignation and draw offer as humans of the
        bot's rating behave (behaviour.py)."""
        ply = len(self.moves)
        x = (wdl, ply, self.elo[ply % 2], FORMATS.get(self.speed, 1), clock, self.base)
        return Decision(
            move=move,
            think=behaviour.think(time, self.rng, clock, self.inc, ply, play.lag) if play.think_time else 0.0,
            wdl=wdl,
            probability=float(probability),
            resign=play.resign and behaviour.resign(*x, self.rng),
            offer_draw=play.draws and behaviour.offer_draw(*x, self.rng),
        )

    def accept_draw(self, play, white):
        """For the bot playing white (or not): accept a draw offer, as a human would?"""
        if not play.draws:
            return False
        try:
            w, d, loss = torch.softmax(self.sync().double()[WDL], 0).tolist()
        except OverflowError:
            return True
        if (len(self.moves) % 2 == 0) != white:  # the head speaks for the side to move
            w, loss = loss, w
        return behaviour.accept_draw((w, d, loss))

    def cell(self, elo):
        """allie.search's cell: format x mover Elo band."""
        band = 0 if elo < 1400 else 1 if elo < 2000 else 2 if elo < 2400 else 3
        return 4 * FORMATS.get(self.speed, 1) + band


def human(game, play, search, clock):
    """A move sampled from the policy at play.temperature."""
    legal, p, wdl, time = game.position()
    if play.temperature <= 0:
        i = int(p.argmax())
    else:
        t = np.log(np.maximum(p, 1e-300))
        with np.errstate(over="ignore"):
            t = np.exp((t - t.max()) / play.temperature)
        i = int(game.rng.choice(len(p), p=t / t.sum()))
    return game.behave(play, clock, legal[i], p[i], wdl, time)


def strongest(game, play, search, clock):
    """The most likely move, or the searched distribution's (play.search simulations)."""
    legal, p, wdl, time = game.position()
    if search is not None and play.search:
        moves, q = search(game, play.search)
        p = np.zeros(len(legal))
        p[[legal.index(m) for m in moves]] = q
    i = int(p.argmax())
    return game.behave(play, clock, legal[i], p[i], wdl, time)


def calibrated(game, play, search, clock):
    """Moves of the quality humans of the bot's rating make at this time control (calibration.py):
    sampled from the policy, or from coverage search's distribution with a budget that grows with
    the predicted human think time and the rating."""
    start = monotonic()
    legal, p, wdl, time = game.position()
    rating = game.elo[len(game.moves) % 2]
    n = 0  # search costs what calibration.HW assumes only on the fast CPU backend
    if search is not None and len(legal) > 1 and game.engine.model.fast is not None:
        left = None if clock is None else clock - (monotonic() - start)  # less any wait for the engine
        most = calibration.ceiling(left, behaviour.PARAMETERS["guard"]["reserve"])
        n = calibration.pick(calibration.budget(rating, time, game.speed), game.rng, most)
    s = p
    if n:
        moves, q = search(game, n)
        s = np.zeros(len(legal))
        s[[legal.index(m) for m in moves]] = q
    i = int(game.rng.choice(len(legal), p=s / s.sum()))
    return game.behave(play, clock, legal[i], s[i], wdl, time)


# play.mode -> fn(game, play, search, clock) -> Decision. A mode chooses the move its own way and
# usually leaves timing, resignation and draws to game.behave.
MODES = dict(human=human, strongest=strongest, calibrated=calibrated)

