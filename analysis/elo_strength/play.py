"""Games between Allie 2.0 players (allie.lichess, read-only) and Stockfish UCI_Elo anchors.

Players: allie-h{R} (sample at T=1), allie-a{R} (argmax), allie-s{N}-{R} (argmax + N-simulation
coverage search), sf-{ELO} (Stockfish UCI_LimitStrength). Allie's header is R v R. Clocks are
simulated 3+2 with Lichess rules; see DESIGN.md.

python play.py TASKS OUT_DIR [--shard i --shards n] [--model DIR] [--threads 8] [--games 2]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
from allie.lichess.engine import Engine, Game, Play

BASE, INC = 180, 2  # default time control; a task may set "tc": [base, increment]


def speed(base, inc):
    """Lichess's speed class from the estimated game duration base + 40 * increment."""
    t = base + 40 * inc
    return (
        "ultraBullet"
        if t < 30
        else "bullet"
        if t < 180
        else "blitz"
        if t < 480
        else "rapid"
        if t < 1500
        else "classical"
    )


MAX_PLIES = 500
SF = os.environ.get(
    "STOCKFISH",
    "/data/group_data/dei-group/yimingz3/allie/tools/stockfish/stockfish/stockfish-linux-x86-64-universal",
)
MODEL = "/data/group_data/dei-group/yimingz3/allie/lichess/allie-2.0-annealed"


def sf_depth(elo):
    """Stockfish 19's pick depth for UCI_Elo (search.h Skill): 1 + int(level)."""
    e = (elo - 1320) / (3190 - 1320)
    level = min(max(((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438, 0.0), 19.0)
    return 1 + int(level)


MAIA = "/data/group_data/dei-group/yimingz3/allie/maia3-bench"


class UCI:
    """A UCI engine over pipes: Stockfish (UCI_Elo or full strength) or Maia-3."""

    def __init__(self, cmd, options=(), depth=1, cwd=None, env=None):
        self.p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )
        self.depth = depth
        self.send("uci", *(f"setoption name {k} value {v}" for k, v in options))
        self.ready()

    def send(self, *lines):
        self.p.stdin.write("".join(line + "\n" for line in lines))

    def ready(self):
        self.send("isready")
        while self.p.stdout.readline().strip() != "readyok":
            pass

    def go(self, moves, depth=None):
        """(best move, last multipv-1 score in cp from the mover's view, or None)."""
        self.send(
            "position startpos moves " + " ".join(moves),
            f"go depth {depth or self.depth}",
        )
        score = None
        while line := self.p.stdout.readline():
            if line.startswith("info") and " multipv 1 " in line:
                if m := re.search(r"score (cp|mate) (-?\d+)", line):
                    score = (
                        int(m[2])
                        if m[1] == "cp"
                        else 100000 * (1 if int(m[2]) > 0 else -1)
                    )
            elif line.startswith("bestmove"):
                return line.split()[1], score
        raise RuntimeError(f"{self.p.args[0]} exited")

    def new_game(self):
        self.send("ucinewgame")
        self.ready()

    def close(self):
        self.send("quit")
        self.p.wait()


def stockfish(elo=None):
    opts = [("Threads", 1), ("Hash", 16)]
    if elo is None:
        return UCI([SF], opts, depth=None)
    return UCI(
        [SF], opts + [("UCI_LimitStrength", "true"), ("UCI_Elo", elo)], sf_depth(elo)
    )


def maia3(elo, seed):
    """Maia-3 79M sampling at T=1 with both Elos = elo, history from the move list, no clock."""
    cmd = [sys.executable, "-m", "maia3.uci", "--model", "maia3-79m", "--device", "cpu",
           "--cache-dir", f"{MAIA}/hf/hub", "--local-files-only", "--elo", str(elo),
           "--temperature", "1", "--multipv", "1", "--use-uci-history", "--seed", str(seed)]  # fmt: skip
    env = os.environ | dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return UCI(cmd, cwd=f"{MAIA}/maia3-repo", env=env)


def parse(spec):
    """'allie-h1600' -> ('allie', Play), 'sf-1500' -> ('sf', 1500), 'maia3-h1500' -> ('maia3', 1500)."""
    if m := re.fullmatch(r"sf-(\d+)", spec):
        return "sf", int(m[1])
    if m := re.fullmatch(r"maia3-h(\d+)", spec):
        return "maia3", int(m[1])
    if m := re.fullmatch(r"allie-h(\d+)", spec):
        return "allie", Play(mode="human", rating=int(m[1]), temperature=1.0)
    if m := re.fullmatch(r"allie-a(\d+)", spec):
        return "allie", Play(mode="strongest", rating=int(m[1]))
    if m := re.fullmatch(r"allie-s(\d+)-(\d+)", spec):
        return "allie", Play(mode="strongest", rating=int(m[2]), search=int(m[1]))
    if m := re.fullmatch(r"allie-c(\d+)", spec):  # calibrated.py's mode
        return "allie", Play(mode="calibrated", rating=int(m[1]))
    if m := re.fullmatch(
        r"allie-t([0-9.]+)-(\d+)", spec
    ):  # sampling at a fixed temperature
        return "allie", Play(mode="human", rating=int(m[2]), temperature=float(m[1]))
    raise ValueError(spec)


class Side:
    """One player in one game: an Allie Game (own header, cache, rng) or a UCI engine."""

    def __init__(self, spec, ctx, seed, tc=(BASE, INC)):
        self.spec, (self.kind, self.cfg) = spec, parse(spec)
        if self.kind == "allie":
            r = self.cfg.rating
            self.game = Game(ctx.engine, r, r, *tc, speed(*tc), seed)
        else:
            self.uci = ctx.uci(self.kind, self.cfg)
            self.uci.new_game()

    def choose(self, ctx, moves, clock):
        """(uci, think seconds or None to let the opponent's model draw it)."""
        if self.kind != "allie":
            return self.uci.go(moves)[0], None
        if self.cfg.mode == "calibrated":
            import calibrated

            return calibrated.decide(self.game, self.cfg.rating, clock)[:2]
        d = self.game.decide(self.cfg, ctx.search, clock)
        return d.move, d.think

    def opponent_think(self, clock):
        """The opponent's think time as this Allie side's time head predicts it (header R v R)."""
        return self.game.think(self.game.sync().double(), Play(), clock)


def play(task, ctx, book):
    seed = zlib.crc32(task["id"].encode())
    opening = book[task["opening"]]
    base, inc = task.get("tc", (BASE, INC))
    sides = [
        Side(task["white"], ctx, seed, (base, inc)),
        Side(task["black"], ctx, seed + 1, (base, inc)),
    ]
    board, moves, clocks, history = chess.Board(), [], [float(base), float(base)], []
    t0 = time.monotonic()
    while (outcome := board.outcome(claim_draw=True)) is None and len(
        moves
    ) < MAX_PLIES:
        ply = len(moves)
        me, other = sides[ply % 2], sides[1 - ply % 2]
        if ply < len(opening["moves"]):
            uci, think = opening["moves"][ply], opening["think"][ply]
        else:
            uci, think = me.choose(ctx, moves, clocks[ply % 2])
            if think is None:
                think = (
                    other.opponent_think(clocks[ply % 2])
                    if other.kind == "allie"
                    else 0.0
                )
        if ply >= 2:
            clocks[ply % 2] = clocks[ply % 2] - think + inc
        board.push_uci(uci)
        moves.append(uci)
        history.append(round(clocks[ply % 2], 1))
        for s in sides:
            if s.kind == "allie":
                s.game.update(moves, *clocks)
    result = outcome.result() if outcome else "1/2-1/2"
    term = outcome.termination.name.lower() if outcome else "max_plies"
    return dict(
        task,
        result=result,
        termination=term,
        plies=len(moves),
        moves=" ".join(moves),
        clocks=history,
        seconds=round(time.monotonic() - t0, 1),
    )


class Context:
    def __init__(self, engine, search):
        self.engine, self.search = engine, search
        self.local = threading.local()

    def uci(self, kind, elo):
        """This thread's engine for (kind, elo), started on first use."""
        pool = self.local.__dict__.setdefault("uci", {})
        if (kind, elo) in pool:
            pool[kind, elo] = pool.pop((kind, elo))  # most recently used last
        else:
            if (
                len(pool) >= 4
            ):  # bound each thread's processes: close the least recently used
                pool.pop(next(iter(pool))).close()
            seed = threading.get_ident() % 2**31 ^ os.getpid()
            pool[kind, elo] = stockfish(elo) if kind == "sf" else maia3(elo, seed)
        return pool[kind, elo]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks")
    p.add_argument("out")
    p.add_argument(
        "--book", default=None, help="openings.json (default: next to tasks)"
    )
    p.add_argument(
        "--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--games", type=int, default=2, help="concurrent games")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    tasks = [json.loads(line) for line in open(a.tasks)][a.shard :: a.shards]
    book = json.loads(
        Path(a.book or Path(a.tasks).with_name("openings.json")).read_text()
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {
        json.loads(line)["id"]
        for f in out.glob("*.jsonl")
        for line in open(f)
        if line.strip()
    }
    tasks = [t for t in tasks if t["id"] not in done]
    if a.limit:
        tasks = tasks[: a.limit]
    print(
        f"shard {a.shard}/{a.shards}: {len(tasks)} games to play, {len(done)} done",
        flush=True,
    )
    if not tasks:
        return
    engine = search = None
    specs = {t[c] for t in tasks for c in ("white", "black")}
    if any(s.startswith("allie") for s in specs):
        import torch
        from allie.lichess.model import Model

        torch.set_num_threads(a.threads)
        t0 = time.monotonic()
        engine = Engine(Model(a.model, a.device))
        print(f"model loaded in {time.monotonic() - t0:.0f} s", flush=True)
        if any(re.fullmatch(r"allie-s\d+-\d+", s) for s in specs):
            from allie.lichess.tree import Coverage

            search = Coverage()
    ctx = Context(engine, search)
    lock = threading.Lock()
    path = (
        out
        / f"{Path(a.tasks).stem}-{a.shard:03d}-{os.environ.get('SLURM_JOB_ID', os.getpid())}.jsonl"
    )

    def run(task):
        try:
            r = play(task, ctx, book)
        except Exception as e:  # noqa: BLE001 - log and move on; the task stays undone
            print(f"{task['id']}: {e!r}", flush=True)
            return
        with lock, open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(
            f"{r['id']} {r['result']} {r['termination']} {r['plies']} plies {r['seconds']} s",
            flush=True,
        )

    with ThreadPoolExecutor(a.games) as pool:
        list(pool.map(run, tasks))
    if engine:
        print(f"forwards {engine.forwards} tokens {engine.tokens}", flush=True)
        engine.close()


if __name__ == "__main__":
    main()
