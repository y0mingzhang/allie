"""allie-bot: export | play | selfplay | bench."""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

import torch


def model(c):
    from .api import resolve
    from .model import Model

    if c.threads:
        torch.set_num_threads(c.threads)
    dtype = dict(bfloat16=torch.bfloat16, float32=torch.float32)[c.dtype]
    int8 = c.int8 if c.int8 is not None else c.device == "cpu" and dtype == torch.bfloat16
    return Model(resolve(c.model), c.device, dtype, c.active_experts or None, int8,
                 c.backend or None, c.threads or None)  # fmt: skip


def coverage(c):
    if not (c.play.mode == "strongest" and c.play.search):
        return None
    from .tree import Coverage

    return Coverage()


def watch(bot, path):
    while not os.path.exists(path):
        time.sleep(5)
    bot.drain()


def dump(out, path):
    if path:
        with open(path, "w") as f:
            f.write(json.dumps(out, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser(prog="allie-bot", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="training checkpoint -> safetensors export")
    e.add_argument("checkpoint")
    e.add_argument("out")
    run = sub.add_parser("play", help="play on Lichess (token: LICHESS_TOKEN)")
    run.add_argument("--drain-file", help="once this file exists, finish the games and exit")
    run.add_argument("--log", help="append the log here (reopened after a failed write; while "
                     "it fails, lines go to a node-local file); default stderr")  # fmt: skip
    run.epilog = "SIGUSR1 also drains: no new games, exit once the current ones end."
    s = sub.add_parser("selfplay", help="offline games on a local mock Lichess")
    s.add_argument("--games", type=int, default=2)
    s.add_argument("--opponent", choices=["self", "random"], default="self")
    s.add_argument("--base", type=int, default=180)
    s.add_argument("--increment", type=int, default=2)
    b = sub.add_parser("bench", help="per-move latency")
    b.add_argument("--moves", type=int, default=60)
    b.add_argument("--concurrent", type=int, default=1)
    for x in (run, s, b):
        x.add_argument("--config", required=True)
        x.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config key, e.g. play.search=25")  # fmt: skip
    for x in (s, b):
        x.add_argument("--out", help="write the result here (JSON)")
    a = p.parse_args()
    from .logs import setup

    listener = setup(getattr(a, "log", None))
    try:
        run_command(a)
    except Exception:
        logging.getLogger("allie").exception("allie-bot failed")  # while the writer still runs
        raise
    finally:
        listener.stop()


def run_command(a):
    if a.command == "export":
        from .export import main as export

        print(json.dumps(export(a.checkpoint, a.out), indent=1))
        return
    from .config import load

    c = load(a.config, a.set)
    if a.command == "play":
        from .bot import Bot
        from .client import Lichess
        from .engine import Engine

        token = os.environ.get("LICHESS_TOKEN")
        if not token:
            sys.exit("set LICHESS_TOKEN to the bot account's API token (bot:play)")
        bot = Bot(c, Lichess(token, c.url), Engine(model(c)), coverage(c))
        if a.drain_file:
            threading.Thread(target=watch, args=(bot, a.drain_file), daemon=True).start()
        signal.signal(signal.SIGUSR1, lambda *_: bot.drain())
        bot.run()
        bot.join()  # the games' last lines are logged before the log stops
        return
    from .selfplay import bench, selfplay

    if a.command == "selfplay":
        args = (a.games, a.opponent, a.base, a.increment)
        out = selfplay(c, model(c), coverage(c), *args)
        print(json.dumps(out["summary"], indent=1))
    else:
        out = bench(c, model(c), coverage(c), a.moves, a.concurrent)
        print(json.dumps(out, indent=1))
    dump(out, a.out)


if __name__ == "__main__":
    main()
