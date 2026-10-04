"""CPU latency of search in the bot's own code path (allie.lichess Tree on its Model): coverage
(batched root branches) and Allie-paper MCTS (one leaf per iteration), per budget, at a few plies
of one game. Writes JSON.

python calib_latency.py OUT.json [--threads 8]
"""

import argparse
import json
import time

import numpy as np
import torch
from allie.lichess.engine import Engine, Game
from allie.lichess.model import Model
from allie.lichess.tree import Coverage, Tree
from allie.search import Search
from allie.search.native import load

from play import MODEL

LINE = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8 e1c1 b8d7 g2g4 b7b5 g4g5 b5b4 c3e2 f6e8 h2h4 a6a5".split()


def mcts(game, budget, cpuct=1.25):
    """The "allie" tree on the game's cache, keeping visits (calib_search.mcts on the bot's model)."""
    z = game.sync()
    feats = np.array(game.features(), np.float32)

    def run():
        tree = Tree(game, z, capacity=4 * budget + 256)
        bridge = tree.handles([list(game.tokens)], [feats])
        t = load().Allie([list(game.tokens)], bridge.root_logits, [budget], [cpuct])
        t.first_prior = t.preserve_depth = True
        Search._advance(t, bridge)
        return t.summaries()[0]

    return game.engine.run(run)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    engine = Engine(Model(MODEL))
    cov = Coverage()
    rows = []
    clocks = [180.0, 180.0]
    game = Game(engine, 2000, 2000, 180, 2)
    for ply, m in enumerate(LINE):
        if ply >= 2:
            clocks[ply % 2] -= 3
        game.update(LINE[: ply + 1], *clocks)
        if ply in (9, 17, 27):
            game.sync()
            for b in (8, 32, 128):
                t = time.perf_counter()
                cov(game, b) if b in (8, 25, 128) else None
                tc = time.perf_counter() - t
                t = time.perf_counter()
                mcts(game, b)
                tm = time.perf_counter() - t
                rows.append(
                    dict(
                        ply=ply,
                        budget=b,
                        coverage_s=round(tc, 3) if b != 32 else None,
                        mcts_s=round(tm, 3),
                    )
                )
                print(rows[-1], flush=True)
    t = time.perf_counter()
    for _ in range(5):
        game.cache.truncate(game.cache.n - 1)
        game.logits = None
        game.sync()
    rows.append(dict(single_token_forward_s=round((time.perf_counter() - t) / 5, 3)))
    print(rows[-1])
    engine.close()
    with open(a.out, "w") as f:
        json.dump(dict(threads=a.threads, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
