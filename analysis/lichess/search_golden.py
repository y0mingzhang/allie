"""Golden outputs of today's searches, for the Rust port's equality gate: at N live-game positions (several plies of
each PGN), the coverage and KL searches' results and the per-leaf logits, saved to an .npz. The C++ forward is
deterministic, so a faithful port must reproduce every array exactly (`compare` checks a second run or the port).

usage: search_golden.py --model DIR --pgns 'GLOB' --out FILE [--positions 200] [--budgets 32,128] [--searchers coverage,kl]
       [--threads 8] [--compare FILE]
"""

import argparse
import glob
import json
import sys
import time

import numpy as np
import torch

from allie.lichess import tree
from allie.lichess.engine import Engine, Game
from allie.lichess.model import Model
from allie.lichess.tokens import MOVE_ID

from search_profile import read_pgn


def positions(files, n, plies=(12, 24, 36, 48, 60)):
    out = []
    for f in files:
        g = read_pgn(f, 10**6)
        for ply in plies:
            if ply + 2 <= len(g["moves"]):
                out.append((f, ply))
    rng = np.random.default_rng(0)
    rng.shuffle(out)
    return sorted(out[:n])


def load_game(engine, g):
    game = Game(engine, g["white"], g["black"], g["base"], g["inc"], g["speed"])
    moves, clocks = g["moves"], g["clocks"]
    for k in range(1, len(moves) + 1):
        j = k - 1
        own, other = clocks[j], clocks[j - 1] if j else None
        wt, bt = (own, other) if j % 2 == 0 else (other, own)
        game.update(moves[:k], wt, bt)
    game.sync()
    return game


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pgns", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=200)
    p.add_argument("--budgets", default="32,128")
    p.add_argument("--searchers", default="coverage,kl")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--compare")
    p.add_argument("--chunk", type=int, help="C++ path: leaves per step (today 8); the kernels' sums depend on the step's size")
    a = p.parse_args()
    torch.set_num_threads(1)
    m = Model(a.model, "cpu", torch.bfloat16, None, True, "fast", a.threads)
    assert m.fast is not None
    engine = Engine(m)
    searchers = {"coverage": tree.Coverage(), "kl": tree.KL()}
    if a.chunk:
        tree.Nodes.fast = lambda self, ids, chunk=a.chunk: tree.Nodes.copied(self, ids, chunk)
    leaf_logits, handles_seen, inner = [], [], tree.Nodes.__call__

    def recorded(self_, handles):
        z = inner(self_, handles)
        leaf_logits.append(np.asarray(z, np.float64).copy())
        handles_seen.append(np.asarray(handles, np.int64).copy())
        return z

    tree.Nodes.__call__ = recorded
    out, t0, n_leaves = {}, time.perf_counter(), 0
    for i, (f, ply) in enumerate(positions(sorted(glob.glob(a.pgns)), a.positions)):
        game = load_game(engine, read_pgn(f, ply))
        if game.board.is_game_over() or game.board.legal_moves.count() < 2:
            continue
        key = f"{f.split('/')[-1][:-4]}@{ply}"
        out[key + "/root_logits"] = game.sync().double().numpy()
        out[key + "/tokens"] = np.array(game.tokens, np.int64)
        out[key + "/features"] = np.array(game.features(), np.float32)
        for name in a.searchers.split(","):
            for budget in map(int, a.budgets.split(",")):
                leaf_logits.clear(), handles_seen.clear()
                res = searchers[name](game, budget)
                assert res is not None
                tag = f"{key}/{name}/{budget}"
                out[tag + "/moves"] = np.array(
                    [MOVE_ID[mv] for mv in res[0]], np.int32
                )
                for j, part in enumerate(
                    ["probabilities", "prior", "q"]
                    if name == "coverage"
                    else ["prior", "q"]
                ):
                    out[f"{tag}/{part}"] = np.asarray(res[1 + j], np.float64)
                z = np.concatenate(leaf_logits) if leaf_logits else np.zeros((0, 2432))
                out[tag + "/leaf_logits"] = z.astype(np.float32)
                out[tag + "/handles"] = np.concatenate(handles_seen) if handles_seen else np.zeros((0, 4), np.int64)
                out[tag + "/calls"] = np.array([len(h) for h in handles_seen], np.int64)
                n_leaves += len(z)
        if i % 20 == 0:
            print(
                json.dumps(
                    dict(
                        done=i,
                        key=key,
                        leaves=n_leaves,
                        seconds=round(time.perf_counter() - t0, 1),
                    )
                ),
                flush=True,
            )
    np.savez_compressed(a.out, **out)
    print(
        json.dumps(
            dict(
                arrays=len(out),
                leaves=n_leaves,
                seconds=round(time.perf_counter() - t0, 1),
                out=a.out,
            )
        ),
        flush=True,
    )
    if a.compare:
        ref = np.load(a.compare)
        bad = [
            k
            for k in out
            if k not in ref
            or out[k].shape != ref[k].shape
            or not np.array_equal(out[k], ref[k])
        ]
        by = {}
        for k in bad:
            by[k.split("/")[-1]] = by.get(k.split("/")[-1], 0) + 1
        searches = {k.rsplit("/", 1)[0] for k in out if k.endswith("/moves")}
        differing = {k.rsplit("/", 1)[0] for k in bad if k.endswith(("/probabilities", "/q", "/prior", "/moves"))}
        gaps = [np.abs(out[k] - ref[k]).max() for k in bad if k.endswith(("/probabilities", "/q")) and out[k].shape == ref[k].shape]
        print(json.dumps(dict(compared=len(out), differing=len(bad), by_array=by, searches=len(searches), searches_differing=len(differing),
                              moves_differing=sum(k.endswith("/moves") for k in bad), max_output_gap=float(max(gaps)) if gaps else 0.0, first=bad[:6])), flush=True)
    engine.close()


if __name__ == "__main__":
    sys.exit(main())
