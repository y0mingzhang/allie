"""The Rust engine's path items against the golden C++ searches (the .npz search_golden.py wrote before the C++
path was deleted): each position's game is
rebuilt, its root logits compared, and every recorded call's handles replayed through tree.Nodes on the Rust backend
(Nodes.leaves: Leaf items), in the golden's chunks of 8 nodes a step (the kernels' partial sums depend on the step's token
count, so a bitwise match needs the golden's batches); the leaf logits must match bit for bit. --searches: also the
searches themselves as the bot runs them (every call one step) against the golden's outputs: how many come out
identical, and the largest difference.

usage: rust_leaves.py --model DIR --golden FILE --pgns DIR [--threads 8] [--positions N] [--chunk 8] [--searches]
       [--out FILE]
"""

import argparse
import json
import sys
import time

import numpy as np
import torch

from allie.lichess import tree
from allie.lichess.engine import Engine
from allie.lichess.model import Model
from allie.lichess.tokens import MOVE_ID

from search_profile import load_game, read_pgn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--pgns", required=True, help="directory of the golden's PGNs")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--positions", type=int)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--searches", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(1)
    g = np.load(a.golden)
    keys = sorted({k.split("/")[0] for k in g.files})[: a.positions]
    m = Model(a.model, "cpu", torch.bfloat16, None, True, "rust", a.threads)
    engine = Engine(m)
    tot = dict(
        positions=0,
        roots_differing=0,
        searches=0,
        leaves=0,
        leaves_differing=0,
        max_abs=0.0,
        isa=m.fast.isa,
        threads=m.fast.threads,
    )
    if a.searches:
        tot |= dict(outputs=0, outputs_identical=0, outputs_same_moves=0, outputs_max_abs=0.0, by={})
    searchers = {"coverage": tree.Coverage(), "kl": tree.KL()}
    t0, bad = time.perf_counter(), []
    for key in keys:
        stem, ply = key.rsplit("@", 1)
        game = load_game(engine, read_pgn(f"{a.pgns}/{stem}.pgn", int(ply)))
        assert game.tokens == g[f"{key}/tokens"].tolist(), key
        z = game.sync().double().numpy()
        tot["positions"] += 1
        tot["roots_differing"] += not np.array_equal(z, g[f"{key}/root_logits"])
        feats = np.array(game.features(), np.float32)
        for name, rule in (("coverage", "predicted"), ("kl", "zero")):
            for budget in (32, 128):
                tag = f"{key}/{name}/{budget}"
                if f"{tag}/handles" not in g:
                    continue
                handles, want = g[f"{tag}/handles"], g[f"{tag}/leaf_logits"]
                t = tree.Tree(
                    game,
                    game.sync(),
                    int(handles[:, 0].max()) + 1 if len(handles) else 1,
                )
                nodes = t.handles([game.tokens], [feats], rule)
                lo, n_bad, mx = 0, 0, 0.0
                for n in g[f"{tag}/calls"].tolist():
                    for c in range(0, n, a.chunk):
                        h = handles[lo + c : lo + min(c + a.chunk, n)]
                        z = nodes(h).astype(np.float32)
                        w = want[lo + c : lo + c + len(h)]
                        n_bad += int((z != w).any(1).sum())
                        mx = max(mx, float(np.abs(z - w).max()))
                    lo += n
                assert lo == len(want), tag
                tot["searches"] += 1
                tot["leaves"] += lo
                tot["leaves_differing"] += n_bad
                tot["max_abs"] = max(tot["max_abs"], mx)
                if n_bad:
                    bad.append(
                        dict(tag=tag, leaves=int(lo), differing=n_bad, max_abs=mx)
                    )
                if a.searches:
                    res = searchers[name](game, budget)
                    parts = ["probabilities", "prior", "q"] if name == "coverage" else ["prior", "q"]
                    got = [np.asarray(x, np.float64) for x in res[1:]]
                    ref = [g[f"{tag}/{x}"] for x in parts]
                    same = [MOVE_ID[mv] for mv in res[0]] == g[f"{tag}/moves"].tolist()
                    tot["outputs"] += 1
                    tot["outputs_same_moves"] += same
                    ok = same and all(np.array_equal(x, y) for x, y in zip(got, ref))
                    tot["outputs_identical"] += ok
                    # identical is expected exactly when no golden call ran in more than one chunk
                    fits = bool((g[f"{tag}/calls"] <= a.chunk).all())
                    b = tot["by"].setdefault(f"{name}/{budget}", dict(fit=0, fit_identical=0, split=0, split_identical=0))
                    b["fit" if fits else "split"] += 1
                    b["fit_identical" if fits else "split_identical"] += ok
                    if same:
                        tot["outputs_max_abs"] = max(tot["outputs_max_abs"], *(float(np.abs(x - y).max()) for x, y in zip(got, ref)))
        if tot["positions"] % 20 == 0:
            print(
                json.dumps(tot | dict(seconds=round(time.perf_counter() - t0, 1))),
                flush=True,
            )
    tot["seconds"] = round(time.perf_counter() - t0, 1)
    tot["first_bad"] = bad[:10]
    print(json.dumps(tot), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(
                json.dumps(tot | dict(model=a.model, golden=a.golden, chunk=a.chunk))
                + "\n"
            )
    engine.close()
    return 1 if tot["leaves_differing"] or tot["roots_differing"] else 0


if __name__ == "__main__":
    sys.exit(main())
