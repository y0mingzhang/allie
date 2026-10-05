"""allie.search.kl on the bot's own model and fast cpu backend (allie.lichess.tree's Tree on each position's game
cache, as the bot's KL searcher), one position at a time; chunk files as gpu_kl.py's.

python cpu_kl.py OUT --subset subset.npy --shard i --shards n --budget 1024 --own 5 --opp 5 --soft --kappa 0.5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [
    "/home/yimingz3/src/allie-wt-elo/analysis/elo_strength",
    "/home/yimingz3/src/allie-wt-seff/analysis/search_eff",
]
from allie.data.vocab import INCREMENTS_ID  # noqa: E402
from allie.lichess.engine import Engine  # noqa: E402
from allie.lichess.model import Model  # noqa: E402
from allie.lichess.tree import Tree  # noqa: E402
from allie.search import kl  # noqa: E402
from allie.search.native import from_prefix  # noqa: E402
from calib_search_cpu import Position  # noqa: E402

MODEL = "/data/group_data/dei-group/yimingz3/allie/lichess/allie-2.0-annealed"
POS = Path(
    "/data/group_data/dei-group/yimingz3/allie/results/recipe10x/elo-strength/calib/positions-human.npz"
)
INC = {v: int(k) for k, v in INCREMENTS_ID.items() if k.isdigit()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--subset", required=True)
    p.add_argument(
        "--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--own", type=float, default=5.0)
    p.add_argument("--opp", type=float, default=5.0)
    p.add_argument("--soft", action="store_true")
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--root", type=float, default=0.0)
    p.add_argument("--clock", default="zero")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    z = np.load(POS)
    off, tokens, feats, meta = z["offsets"], z["tokens"], z["feats"], z["meta"]
    sel = np.load(a.subset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(
        json.dumps(vars(a) | dict(views="0:" + a.clock), indent=1) + "\n"
    )
    engine = Engine(Model(MODEL, int8=True, backend="fast", threads=a.threads))
    for lo in list(range(0, len(sel), a.chunk))[a.shard :: a.shards]:
        dst = out / f"{lo:06d}.npz"
        if dst.exists():
            continue
        t0, parts = time.monotonic(), []
        idx = sel[lo : lo + a.chunk]
        for i in idx:
            game = Position(
                engine,
                tokens[off[i] : off[i + 1]],
                feats[off[i] : off[i + 1]],
                int(meta[i, 9]),
            )
            game.inc, game.used = INC.get(int(tokens[off[i] + 2])), []
            z0 = game.sync()

            def go():
                tree = Tree(game, z0, capacity=2 * a.budget + 256)
                bridge = tree.handles(
                    [game.tokens], [np.array(game.features(), np.float32)], a.clock
                )
                F = kl.Forest(
                    bridge,
                    [from_prefix(np.asarray(game.tokens))],
                    [len(game.tokens)],
                    tree.capacity,
                )
                kl.grow(F, a.budget, a.own, a.opp, a.soft, a.kappa, root=a.root)
                return F.dump()

            parts.append(engine.run(go))
        n0 = np.cumsum([0] + [len(d["parent"]) for d in parts])
        cat = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
        cat["parent"] = np.concatenate(
            [
                np.where(d["parent"] >= 0, d["parent"] + n0[j], -1)
                for j, d in enumerate(parts)
            ]
        )
        cat["owner"] = np.concatenate([d["owner"] + j for j, d in enumerate(parts)])
        cat["roots"] = n0[:-1]
        tmp = dst.with_suffix(".partial.npz")
        np.savez(tmp, index=idx, **cat)
        tmp.replace(dst)
        print(
            json.dumps(
                dict(
                    chunk=lo,
                    n=len(idx),
                    seconds=round(time.monotonic() - t0),
                    leaves=float(cat["leaves"].mean()),
                )
            ),
            flush=True,
        )
    engine.close()


if __name__ == "__main__":
    main()
