"""Bot-path parity: allie.lichess.tree.KL (the bot's model on the fast cpu backend, int8) against a gpu_kl.py run
(training checkpoint, bf16) on the same positions and budget: per position the root Q of every searched move,
the trees' sizes, and the tilted outputs' total variation.

python bot_check.py RUN --n 24 --budget 256 [--own 4 --opp 0 --clock zero]
"""

import argparse
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
from allie.lichess.tree import KL  # noqa: E402
from allie.search import kl  # noqa: E402
from calib_search_cpu import Position  # noqa: E402

import common as c  # noqa: E402
import evaluate as ev  # noqa: E402

MODEL = "/data/group_data/dei-group/yimingz3/allie/lichess/allie-2.0-annealed"
POS = c.CALIB / "positions-human.npz"
INC = {v: int(k) for k, v in INCREMENTS_ID.items() if k.isdigit()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--own", type=float, default=5.0)
    p.add_argument("--opp", type=float, default=5.0)
    p.add_argument("--soft", action="store_true")
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--clock", default="zero")
    p.add_argument(
        "--beta", type=float, default=8.0, help="output tilt for the total variation"
    )
    a = p.parse_args()
    torch.set_num_threads(4)
    F = ev.load_forest(c.OUT / "pikl" / a.run)
    keep = F["cost"] <= a.budget
    tilt = dict(own=a.own, opp=a.opp, soft=a.soft, kappa=a.kappa)
    V, _, _ = kl.backup(F, keep, **tilt)
    z = np.load(POS)
    off, tokens, feats, meta = z["offsets"], z["tokens"], z["feats"], z["meta"]
    engine = Engine(Model(MODEL, int8=True, backend="fast", threads=4))
    search = KL(**tilt, clock_rule=a.clock)
    rows = np.linspace(0, len(F["index"]) - 1, a.n).astype(int)
    tv, corr, agree = [], [], []
    for r in rows:
        i = F["index"][r]
        game = Position(
            engine,
            tokens[off[i] : off[i + 1]],
            feats[off[i] : off[i + 1]],
            int(meta[i, 9]),
        )
        game.inc, game.used = INC.get(int(tokens[off[i] + 2])), []
        t0 = time.monotonic()
        moves, prior, q = search(game, a.budget)
        secs = time.monotonic() - t0
        kids = np.flatnonzero(keep & (F["parent"] == F["roots"][r]))
        gq = {int(F["token"][k]): -V[k] for k in kids}
        gv = F["wdl"][F["roots"][r]]
        tok = [int(t) for t in moves]
        g = np.array([gq.get(t, gv[0] - gv[2]) for t in tok])
        both = np.array([t in gq for t in tok])
        lp = np.log(np.maximum(prior, 1e-300))
        pb, pg = (
            np.exp(x - x.max()) / np.exp(x - x.max()).sum()
            for x in (lp + a.beta * q, lp + a.beta * g)
        )
        tv.append(0.5 * np.abs(pb - pg).sum())
        corr.append(np.corrcoef(q[both], g[both])[0, 1] if both.sum() > 2 else np.nan)
        agree.append(int(np.argmax(pb) == np.argmax(pg)))
        print(f"pos {i}: {secs:.1f}s searched {both.sum()}/{len(tok)} (gpu) Q corr {corr[-1]:.3f} "
              f"max|dQ| {np.abs(q - g)[both].max():.3f} TV(beta {a.beta:g}) {tv[-1]:.3f}", flush=True)  # fmt: skip
    print(
        f"mean TV {np.mean(tv):.3f}, median Q corr {np.nanmedian(corr):.3f}, top move agreement {np.mean(agree):.2f}"
    )
    engine.close()


if __name__ == "__main__":
    main()
