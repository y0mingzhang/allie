"""Time allie.lichess's model per new token, as a live game uses it: one token appended to a
game's cache (batch 1), or one token each for B games at once.

usage: python analysis/lichess/speed.py --model DIR [--device cpu] [--int8] [--active-experts K]
       [--backend fast|torch] [--threads N] [--batch 1,16] [--profile]
"""

import argparse
import json
import time

import numpy as np
import torch

from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import HEADER, MOVE_ID, START, advance, features, header


def game(n=80, seed=0):
    import chess

    rng, moves = np.random.default_rng(seed), []
    while len(moves) < n:  # a random game that ends early is replaced
        board, moves = chess.Board(), []
        while len(moves) < n and not board.is_game_over():
            m = list(board.legal_moves)[rng.integers(board.legal_moves.count())]
            board.push(m)
            moves.append(m.uci())
    tokens = header(180, 2, 1500, 1500) + [MOVE_ID[m] for m in moves]
    clocks = [180 - k for k in range(len(moves))]
    feats = [[-1] * 3] * (HEADER - 1) + [
        features(k, 180, 2, clocks) for k in range(len(moves) + 1)
    ]
    boards = [START] * HEADER
    for t in tokens[HEADER:]:
        boards.append(advance(boards[-1], t))
    b = np.frombuffer(b"".join(boards), np.uint8).reshape(-1, 68)
    return (
        torch.tensor(tokens),
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(b),
    )


def traffic(m, b, n):
    """Bytes a step of b games, each n tokens into its cache, reads: every matrix but the
    experts, the experts the batch routes to (expected number of distinct ones), embedding rows,
    and the caches' keys and values."""
    size = lambda k: m.w[k].numel() * m.w[k].element_size() + (
        m.scales[k].numel() * m.scales[k].element_size() if k in m.scales else 0
    )
    rows = ("embed", "embed2", "value_embed", "cos", "sin")
    dense = sum(size(k) for k in m.w if k.split(".")[-1] not in ("up", "down") and not k.startswith(rows))
    e, keep = m.config["experts"], m.keep
    experts = sum(size(k) for k in m.w if k.split(".")[-1] in ("up", "down"))
    experts *= 1 - (1 - keep / e) ** b
    item = torch.tensor([], dtype=m.dtype).element_size()
    embed = b * (2 + m.ve) * m.width * item
    kv = b * 2 * m.layers * m.heads * n * m.head_dim * item
    return dense + experts + embed + kv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--active-experts", type=int)
    p.add_argument("--backend", choices=["fast", "torch"])
    p.add_argument("--threads", type=int)
    p.add_argument("--batch", default="1")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    m = Model(a.model, a.device, torch.bfloat16, a.active_experts, a.int8, a.backend, a.threads)
    backend = "fast" if m.fast or m.graphs else "torch"
    threads = m.fast.threads if m.fast else torch.get_num_threads()
    out = dict(device=str(m.device), backend=backend, int8=a.int8, experts=m.keep, threads=threads,
               load_seconds=round(time.perf_counter() - t0, 1))  # fmt: skip
    sync = torch.cuda.synchronize if m.device.type == "cuda" else lambda: None
    caches = None
    for b in map(int, a.batch.split(",")):
        caches = None  # the last batch's gpu pool slots back first
        games = [game(80 + a.steps + 21, s) for s in range(b)]
        caches = [Cache(m) for _ in games]
        step(m, [(c, x[0][:80], x[1][:80], x[2][:80]) for c, x in zip(caches, games)])
        times = []
        for i in range(80, 80 + a.steps):
            sync()
            t = time.perf_counter()
            step(
                m,
                [
                    (c, x[0][i : i + 1], x[1][i : i + 1], x[2][i : i + 1])
                    for c, x in zip(caches, games)
                ],
            )
            sync()
            times.append(1000 * (time.perf_counter() - t))
        q = np.percentile(times[5:], [50, 90])
        gb = traffic(m, b, 80 + a.steps // 2) / 1e9
        out[f"batch{b}_ms"] = dict(median=round(q[0], 1), p90=round(q[1], 1), gb=round(gb, 3),
                                   gb_per_s=round(gb / q[0] * 1000, 1))  # fmt: skip
    if a.profile and m.fast:
        c, x = caches[0], games[0]
        m.fast.profile()
        for i in range(80 + a.steps, 80 + a.steps + 20):
            step(m, [(c, x[0][i : i + 1], x[1][i : i + 1], x[2][i : i + 1])])
        out["phase_ms"] = {k: round(50 * v, 3) for k, v in m.fast.profile(False).items()}
    elif a.profile:
        from torch.profiler import ProfilerActivity, profile

        c, x = caches[0], games[0]
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            i = 80 + a.steps
            step(m, [(c, x[0][i : i + 1], x[1][i : i + 1], x[2][i : i + 1])])
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
    import resource

    out["max_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
