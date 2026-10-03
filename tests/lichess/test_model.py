import threading

import numpy as np
import torch
from torch.nn import functional as F

from allie.data.vocab import MOVE_ID
from allie.lichess.engine import Engine, Game
from allie.lichess.model import Cache, Model, step, swiglu
from allie.lichess.tokens import HEADER, START, advance, features, header

from .test_tokens import random_game


def inputs(moves, base=180, inc=2, elo=(1500, 1700)):
    tokens = header(base, inc, *elo) + [MOVE_ID[m] for m in moves]
    clocks = [base - k // 2 for k in range(len(moves))]
    feats = [[-1] * 3] * (HEADER - 1) + [
        features(k, base, inc, clocks) for k in range(len(moves) + 1)
    ]
    boards = [START] * HEADER
    for t in tokens[HEADER:]:
        boards.append(advance(boards[-1], t))
    boards = np.frombuffer(b"".join(boards), np.uint8).reshape(-1, 68)
    return (
        torch.tensor(tokens),
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(boards),
    )


def prefill(model, x, n):
    return step(model, [(Cache(model), x[0][:n], x[1][:n], x[2][:n])])[0]


def test_incremental_matches_prefill(tiny):
    x = inputs(random_game(1, 40))
    cache, n = Cache(tiny, capacity=8), 0  # also exercises growth
    rng = np.random.default_rng(0)
    while n < len(x[0]):
        m = min(len(x[0]), n + (HEADER if n == 0 else int(rng.integers(1, 4))))
        z = step(tiny, [(cache, x[0][n:m], x[1][n:m], x[2][n:m])])[0]
        torch.testing.assert_close(z, prefill(tiny, x, m), atol=2e-5, rtol=0)
        n = m


def test_batched_matches_single(tiny):
    games = [inputs(random_game(s, 30 + 7 * s)) for s in range(3)]
    caches = [Cache(tiny) for _ in games]
    lengths = [HEADER + 5, HEADER, HEADER + 9]
    batched = step(
        tiny,
        [(c, x[0][:n], x[1][:n], x[2][:n]) for c, x, n in zip(caches, games, lengths)],
    )
    for z, x, n in zip(batched, games, lengths):
        torch.testing.assert_close(z, prefill(tiny, x, n), atol=2e-5, rtol=0)
    more = step(tiny, [(c, x[0][n : n + 2], x[1][n : n + 2], x[2][n : n + 2])
                       for c, x, n in zip(caches, games, lengths)])  # fmt: skip
    for z, x, n in zip(more, games, lengths):
        torch.testing.assert_close(z, prefill(tiny, x, n + 2), atol=2e-5, rtol=0)


def naive_moe(m, i, h, keep):
    w = m.w
    out = []
    for t in h:
        s = torch.sigmoid(F.linear(t.float() - w[f"{i}.mu"], w[f"{i}.router"]))
        idx = torch.topk(s + w[f"{i}.moe_bias"], m.topk).indices
        g = s[idx] * m.topk**0.5 / s[idx].sum()
        y = sum(g[j] * F.linear(swiglu(F.linear(t, w[f"{i}.up"][e])), w[f"{i}.down"][e])
                for j, e in enumerate(idx[:keep].tolist()))  # fmt: skip
        out.append(
            y
            + F.linear(swiglu(F.linear(t, w[f"{i}.shared_up"])), w[f"{i}.shared_down"])
        )
    return torch.stack(out)


def test_moe_matches_naive(tiny_path):
    h = torch.randn(13, 64)
    for keep in (None, 2):
        m = Model(tiny_path, dtype=torch.float32, experts=keep)
        for i in (1, 3):
            torch.testing.assert_close(m.mlp(i, h), naive_moe(m, i, h, keep or m.topk))


def test_engine_batches_games(tiny):
    engine = Engine(tiny)
    games = [inputs(random_game(10 + s, 24)) for s in range(6)]
    out = [None] * len(games)

    def run(j):
        c, x = Cache(tiny), games[j]
        out[j] = [engine.extend(c, x[0][:n], x[1][:n], x[2][:n]) if n == HEADER
                  else engine.extend(c, x[0][n - 1 : n], x[1][n - 1 : n], x[2][n - 1 : n])
                  for n in range(HEADER, len(x[0]) + 1)]  # fmt: skip

    threads = [threading.Thread(target=run, args=(j,)) for j in range(len(games))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert engine.forwards < sum(len(x[0]) - HEADER + 1 for x in games)  # some batching
    for x, zs in zip(games, out):
        for n, z in zip(range(HEADER, len(x[0]) + 1), zs):
            torch.testing.assert_close(z, prefill(tiny, x, n), atol=2e-5, rtol=0)


def test_takeback_rewinds(tiny):
    engine = Engine(tiny)
    moves = random_game(3, 20)
    a = Game(engine, 1500, 1500, 180, 2)
    a.update(moves, 170, 171)
    a.sync()
    a.update(moves[:15])
    b = Game(engine, 1500, 1500, 180, 2)
    b.update(moves[:15])
    torch.testing.assert_close(a.sync(), b.sync(), atol=2e-5, rtol=0)
    assert a.board.fen() == b.board.fen()


def test_int8_close_to_dequantized(tiny_path):
    q = Model(tiny_path, dtype=torch.bfloat16, int8=True)
    ref = Model(tiny_path, dtype=torch.bfloat16)
    for k, s in q.scales.items():
        ref.w[k] = (q.w[k].float() * s.float()[..., None]).bfloat16()
    x = inputs(random_game(5, 20))
    a, b = prefill(q, x, len(x[0])), prefill(ref, x, len(x[0]))
    assert (a - b).abs().max() < 0.05 * b.abs().max()


def test_reader_matches_safetensors(tiny_path):
    from safetensors.torch import load_file

    from allie.lichess.model import read

    ref = load_file(tiny_path / "model.safetensors")
    ours = dict(read(tiny_path / "model.safetensors"))
    assert ours.keys() == ref.keys()
    for k, v in ref.items():
        assert ours[k].dtype == v.dtype and torch.equal(ours[k], v)
