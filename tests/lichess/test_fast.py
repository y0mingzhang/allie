import threading

import pytest
import torch

from allie.lichess import fast
from allie.lichess.engine import Engine
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import HEADER

from .test_model import inputs
from .test_tokens import random_game

try:
    fast.library()
except Exception as e:  # noqa: BLE001 - no compiler: the fallback is the reference itself
    pytest.skip(f"no fast kernels: {e}", allow_module_level=True)


def play(model, games, lengths, steps=10):
    """Logits of a prefill of every game, then `steps` decode steps of one or two tokens."""
    caches = [Cache(model, 8) for _ in games]  # also exercises growth
    out = [
        step(
            model,
            [
                (c, x[0][:n], x[1][:n], x[2][:n])
                for c, x, n in zip(caches, games, lengths)
            ],
        )
    ]
    n = list(lengths)
    for r in range(steps):
        k = 2 if r % 3 == 0 else 1
        out.append(step(model, [(c, x[0][a : a + k], x[1][a : a + k], x[2][a : a + k])
                                for c, x, a in zip(caches, games, n)]))  # fmt: skip
        n = [a + k for a in n]
    return torch.cat(out)


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("keep", [None, 2])
def test_fast_matches_reference(tiny_path, tiny, int8, keep):
    """Fast and PyTorch BF16 paths differ by BF16 rounding only: the fast one is no further
    from FP32 than the reference is (a routing near-tie can flip in either)."""
    games = [inputs(random_game(s, 30 + 7 * s)) for s in range(3)]
    lengths = [HEADER + 5, HEADER, HEADER + 9]
    kw = dict(dtype=torch.bfloat16, int8=int8, active_experts=keep)
    ref = play(Model(tiny_path, backend="torch", **kw), games, lengths)
    got = play(Model(tiny_path, backend="fast", threads=3, **kw), games, lengths)
    hi = play(Model(tiny_path, dtype=torch.float32, active_experts=keep), games, lengths)
    assert (got - hi).abs().mean() < 1.25 * (ref - hi).abs().mean() + 1e-3
    assert (got - hi).abs().max() < 2 * (ref - hi).abs().max() + 0.02
    p, q = (torch.softmax(z[:, 378:2346], -1) for z in (got, ref))
    assert (p - q).abs().max() < 1e-3


def test_backend_choice(tiny_path, tiny):
    assert tiny.fast is None  # FP32: the reference
    assert Model(tiny_path, dtype=torch.bfloat16).fast is not None  # CPU BF16 default
    assert Model(tiny_path, dtype=torch.bfloat16, backend="torch").fast is None
    with pytest.raises(AssertionError):
        Model(tiny_path, dtype=torch.float32, backend="fast")


def test_fast_engine_and_takeback(tiny_path):
    """Concurrent games batched by the engine, and a rewound cache, match fresh prefills."""
    m = Model(tiny_path, dtype=torch.bfloat16, int8=True, threads=2)
    engine = Engine(m)
    games = [inputs(random_game(10 + s, 24)) for s in range(4)]
    out = [None] * len(games)

    def run(j):
        c, x = Cache(m), games[j]
        out[j] = [engine.extend(c, x[0][:n], x[1][:n], x[2][:n]) if n == HEADER
                  else engine.extend(c, x[0][n - 1 : n], x[1][n - 1 : n], x[2][n - 1 : n])
                  for n in range(HEADER, len(x[0]) + 1)]  # fmt: skip
        c.truncate(HEADER + 3)  # takeback, then the same moves again
        n = len(x[0])
        out[j].append(
            engine.extend(
                c, x[0][HEADER + 3 : n], x[1][HEADER + 3 : n], x[2][HEADER + 3 : n]
            )
        )

    threads = [threading.Thread(target=run, args=(j,)) for j in range(len(games))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    engine.close()
    assert engine.forwards < sum(len(x[0]) - HEADER + 2 for x in games)  # some batching
    for x, zs in zip(games, out):
        for n, z in zip([*range(HEADER, len(x[0]) + 1), len(x[0])], zs):
            fresh = step(m, [(Cache(m), x[0][:n], x[1][:n], x[2][:n])])[0]
            torch.testing.assert_close(z, fresh, atol=0.1, rtol=0)


def test_profile_and_bandwidth(tiny_path):
    m = Model(tiny_path, dtype=torch.bfloat16, threads=2)
    m.fast.profile()
    x = inputs(random_game(3, 10))
    step(m, [(Cache(m), *x)])
    t = m.fast.profile(False)
    assert set(t) == set(fast.PHASES) and sum(t.values()) > 0
    assert fast.bandwidth(1, 0.01, 1) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_graphs_match_eager(tiny_path, dtype):
    """Decode steps replayed from CUDA graphs on pooled caches equal the eager path's."""
    games = [inputs(random_game(s, 30 + 7 * s)) for s in range(3)]
    games = [tuple(t.cuda() for t in x) for x in games]
    lengths = [HEADER + 5, HEADER, HEADER + 9]
    ref = Model(tiny_path, "cuda", dtype, backend="torch")
    m = Model(tiny_path, "cuda", dtype)
    assert m.graphs is not None and ref.graphs is None
    tol = dict(atol=2e-4, rtol=0) if dtype == torch.float32 else dict(atol=0.1, rtol=0)
    torch.testing.assert_close(play(m, games, lengths, 20), play(ref, games, lengths, 20), **tol)
    assert len(m.graphs.graphs) >= 2  # replays happened: batches of three, then one game
    one = play(m, games[:1], lengths[:1], 5)
    torch.testing.assert_close(one, play(ref, games[:1], lengths[:1], 5), **tol)
