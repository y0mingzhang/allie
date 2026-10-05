"""The Rust engine (fastrs.RustFast) against the C++ kernels (fast.Fast): bitwise equal logits and cache
rows on the same ISA, for each of the three variants (the C++ rebuilt for the matching -march)."""

import multiprocessing
import threading

import pytest
import torch

from allie.lichess import fast
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import HEADER

from .test_model import inputs
from .test_tokens import random_game

try:
    fast.library()
except Exception as e:  # noqa: BLE001
    pytest.skip(f"no fast kernels: {e}", allow_module_level=True)
allie_fast = pytest.importorskip("allie_fast")
if not hasattr(allie_fast, "Engine"):
    pytest.skip("allie_fast built without Engine", allow_module_level=True)

# C++ -march per Rust variant: native for the CPU's best; haswell = AVX2 + FMA; x86-64 = no AVX
MARCH = dict(native=None, avx2="-march=haswell", scalar="-march=x86-64")
GAMES = [inputs(random_game(s, 36 + s % 5)) for s in range(64)]


def cxx(tiny_path, march, monkeypatch, **kw):
    """A C++ Model built for `march` (fast.library caches one build per process: cleared around it)."""
    if march:
        monkeypatch.setenv("ALLIE_MARCH", march)
    else:
        monkeypatch.delenv("ALLIE_MARCH", raising=False)
    fast.library.cache_clear()
    try:
        return Model(tiny_path, dtype=torch.bfloat16, backend="fast", **kw)
    finally:
        fast.library.cache_clear()
        monkeypatch.delenv("ALLIE_MARCH", raising=False)


def rust(tiny_path, isa, monkeypatch, **kw):
    if isa == "native":
        monkeypatch.delenv("ALLIE_RUST_ISA", raising=False)
    else:
        monkeypatch.setenv("ALLIE_RUST_ISA", isa)
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", **kw)
    monkeypatch.delenv("ALLIE_RUST_ISA", raising=False)
    return m


def replay(model, n=64):
    """Fresh caches for n games, prefilled to different lengths in one batch, then single-token steps
    at batch sizes 1, 2, 3, 8, 24, 64 (one new token a game: the fused attention path), then a mixed
    step of one or two tokens a game. Returns every step's logits and the caches' filled rows."""
    caches = [Cache(model, 8) for _ in range(n)]
    pos = [HEADER + (j % 7) * 2 for j in range(n)]
    item = lambda j, a, b: (
        caches[j],
        GAMES[j][0][a:b],
        GAMES[j][1][a:b],
        GAMES[j][2][a:b],
    )
    out = [step(model, [item(j, 0, pos[j]) for j in range(n)])]
    for b in (1, 2, 3, 8, 24, 64):
        b = min(b, n)
        out.append(step(model, [item(j, pos[j], pos[j] + 1) for j in range(b)]))
        for j in range(b):
            pos[j] += 1
    out.append(step(model, [item(j, pos[j], pos[j] + 1 + j % 2) for j in range(n)]))
    for j in range(n):
        pos[j] += 1 + j % 2
    rows = [
        (c.k[:, :, : c.n].clone(), c.v[:, :, : c.n].clone(), c.e[: c.n].clone())
        for c in caches
    ]
    assert all(c.n == p for c, p in zip(caches, pos))
    return out, rows


def same(a, b):
    (oa, ra), (ob, rb) = a, b
    for i, (x, y) in enumerate(zip(oa, ob)):
        bad = (x != y).sum().item()
        assert bad == 0, (
            f"step {i}: {bad} of {x.numel()} logits differ, max {(x - y).abs().max().item()}"
        )
    for j, (x, y) in enumerate(zip(ra, rb)):
        for name, s, t in zip("kve", x, y):
            bad = (s != t).sum().item()
            assert bad == 0, (
                f"game {j} cache {name}: {bad} of {s.numel()} elements differ"
            )


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("isa", ["native", "avx2", "scalar"])
def test_rust_matches_cxx_bitwise(tiny_path, isa, int8, monkeypatch):
    ref = cxx(tiny_path, MARCH[isa], monkeypatch, int8=int8, threads=3)
    got = rust(tiny_path, isa, monkeypatch, int8=int8, threads=3)
    want = {b"avx512": "avx512", b"avx2": "avx2", b"generic": "scalar"}[ref.fast.lib.allie_isa()]
    assert got.fast.isa == want and (isa == "native" or want == isa), (got.fast.isa, want, isa)
    same(replay(got), replay(ref))


def test_rust_thread_count_does_not_change_results(tiny_path, monkeypatch):
    """Every output element is one thread's fixed-order reduction: 1 and 3 threads agree bitwise."""
    a = rust(tiny_path, "native", monkeypatch, int8=True, threads=1)
    b = rust(tiny_path, "native", monkeypatch, int8=True, threads=3)
    assert a.fast.threads == 1 and b.fast.threads == 3
    same(replay(a, 9), replay(b, 9))


def test_rust_rejects_bad_inputs(tiny_path):
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    ids, feats, boards = inputs(random_game(4, 6))
    for bad in (ids.clone().fill_(2432), ids.clone().fill_(-1)):
        with pytest.raises(ValueError, match="vocabulary"):
            step(m, [(Cache(m), bad, feats, boards)])
    b = boards.clone()
    b[:, 65] = 16
    with pytest.raises(ValueError, match="board"):
        step(m, [(Cache(m), ids, feats, b)])
    with pytest.raises(ValueError, match="an item"):
        step(
            m,
            [
                (Cache(m), ids[:1], feats[:0], boards[:1]),
                (Cache(m), ids[1:2], feats[:2], boards[1:2]),
            ],
        )
    with pytest.raises(ValueError, match="another model"):
        step(
            m,
            [
                (
                    Cache(Model(tiny_path, dtype=torch.bfloat16, backend="torch")),
                    ids,
                    feats,
                    boards,
                )
            ],
        )
    # the engine's own span check: a cache claiming more rows than its capacity
    e = m.fast.native()
    c = Cache(m)
    meta = [c.capacity, 1, c.capacity, 0]
    assert e.step(1, 1, ids[:1].data_ptr(), feats[:1].data_ptr(), boards[:1].data_ptr(), meta,
                  [c.k.data_ptr(), c.v.data_ptr(), c.e.data_ptr()], torch.empty(1, 2432).data_ptr()) == 3  # fmt: skip


def test_rust_profile_threads_isa(tiny_path):
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    assert m.fast.threads == 2 and m.fast.handle.threads == 2
    assert m.fast.isa in ("avx512", "avx2", "scalar")
    m.fast.profile()
    x = inputs(random_game(3, 10))
    step(m, [(Cache(m), *x)])
    t = m.fast.profile(False)
    assert list(t) == list(fast.PHASES) and len(t) == 17 and sum(t.values()) > 0


def test_rust_concurrent_steps(tiny_path):
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    games = [inputs(random_game(20 + s, 30)) for s in range(4)]
    want = [step(m, [(Cache(m), *x)])[0] for x in games]
    got = [None] * len(games)
    go = threading.Barrier(len(games))

    def run(j):
        go.wait()
        got[j] = step(m, [(Cache(m), *games[j])])[0]

    ts = [threading.Thread(target=run, args=(j,)) for j in range(len(games))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for a, b in zip(got, want):
        assert torch.equal(a, b)


FORKED = {}


def _forked_step():
    m, x = FORKED["model"], FORKED["x"]
    return step(m, [(Cache(m), *x)])[0]


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_rust_after_fork(tiny_path):
    """A forked child has none of the parent's pool threads: its first step makes its own."""
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    x = inputs(random_game(3, 10))
    want = step(m, [(Cache(m), *x)])[0]
    FORKED.update(model=m, x=x)
    with (
        m.fast.lock
    ):  # another thread mid-step at the fork: the child must not wait for it
        with multiprocessing.get_context("fork").Pool(1) as pool:
            got = pool.apply_async(_forked_step).get(timeout=120)
    assert torch.equal(got, want)
