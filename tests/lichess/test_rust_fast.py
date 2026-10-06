"""The Rust engine (fastrs.RustFast): each of its three kernel variants against the reference and against the
bits of the C++ kernels it replaced (recorded), leaves in place against copied caches, threads, errors."""

import hashlib
import multiprocessing
import threading

import numpy as np
import pytest
import torch

from allie.lichess import fastrs
from allie.lichess.fastrs import Leaf
from allie.lichess.model import Cache, Model, step
from allie.lichess.tokens import CONTEXT, HEADER

from .test_model import inputs
from .test_tokens import random_game

allie_fast = pytest.importorskip("allie_fast")
if not hasattr(allie_fast, "Engine"):
    pytest.skip("allie_fast built without Engine", allow_module_level=True)

GAMES = [inputs(random_game(s, 36 + s % 5)) for s in range(64)]


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


def digest(out):
    """replay()'s logits and cache rows, hashed bit for bit."""
    h = hashlib.sha256()
    for z in out[0]:
        h.update(z.contiguous().numpy().tobytes())
    for rows in out[1]:
        for t in rows:
            h.update(t.contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()[:16]


# digest(replay()) per (isa, int8): the C++ kernels' bits, which the Rust port reproduced on the same ISA (the
# equality gate, 2026-10-05); a change here changes the model's outputs
GOLDEN = {
    ("avx512", False): "395313f798d99c0e",
    ("avx512", True): "076463256ccb2c30",
    ("avx2", False): "531f923324856ba0",
    ("avx2", True): "3972dd998c02dd04",
    ("scalar", False): "adf623afb91bcb5e",
    ("scalar", True): "ea60a73cacaa0822",
}


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("isa", ["native", "avx2", "scalar"])
def test_rust_isas(tiny_path, isa, int8, monkeypatch):
    """Each kernel variant is the reference up to BF16 rounding, and bitwise the recorded C++ outputs."""
    got = rust(tiny_path, isa, monkeypatch, int8=int8, threads=3)
    assert got.fast.isa in ("avx512", "avx2", "scalar") and (isa == "native" or got.fast.isa == isa)
    out = replay(got)
    ref = replay(Model(tiny_path, dtype=torch.bfloat16, int8=int8, backend="torch"))
    for z, w in zip(out[0], ref[0]):
        p, q = (torch.softmax(x[:, 378:2346], -1) for x in (z, w))
        assert (p - q).abs().max() < 2e-3
    want = GOLDEN.get((got.fast.isa, int8))
    assert want is None or digest(out) == want, f"{got.fast.isa} int8={int8}: the kernels' bits changed"


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
    meta = [c.capacity, 1, c.capacity, 0, 0, 0, 0, 0]
    assert e.step(1, 1, ids[:1].data_ptr(), feats[:1].data_ptr(), boards[:1].data_ptr(), meta,
                  [c.k.data_ptr(), c.v.data_ptr(), c.e.data_ptr(), 0, 0, 0], [], torch.empty(1, 2432).data_ptr()) == 3  # fmt: skip
    # leaves: a slot or path entry past the tree, a path past the context, two tokens
    step(m, [(c, ids[:4], feats[:4], boards[:4])])
    t, one = Slots(m, 8), (ids[4:5], feats[4:5], boards[4:5])
    for bad in (
        Leaf(c, *one, t, (1,), 8),
        Leaf(c, *one, t, (1, 9), 2),
        Leaf(c, ids[4:6], feats[4:6], boards[4:6], t, (), 2),
    ):
        with pytest.raises(ValueError, match="a leaf"):
            step(m, [bad])
    with pytest.raises(ValueError, match="past"):
        step(m, [Leaf(c, *one, t, (1,) * (CONTEXT - 4), 2)])
    assert c.n == 4


class Slots:
    """A tree's slot buffer as tree.Tree keeps it: k, v [L, H, cap, hd], e [cap, D]; NaN where never written."""

    def __init__(self, model, capacity):
        kw = dict(dtype=model.dtype, device=model.device)
        self.k = torch.full(
            (model.layers, model.heads, capacity, model.head_dim), torch.nan, **kw
        )
        self.v, self.e = (
            torch.full_like(self.k, torch.nan),
            torch.full((capacity, model.width), torch.nan, **kw),
        )
        self.capacity = capacity


def random_tree(rng, nodes, depth):
    """parent[i] for nodes 1..nodes (0 the root; parents precede their children), depths at most `depth`:
    chains with probability 0.4, else a random shallower node."""
    parent, dep = np.zeros(nodes + 1, np.int64), np.zeros(nodes + 1, np.int64)
    for i in range(1, nodes + 1):
        p = (
            i - 1
            if rng.random() < 0.4 and dep[i - 1] < depth
            else rng.choice(np.flatnonzero(dep[:i] < depth))
        )
        parent[i], dep[i] = p, dep[p] + 1
    return parent, dep


def path(parent, i):
    p = []
    while i:
        p.append(int(i))
        i = parent[i]
    return p[::-1]


def leaves_both_ways(model, seed=0, roots=3, nodes=40, depth=8):
    """Random trees over several games' caches, their nodes evaluated in the same batches both as Leaf items (in
    place) and today's way (the game's prefix copied into a pooled cache, the ancestors' rows at n0.., a plain
    one-token item), batches mixing roots and plain one- and two-token appends of another game: (each batch's
    logits, the slot buffers, the game caches' rows) each way."""
    rng = np.random.default_rng(seed)
    games, n0s = [], []
    for r in range(roots):
        ids, feats, boards = GAMES[r]
        n0 = HEADER + 4 + 3 * r
        c = Cache(model, n0 + 1)
        step(model, [(c, ids[:n0], feats[:n0], boards[:n0])])
        games.append(c)
        n0s.append(n0)
    trees = [random_tree(rng, nodes, depth) for _ in range(roots)]
    slots = [[Slots(model, nodes + 1) for _ in range(roots)] for _ in range(2)]
    spare = [Cache(model, 8) for _ in range(2)]
    at, zs = 0, [[], []]
    for i in range(1, nodes + 1):
        order = rng.permutation(roots) if rng.random() < 0.5 else range(roots)
        ids, feats, boards = GAMES[roots + 1]  # a plain item of 1 or 2 of its tokens
        extra = min(int(rng.integers(0, 3)) if rng.random() < 0.5 else 0, len(ids) - at)
        items, pooled = [[], []], []
        for r in order:
            ids, feats, boards = GAMES[(r * 13 + i) % len(GAMES)]
            j = (i * 5) % 36
            tok = (ids[j : j + 1], feats[j : j + 1], boards[j : j + 1])
            parent, dep = trees[r]
            anc, n0 = path(parent, parent[i]), n0s[r]
            items[0].append(Leaf(games[r], *tok, slots[0][r], tuple(anc), i))
            c = Cache(model, n0 + len(anc) + 1)
            c.k[:, :, :n0], c.v[:, :, :n0], c.e[:n0] = (
                games[r].k[:, :, :n0],
                games[r].v[:, :, :n0],
                games[r].e[:n0],
            )
            s = slots[1][r]
            for d, a in enumerate(anc):
                c.k[:, :, n0 + d], c.v[:, :, n0 + d], c.e[n0 + d] = (
                    s.k[:, :, a],
                    s.v[:, :, a],
                    s.e[a],
                )
            c.n = n0 + len(anc)
            items[1].append((c, *tok))
            pooled.append((c, r))
        if extra:
            for w in range(2):
                items[w].append(
                    (
                        spare[w],
                        ids[at : at + extra],
                        feats[at : at + extra],
                        boards[at : at + extra],
                    )
                )
            at += extra
        for w in range(2):
            zs[w].append(step(model, items[w]))
        for c, r in pooled:
            s, p = slots[1][r], c.n - 1
            s.k[:, :, i], s.v[:, :, i], s.e[i] = c.k[:, :, p], c.v[:, :, p], c.e[p]
    rows = lambda c: (
        c.k[:, :, : c.n].clone(),
        c.v[:, :, : c.n].clone(),
        c.e[: c.n].clone(),
    )
    assert all(c.n == n0 for c, n0 in zip(games, n0s)), "a leaf grew the game's cache"
    return zs, slots, [rows(c) for c in games + spare]


def same_leaves(a, b):
    (za, sa, ga), (zb, sb, gb) = a, b
    for i, (x, y) in enumerate(zip(za, zb)):
        bad = (x != y).sum().item()
        assert bad == 0, (
            f"batch {i}: {bad} of {x.numel()} logits differ, max {(x - y).abs().max().item()}"
        )
    for r, (x, y) in enumerate(zip(sa, sb)):
        for name in "kve":
            s, t = getattr(x, name)[..., 1:, :], getattr(y, name)[..., 1:, :]
            assert torch.equal(s, t), f"tree {r} slots {name} differ"
    for j, (x, y) in enumerate(zip(ga, gb)):
        assert all(torch.equal(s, t) for s, t in zip(x, y)), f"cache {j} differs"


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("isa", ["native", "avx2", "scalar"])
def test_rust_leaves_match_copied_bitwise(tiny_path, isa, int8, monkeypatch):
    """A node evaluated in place (a Leaf item: the cache's rows, then its ancestors' slots) is bitwise the node
    evaluated on a copy of the cache with the path appended, logits and the slot's new k, v and e alike."""
    m = rust(tiny_path, isa, monkeypatch, int8=int8, threads=3)
    zs, slots, rows = leaves_both_ways(m)
    same_leaves((zs[0], slots[0], rows), (zs[1], slots[1], rows))
    assert all(torch.isfinite(z).all() for z in zs[0]) and len(zs[0]) == 40
    assert all(
        torch.isnan(s.k[:, :, 0]).all() for s in slots[0]
    )  # the root's slot is never touched


def test_rust_leaves_thread_count_does_not_change_results(tiny_path, monkeypatch):
    a = rust(tiny_path, "native", monkeypatch, int8=True, threads=1)
    b = rust(tiny_path, "native", monkeypatch, int8=True, threads=3)
    (za, sa, ra), (zb, sb, rb) = (leaves_both_ways(x, nodes=16) for x in (a, b))
    same_leaves((za[0], sa[0], ra), (zb[0], sb[0], rb))


def test_rust_profile_threads_isa(tiny_path):
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    assert m.fast.threads == 2 and m.fast.handle.threads == 2
    assert m.fast.isa in ("avx512", "avx2", "scalar")
    m.fast.profile()
    x = inputs(random_game(3, 10))
    step(m, [(Cache(m), *x)])
    t = m.fast.profile(False)
    assert list(t) == list(fastrs.PHASES) and len(t) == 17 and sum(t.values()) > 0


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


def test_rust_rejects_aliasing_items(tiny_path):
    """A step's writes never overlap its reads or each other: every rejection in Python and in the engine itself."""
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    ids, feats, boards = inputs(random_game(5, 8))
    c = Cache(m)
    step(m, [(c, ids[:4], feats[:4], boards[:4])])
    t, one = Slots(m, 8), (ids[4:5], feats[4:5], boards[4:5])
    step(m, [Leaf(c, *one, t, (), 1)])
    bad = {
        "own path": [Leaf(c, *one, t, (1,), 1)],
        "same slot": [Leaf(c, *one, t, (1,), 2), Leaf(c, *one, t, (1,), 2)],
        "slot in another path": [Leaf(c, *one, t, (1,), 2), Leaf(c, *one, t, (1, 2), 3)],
        "tree as cache": [Leaf(c, *one, c, (1,), 2)],
        "two plain": [(c, *one), (c, *one)],
    }
    for name, items in bad.items():
        with pytest.raises(ValueError, match="alias"):
            step(m, items)
        assert c.n == 4, name
    # the engine's own check, bypassing the Python mirror: a leaf whose destination is in its path (error 4)
    e = m.fast.native()
    meta = [c.n, 1, c.capacity, 0, t.capacity, 1, 0, 1]
    ptrs = [c.k.data_ptr(), c.v.data_ptr(), c.e.data_ptr(), t.k.data_ptr(), t.v.data_ptr(), t.e.data_ptr()]
    assert e.step(1, 1, ids[4:5].data_ptr(), feats[4:5].data_ptr(), boards[4:5].data_ptr(), meta, ptrs, [1],
                  torch.empty(1, 2432).data_ptr()) == 4  # fmt: skip
    # raw pointer cases the Python mirror cannot see: a single leaf whose slot buffer is its own cache (every
    # pointer equal), a cache whose k and v are one buffer, two caches sharing only their v buffer
    cp = [c.k.data_ptr(), c.v.data_ptr(), c.e.data_ptr()]
    tp = [t.k.data_ptr(), t.v.data_ptr(), t.e.data_ptr()]
    one_leaf = [c.n, 1, c.capacity, 0, t.capacity, 0, 0, 2]
    out2 = torch.empty(2, 2432)  # alive for the whole call: the step writes it (a temporary's pointer dangles)
    args = lambda m_, p_, pa=[]: e.step(len(m_) // 8, len(m_) // 8, ids[4:6].data_ptr(), feats[4:6].data_ptr(), boards[4:6].data_ptr(), m_, p_, pa, out2.data_ptr())  # noqa: E731
    assert args(one_leaf, cp + cp) == 4
    assert args(one_leaf, cp + [t.k.data_ptr(), c.v.data_ptr(), t.e.data_ptr()]) == 4
    assert args([c.n, 1, c.capacity, 0, 0, 0, 0, 0], [cp[0], cp[0], cp[2], 0, 0, 0]) == 4
    c2 = Cache(m, c.capacity)
    two = [c.n, 1, c.capacity, 0, 0, 0, 0, 0, 0, 1, c2.capacity, 1, 0, 0, 0, 0]
    assert args(two, cp + [0, 0, 0] + [c2.k.data_ptr(), c.v.data_ptr(), c2.e.data_ptr()] + [0, 0, 0]) == 4
    assert args(one_leaf, cp + tp, [0]) == 0  # the same leaf with its own tree runs
    # a leaf reading rows below a plain item's first write on the same cache is fine, and so are many leaves
    # of one tree with distinct slots and shared paths
    z = step(m, [(c, *one), Leaf(c, *one, t, (1,), 2), Leaf(c, *one, t, (1,), 3), Leaf(c, *one, t, (), 4)])
    assert z.shape == (4, 2432) and c.n == 5


def test_rust_rejects_overflowing_spans(tiny_path):
    """Spans near i64's end are refused by the engine, not computed."""
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    ids, feats, boards = inputs(random_game(6, 6))
    c = Cache(m)
    e = m.fast.native()
    out = torch.empty(1, 2432)
    ptrs = [c.k.data_ptr(), c.v.data_ptr(), c.e.data_ptr(), 0, 0, 0]
    for meta in ([2**63 - 1, 1, c.capacity, 0, 0, 0, 0, 0], [0, 2**63 - 1, c.capacity, 0, 0, 0, 0, 0],
                 [0, 1, 2**62, 0, 0, 0, 0, 0], [2**63 - 1, 1, c.capacity, 0, 8, 1, 0, 1], [0, 1, c.capacity, 0, 8, 2**63 - 1, 0, 1]):  # fmt: skip
        assert e.step(1, 1, ids[:1].data_ptr(), feats[:1].data_ptr(), boards[:1].data_ptr(), meta, ptrs, [1], out.data_ptr()) == 3, meta


def test_rust_leaf_before_cache_growth_in_one_step(tiny_path):
    """A plain item growing a cache in the same step as a leaf reading it: the leaf must see the grown cache's
    layout (capacity is the row stride), the same bits as with the cache grown beforehand."""
    m = Model(tiny_path, dtype=torch.bfloat16, backend="rust", threads=2)
    ids, feats, boards = inputs(random_game(7, 20))
    out = []
    for grow_early in (True, False):
        root = Cache(m, 8)
        step(m, [(root, ids[:4], feats[:4], boards[:4])])
        t = Slots(m, 16)
        if grow_early:
            root.reserve(12)
        z = step(m, [Leaf(root, ids[4:5], feats[4:5], boards[4:5], t, (), 1), (root, ids[5:13], feats[5:13], boards[5:13])])
        assert torch.isfinite(z).all() and root.n == 12
        out.append((z, t.k[:, :, 1].clone(), t.v[:, :, 1].clone(), t.e[1].clone()))
    for a, b in zip(*out):
        assert torch.equal(a, b)
