"""The Rust port of fast.py's CPU backend (rust/allie-fast, module allie_fast): the same step() on the
same weights and caches, bit for bit the C++ kernels' logits on the same ISA. `RustFast` has `Fast`'s
constructor, attributes and methods; the cfg and pointer lists it hands the engine are built as
`Fast.__init__` builds them. The engine reads the tensors in place through raw pointers, so they are
kept alive here. Threads are pinned and the weights' pages placed on their NUMA node as in fast.py.

The engine lives in an `allie_fast.Server`: its own thread runs the steps, merging what every caller (games'
moves, searches on their threads, the Rust searchers' native loops) has queued into one step of up to
max_items items, with a gather window of `gather` seconds after a step's first request for the other
requests of the last step. step() blocks for its reply with the GIL released; the caller's tensors stay
alive until then.
"""

import os
import threading
import weakref
from pathlib import Path
from typing import NamedTuple

import torch

from .fast import LAYER, PHASES, SCALED, cpu_order, cpus, threads_default


class Leaf(NamedTuple):
    """A search node evaluated in place (the engine's path item): the token `ids` [1] at position cache.n +
    len(path) attends to the cache's rows and then to the slots `path` (its ancestors below the root, in order)
    of the tree's buffer (k, v [L, H, capacity, hd], e [capacity, D]); its keys, values and embedding are written
    to `slot`. The cache is read only. Its first four fields are a plain item's, so engine.py batches it as one."""

    cache: object
    ids: torch.Tensor
    feats: torch.Tensor
    boards: torch.Tensor
    tree: object
    path: tuple
    slot: int


class RustFast:
    """model.py's step() for a CPU Model with BF16 activations (int8 or BF16 matrices), in Rust."""

    def __init__(self, model, threads=None, pin=True, spin=0.002, place=True, max_items=128, min_items=48, gather=3e-4):
        import allie_fast

        assert model.device.type == "cpu" and model.dtype == torch.bfloat16
        c, w, n = model.config, model.w, model.layers
        assert model.head_dim % 16 == 0 and model.head_dim <= 256 and model.topk <= 64
        assert (
            w["board.first"].shape == (32, 13, 3, 3)
            and w["board.output"].shape[0] == 544
        )
        self.lib, self.model = allie_fast, model
        if threads is not None and threads < 1:
            raise ValueError("threads must be at least 1")
        self.threads = min(
            threads or threads_default(), len(cpus())
        )  # never more than the CPUs
        self.lock = threading.Lock()
        int8 = bool(model.scales)
        ve = [None] * n
        for j in range(model.ve):
            ve[j], ve[n - model.ve + j] = j, j
        skip_in = [i * n // 16 for i in (2, 4, 6)]
        skip_out = [9 * n // 16 + i for i in range(3)]
        dense = next(
            (w[f"{i}.fc"].shape[0] // 2 for i in range(n) if f"{i}.fc" in w), 0
        )
        cfg = [n, model.width, model.heads, model.head_dim, c["vocab"], model.ve, c["experts"],
               model.topk, model.keep, c["expert_hidden"], c["shared_hidden"], dense, int(int8),
               w["cos"].shape[0], skip_out[-1], self.threads, 0]  # fmt: skip
        for i in range(n):
            cfg += [w[f"{i}.gates"].shape[0], -1 if ve[i] is None else ve[i],
                    skip_in.index(i) if i in skip_in else -1,
                    skip_out.index(i) if i in skip_out else -1]  # fmt: skip
        glob = ["embed", "embed2", "lm_head", "feat_embed", "smear_gate", "scalars", "x0_lambdas",
                "cos", "sin", "board.first", "board.residual.0", "board.residual.1",
                "board.squeeze", "board.meta", "board.output", "skip_gate.0", "skip_gate.1",
                "skip_gate.2", *[f"value_embed.{j}" for j in range(model.ve)]]  # fmt: skip
        ptrs, self.tensors = [], []  # the engine reads these in place: keep them alive
        for k in glob:
            t = w[k]
            assert t.is_contiguous() and t.dtype == (
                torch.float32 if k in ("scalars", "x0_lambdas") else torch.bfloat16
            ), k
            ptrs.append(t.data_ptr())
            self.tensors.append(t)
        lay = []
        for i in range(n):
            for k in LAYER:
                t = w.get(f"{i}.{k}")
                if t is not None:
                    want = (torch.float32 if k in ("router", "moe_bias", "mu")
                            else torch.int8 if int8 and k in SCALED else torch.bfloat16)  # fmt: skip
                    assert t.is_contiguous() and t.dtype == want, (i, k, t.dtype)
                lay.append(0 if t is None else t.data_ptr())
                self.tensors.append(t)
                if k in SCALED:
                    s = model.scales.get(f"{i}.{k}")
                    assert (s is not None) == (int8 and t is not None), (i, k)
                    lay.append(0 if s is None else s.data_ptr())
                    self.tensors.append(s)
        order = cpu_order(self.threads) if pin else None
        self.cpus, nodes, groups = order or (None, None, None)
        place = place and os.environ.get("ALLIE_NUMA") != "0"
        if place and nodes and len(set(nodes)) == 1 and Path("/sys/devices/system/node/node1").exists():
            # the weights' pages moved to the threads' NUMA node (as fast.py: threads on several nodes
            # leave them where they are)
            ts = [t for t in self.tensors if t is not None]
            allie_fast.place([t.data_ptr() for t in ts], [t.numel() * t.element_size() for t in ts], sorted(set(nodes)))
        self.args = (
            cfg,
            float(model.scale),
            float(model.floor),
            ptrs,
            lay,
            self.cpus,
            groups,
            spin,
        )
        self.serve = dict(max_items=max_items, min_items=min_items, gather=gather)
        self.handle, self.stale = self.lib.Server(*self.args, **self.serve), False
        if hasattr(os, "register_at_fork"):
            ref = weakref.ref(self)
            os.register_at_fork(
                after_in_child=lambda: ref() is not None and ref().forked()
            )

    @property
    def isa(self):
        return self.handle.isa

    def forked(self):
        """In a fork's child, before its threads run: the parent's server and pool threads are gone and its
        lock may have been held. A fresh lock; the server is rebuilt on first use."""
        self.lock, self.stale = threading.Lock(), True

    def native(self):
        if self.stale:
            self.handle, self.stale = self.lib.Server(*self.args, **self.serve), False
        return self.handle

    server = property(native)

    def profile(self, on=True):
        """Seconds spent in each phase of step() since the last call; on: keep counting."""
        with self.lock:
            out = self.native().profile(on)
        assert len(out) == len(PHASES)
        return dict(zip(PHASES, out))

    def step(self, items):
        self.native()
        return self._step(items)  # the server serializes the steps and merges concurrent callers'

    def check(self, cache, ids, feats, boards, *leaf):
        """An item's shapes, and the cache grown for a plain one; its tensors (and a Leaf's tree's) are
        checked by layout()."""
        n = len(ids)
        if (
            n < 1
            or ids.shape != (n,)
            or feats.shape != (n, 3)
            or boards.shape != (n, 68)
        ):
            raise ValueError(
                "an item: ids [m], feats [m, 3] and boards [m, 68], m >= 1"
            )
        if cache.model is not self.model:
            raise ValueError("a cache of another model")
        if not leaf:
            cache.reserve(cache.n + n)
            return
        tree, path, slot = leaf
        if n != 1 or not all(0 <= r < tree.capacity for r in (slot, *path)):
            raise ValueError(
                "a leaf: one token, its path and slot within the tree's capacity"
            )

    def layout(self, c, seen):
        """c's k, v and e in the model's layout at c.capacity (a cache or a tree buffer, once a step)."""
        if id(c) in seen:
            return
        seen.add(id(c))
        m = self.model
        shape = (m.layers, m.heads, c.capacity, m.head_dim)
        for t, s in ((c.k, shape), (c.v, shape), (c.e, (c.capacity, m.width))):
            if (
                t.shape != s
                or t.dtype != m.dtype
                or t.device.type != "cpu"
                or not t.is_contiguous()
            ):
                raise ValueError("a cache whose tensors are not the model's layout")

    @staticmethod
    def disjoint(items):
        """A step's writes must not alias its reads or each other (the engine checks the pointers too): a leaf's
        slot is not in its path, nor another leaf's slot or path on the same tree; a tree is never a cache; a cache
        is appended by one plain item at most (leaves read its rows below that item's first write)."""
        plain, trees = {}, {}
        for item in items:
            if isinstance(item, Leaf):
                if item.slot in item.path:
                    raise ValueError("items alias: a leaf's slot in its own path")
                for other in trees.setdefault(id(item.tree), []):
                    if item.slot == other.slot or item.slot in other.path or other.slot in item.path:
                        raise ValueError("items alias: two leaves of one tree write or read the same slot")
                trees[id(item.tree)].append(item)
            else:
                if id(item[0]) in plain:
                    raise ValueError("items alias: two plain items append one cache")
                plain[id(item[0])] = item
        for item in items:
            c = item[0]
            if id(c) in trees:
                raise ValueError("items alias: a tree used as a cache")
            if isinstance(item, Leaf) and id(c) in plain and c.n != plain[id(c)][0].n:
                raise ValueError("items alias: a leaf reads rows a plain item writes")

    def _step(self, items):
        for item in items:  # every cache grown first: a later item's reserve() would leave an earlier item's
            self.check(*item)  # capacity (the engine's row stride) stale against the pointers taken below
        meta, paths, lo, seen = [], [], 0, set()
        for item in items:
            cache, ids = item[0], item[1]
            if isinstance(item, Leaf):
                meta += [cache.n, 1, cache.capacity, lo, item.tree.capacity, len(item.path), len(paths), item.slot]  # fmt: skip
                paths += item.path
                self.layout(item.tree, seen)
            else:
                meta += [cache.n, len(ids), cache.capacity, lo, 0, 0, 0, 0]
            self.layout(cache, seen)
            lo += len(ids)
        self.disjoint(items)
        caches = []  # after every reserve(): a grown cache's tensors are new
        for item in items:
            c, t = item[0], item.tree if isinstance(item, Leaf) else None
            caches += [c.k.data_ptr(), c.v.data_ptr(), c.e.data_ptr()]
            caches += [0, 0, 0] if t is None else [t.k.data_ptr(), t.v.data_ptr(), t.e.data_ptr()]  # fmt: skip

        def cat(j, dtype):
            return torch.cat([it[j] for it in items]).to("cpu", dtype).contiguous()

        ids, feats, boards = (
            cat(1, torch.int64),
            cat(2, torch.float32),
            cat(3, torch.uint8),
        )
        out = torch.empty(
            len(items), self.model.config["vocab"], dtype=torch.float32, device="cpu"
        )
        err = self.handle.step(lo, len(items), ids.data_ptr(), feats.data_ptr(), boards.data_ptr(),
                               meta, caches, paths, out.data_ptr())  # fmt: skip
        if err:
            raise ValueError(("token outside the vocabulary", "board state out of range",
                              "tokens, paths or slots past the cache, the tree or the context",
                              "items alias: a write overlaps a read or another write")[err - 1])  # fmt: skip
        for item in items:
            if not isinstance(item, Leaf):
                item[0].n += len(item[1])
        return out
