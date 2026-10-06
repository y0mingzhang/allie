"""Allie 2.0's GPU fast path for model.py's step(): Graphs replays model.py's forward, captured once per batch
and attention span, for steps that add one token to each game; a game's cache is a slot of one preallocated
pool. The CPU's is the Rust engine (fastrs.py)."""

import threading
import warnings
import weakref

import torch
from torch.nn import functional as F


class Graphs:
    """CUDA graphs of model.py's forward for steps that add one token to each of up to 32 games.
    Each game's cache is a slot of one preallocated pool (Cache gets views into it, so the
    PyTorch path runs on the same memory); a step of B games replays the graph captured for
    the smallest batch >= B and attention span > every game's length, with padding rows on a
    spare slot. Launch overhead, not memory, bounds a single game's step on a GPU: one replay
    replaces about a thousand kernel launches."""

    BATCHES = (1, 2, 4, 8, 16, 32)
    SPANS = (128, 256, 512, 1025)

    def __init__(self, model, slots=None, strict=False):
        m, ctx = model, model.w["cos"].shape[0]
        per = (
            2 * m.layers * m.heads * ctx * m.head_dim + ctx * m.width
        )  # elements per slot
        size = per * torch.tensor([], dtype=m.dtype).element_size()
        if slots is None:  # a third of the free memory, at most 64 games
            slots = min(64, int(torch.cuda.mem_get_info(m.device)[0] / 3 // size))
        if slots < 1:
            raise RuntimeError("no GPU memory for a cache pool")
        kw = dict(dtype=m.dtype, device=m.device)
        self.k = torch.zeros(
            m.layers, slots + 1, m.heads, ctx, m.head_dim, **kw
        )  # last: padding
        self.v = torch.zeros_like(self.k)
        self.e = torch.zeros(slots + 1, ctx, m.width, **kw)
        self.model, self.slots, self.ctx = m, slots, ctx
        self.free = list(range(slots))[::-1]
        self.graphs, self.pool, self.lock = {}, None, threading.Lock()
        self.stream = torch.cuda.Stream(m.device)  # captures run on the model's device
        self.strict, self.broken = strict, False  # strict: a failed capture raises

    def attach(self, cache):
        """A new cache's storage: a free slot (False when none is left)."""
        try:
            j = self.free.pop()  # atomic under the GIL: two new caches never get one slot
        except IndexError:
            return False
        cache.k, cache.v, cache.e = self.k[:, j], self.v[:, j], self.e[j]
        cache.capacity, cache.slot = self.ctx, j
        weakref.finalize(cache, self.free.append, j)
        return True

    def capture(self, b, span):
        m, dev = self.model, self.model.device
        x = dict(ids=torch.zeros(b, dtype=torch.long, device=dev),
                 feats=torch.full((b, 3), -1.0, device=dev),
                 boards=torch.zeros(b, 68, dtype=torch.uint8, device=dev),
                 pos=torch.zeros(b, dtype=torch.long, device=dev),
                 slot=torch.full((b,), self.slots, dtype=torch.long, device=dev))  # fmt: skip
        x["keys"] = keys = torch.arange(span, device=dev)  # kept: the graph reads its memory

        def previous(e):
            p = self.e[x["slot"], (x["pos"] - 1).clamp(min=0)] * (x["pos"] > 0)[:, None]
            self.e[x["slot"], x["pos"]] = e
            return p

        def attend(i, q, k, v):
            self.k[i, x["slot"], :, x["pos"]] = k
            self.v[i, x["slot"], :, x["pos"]] = v
            mask = (keys <= x["pos"][:, None])[:, None, None]
            y = F.scaled_dot_product_attention(q[:, :, None], self.k[i, x["slot"], :, :span],
                                               self.v[i, x["slot"], :, :span], attn_mask=mask,
                                               scale=m.scale)  # fmt: skip
            return y[:, :, 0]

        def run():
            return m.forward(
                x["ids"], x["pos"], x["feats"], x["boards"], previous, attend
            )

        side = self.stream
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up the libraries outside the capture
                run()
        torch.cuda.current_stream(dev).wait_stream(side)
        self.pool = self.pool or torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool, stream=side):
            out = run()
        return g, x, out

    def step(self, items):
        """step()'s logits, or None when the step is not one new token per pooled game. The
        work runs on the graphs' own stream, one call at a time, after the caller's stream."""
        with self.lock, torch.cuda.device(self.model.device):
            if self.broken:
                return None
            caller = torch.cuda.current_stream()
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                z = self._step(items)
            caller.wait_stream(self.stream)
            if z is not None:
                z.record_stream(caller)
            return z

    def _step(self, items):
        n = len(items)
        if n > self.BATCHES[-1] or any(len(ids) != 1 or getattr(c, "slot", None) is None
                                       for c, ids, *_ in items):  # fmt: skip
            return None
        b = next(x for x in self.BATCHES if x >= n)
        span = next(x for x in self.SPANS if x > max(c.n for c, *_ in items))
        if (b, span) not in self.graphs:
            try:
                self.graphs[b, span] = self.capture(b, span)
            except Exception as e:
                if self.strict:
                    raise
                warnings.warn(f"CUDA graphs failed, running without them: {e}")
                self.broken = True
                return None
        g, x, out = self.graphs[b, span]
        for j, k in enumerate(("ids", "feats", "boards"), 1):
            x[k][:n].copy_(torch.cat([it[j] for it in items]))
        x["pos"][:n].copy_(torch.tensor([c.n for c, *_ in items]))
        x["slot"][:n].copy_(torch.tensor([c.slot for c, *_ in items]))
        if b > n:  # padding rows: a spare slot's first position
            x["pos"][n:], x["slot"][n:] = 0, self.slots
        g.replay()
        for c, *_ in items:
            c.n += 1
        return out[:n].clone()
