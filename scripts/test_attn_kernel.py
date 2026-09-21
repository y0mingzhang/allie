"""modded_attn (Triton FlashAttention-2) vs the FlexAttention path and an FP32 reference: output and
q/k/v grads on real rows (64 x 1024 tokens, 24 heads of 64) and on synthetic game layouts (games of
1-2000 tokens, boundaries on block and row edges, a row without a game start, a row of 1-token games),
for the WSD windows and sliding ones; then CUDA-event medians (REPS) of forward and backward: Flex
(128-token blocks, as trained), Flex at 64-token blocks, and the Triton kernels.

    <runtime python> scripts/test_attn_kernel.py [--rows rows.npy] [--sweep]
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    _create_sparse_block_from_block_mask,
    create_block_mask,
    flex_attention,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modded_attn as ma
import modded_medium as mm

H, D, SCALE, REPS = 24, 64, 0.1, 30
DATA = "/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2"
WSD, SLIDING = (11 * 128, 23 * 128), (128, 384)
flex64 = torch.compile(flex_attention, dynamic=False)


def load_rows(argv):
    if "--rows" in argv:
        return np.load(argv[argv.index("--rows") + 1]).astype(np.int64)[:, :-1]
    from lm_data import Packed

    val = Packed(DATA, "val")
    pick = np.random.default_rng(0).choice(int(val.ends[-1]), 64, replace=False)
    return val.rows(pick)[:, :-1]


def synthetic(rng, n, length):
    rows = rng.integers(0, mm.BOS, size=(n, length))
    for r in range(n):
        pos = 0
        while pos < length:
            rows[r, pos] = mm.BOS
            pos += int(rng.choice([1, 2, 5, 60, 127, 128, 129, 300, 700, 2000]))
    rows[0, [63, 64, 127, 128, 255, 256, length - 1]] = mm.BOS
    rows[1] = rng.integers(
        0, mm.BOS, size=length
    )  # no game start: the row start splits
    rows[2] = mm.BOS  # every token starts a game
    return rows


def inputs(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(1, T, 3 * H, D, generator=g, device="cuda").bfloat16()
    q, k, v = qkv.chunk(3, 2)
    q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
    do = torch.randn(1, T, H, D, generator=g, device="cuda").bfloat16()
    gate = torch.randn(1, T, H, 1, generator=g, device="cuda").sigmoid().bfloat16()
    # v stays a strided view of qkv, as in the model
    return [x.detach().requires_grad_() for x in (q, k, v, gate)], do


def reference(q, k, v, gate, do, docs, window, row):
    """FP32 per-row dense attention (times gate, if given) and grads under make_context's dense mask."""
    outs = [[] for _ in range(4 if gate is None else 5)]
    t = torch.arange(row, device="cuda")
    for r in range(q.shape[1] // row):
        s = slice(r * row, (r + 1) * row)
        qq, kk, vv, gg = (
            None if x is None else x[0, s].float().detach().requires_grad_()
            for x in (q, k, v, gate)
        )
        d = docs[s]
        allow = (t[None] <= t[:, None]) & (t[:, None] - t[None] <= window)
        allow &= d[:, None] == d[None]
        p = torch.einsum("thd,shd->hts", qq, kk) * SCALE
        o = torch.einsum(
            "hts,shd->thd", p.masked_fill(~allow, -torch.inf).softmax(-1), vv
        )
        wrt = (qq, kk, vv) if gate is None else (qq, kk, vv, gg)
        o = o if gate is None else o * gg
        grads = torch.autograd.grad(o, wrt, do[0, s].float())
        for out, x in zip(outs, (o, *grads)):
            out.append(x.detach())
    return [torch.cat(x)[None] for x in outs]


def mask64(docs, window, row):
    n = docs.numel()

    def allowed(b, h, q, k):
        return (q >= k) & (q - k <= window) & (docs[q] == docs[k])

    if window >= row - 1:
        blocks = mm.game_blocks(docs, 64)
        return _create_sparse_block_from_block_mask(blocks, allowed, (n, n), 64, 64)
    return create_block_mask(
        allowed, 1, None, n, n, "cuda", BLOCK_SIZE=64, _compile=True
    )


def err(a, b):
    d = (a.float() - b.float()).abs().max()
    rel = (d / b.float().abs().max()).item()
    return f"{d.item():.2e}/{rel:.2e}{' bitwise' if torch.equal(a, b) else ''}"


def bench(fn, setup=lambda: None):
    """Median ms of fn(setup()) over REPS, timing fn alone (a fresh forward per backward)."""
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(REPS)]
    for _ in range(3):
        fn(setup())
    for a, b in ev:
        x = setup()
        a.record()
        fn(x)
        b.record()
    torch.cuda.synchronize()
    return float(np.median([a.elapsed_time(b) for a, b in ev]))


def grads(o, qkv, do):
    return [o.detach(), *torch.autograd.grad(o, qkv, do)]


def check(rows, windows, what, gated=False):
    """Flex's gated path is the model's: its output times the gate, in BF16."""
    n, row = rows.shape
    flex = mm.make_context(rows, *windows, backend="flex")
    tri = mm.make_context(rows, *windows, backend="triton")
    for w in windows:
        (q, k, v, g), do = inputs(n * row, 0)
        wrt, g = ((q, k, v, g), g) if gated else ((q, k, v), None)
        y = mm.attention(q, k, v, flex, w, SCALE)
        f = grads(y if g is None else y * g, wrt, do)
        t = grads(mm.attention(q, k, v, tri, w, SCALE, g), wrt, do)
        again = grads(mm.attention(q, k, v, tri, w, SCALE, g), wrt, do)
        ref = reference(q, k, v, g, do, flex.documents, w, row)
        det = all(torch.equal(a, b) for a, b in zip(t, again))
        names = "o dq dk dv" + " dgate" * gated
        print(
            f"{what}{' gated' * gated} window {w} ({names} max abs/rel err;"
            f" triton rerun bitwise {det})"
        )
        for label, a, b in (
            ("triton-flex", t, f),
            ("triton-fp32", t, ref),
            ("flex-fp32", f, ref),
        ):
            print(f"  {label:12s}", *(err(x, y) for x, y in zip(a, b)), flush=True)


def timing(rows, windows, timed):
    n, row = rows.shape
    flex = mm.make_context(rows, *windows, backend="flex")
    tri = mm.make_context(rows, *windows, backend="triton")
    delta = torch.compile(
        lambda do, o: (do.float() * o.float()).sum(-1)[0].t().contiguous()
    )
    for w in timed:
        (*qkv, g), do = inputs(n * row, 1)
        out = [f"window {w} ms:"]
        run = lambda: mm.attention(*qkv, flex, w, SCALE)  # noqa: E731
        f = bench(lambda _: run())
        b = bench(lambda o: torch.autograd.grad(o, qkv, do), run)
        out.append(f"flex fwd {f:.3f} bwd {b:.3f} sum {f + b:.3f} |")
        lo, hi = tri.short_mask if w == windows[0] else tri.long_mask
        run = lambda: mm.attention(*qkv, tri, w, SCALE)  # noqa: E731
        eager = bench(lambda o: torch.autograd.grad(o, qkv, do), run)
        with torch.no_grad():
            o, lse = ma.fwd(*qkv, None, lo, hi, SCALE)
            dl = delta(do, o)
            f = bench(lambda _: ma.fwd(*qkv, None, lo, hi, SCALE))
            d = bench(lambda _: delta(do, o))
            b = bench(lambda _: ma.bwd(*qkv, do, None, lse, dl, lo, hi, SCALE))
            fg = bench(lambda _: ma.fwd(*qkv, g, lo, hi, SCALE))
            bg = bench(lambda _: ma.bwd(*qkv, do, g, lse, dl, lo, hi, SCALE))
        out.append(
            f"triton fwd {f:.3f} bwd {b:.3f} + delta {d:.3f} sum {f + b + d:.3f}"
            f" (eager autograd bwd {eager:.3f}; gated fwd {fg:.3f} bwd {bg:.3f})"
        )
        try:
            m64 = mask64(flex.documents, w, row)
            qt = [x.transpose(1, 2) for x in qkv]
            kw = dict(block_mask=m64, scale=SCALE, kernel_options=dict(fwd_BLOCK_M=64))
            f = bench(lambda _: flex64(*qt, **kw))
            dt = do.transpose(1, 2)
            b = bench(lambda o: torch.autograd.grad(o, qkv, dt), lambda: flex64(*qt, **kw))
            out.append(f"| flex64 fwd {f:.3f} bwd {b:.3f} sum {f + b:.3f}")
        except Exception as e:  # a data point only
            out.append(f"| flex64 failed: {type(e).__name__}: {str(e)[:200]}")
        print(*out, flush=True)
    for backend in ("flex", "triton"):
        ms = bench(lambda _: mm.make_context(rows, *windows, backend=backend))
        print(f"make_context {backend} {ms:.3f} ms", flush=True)
    return tri


def sweep(rows, tri):
    n, row = rows.shape
    (*qkv, _), do = inputs(n * row, 2)
    lo, hi = tri.short_mask
    fwds = [
        (64, 32, 4, 2, False),
        (64, 32, 4, 3, False),
        (64, 64, 4, 2, False),
        (64, 16, 4, 3, False),
        (128, 32, 4, 2, False),
        (128, 32, 8, 2, False),
        (128, 64, 8, 2, False),
        (32, 32, 2, 2, False),
        (32, 16, 2, 3, False),
        (32, 32, 4, 2, False),
        (64, 32, 4, 2, True),
        (64, 64, 4, 2, True),
    ]
    bwds = [
        (64, 32, 4, 2, False),
        (64, 32, 8, 2, False),
        (64, 64, 4, 2, False),
        (64, 64, 8, 2, False),
        (64, 16, 4, 2, False),
        (32, 32, 4, 2, False),
        (32, 16, 2, 2, False),
        (32, 32, 2, 2, False),
        (128, 32, 8, 2, False),
        (64, 32, 4, 3, False),
        (64, 32, 4, 2, True),
        (64, 64, 4, 2, True),
    ]
    with torch.no_grad():
        o, lse = ma.fwd(*qkv, None, lo, hi, SCALE)
        dl = (do.float() * o.float()).sum(-1)[0].t().contiguous()
        for cfg in fwds:
            ma.FWD = cfg
            print(
                f"fwd {cfg}: {bench(lambda _: ma.fwd(*qkv, None, lo, hi, SCALE)):.3f}",
                flush=True,
            )
        for cfg in bwds:
            ma.BWD = cfg
            ms = bench(lambda _: ma.bwd(*qkv, do, None, lse, dl, lo, hi, SCALE))
            print(f"bwd {cfg}: {ms:.3f}", flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 64
    rows = torch.as_tensor(load_rows(sys.argv), device="cuda")
    starts = (rows == mm.BOS).sum().item() + (rows[:, 0] != mm.BOS).sum().item()
    print(
        f"{torch.cuda.get_device_name()}: {tuple(rows.shape)} rows, {starts} games, "
        f"{H} heads of {D}; FWD {ma.FWD} BWD {ma.BWD}",
        flush=True,
    )
    check(rows, WSD, "real")
    tri = timing(rows, WSD, WSD[:1])
    check(rows, WSD[:1], "real", gated=True)
    check(rows, SLIDING, "real")
    timing(rows, SLIDING, SLIDING)
    syn = torch.as_tensor(synthetic(np.random.default_rng(0), 16, 1024), device="cuda")
    for windows in (WSD, SLIDING, (1023, 64)):
        check(syn, windows, "synthetic")
    if "--sweep" in sys.argv:
        sweep(rows, tri)


if __name__ == "__main__":
    main()
