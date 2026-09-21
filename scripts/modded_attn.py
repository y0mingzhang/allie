"""FlashAttention-2 in Triton for causal, sliding-window attention within packed games: BF16 q/k/v
(1, T, H, 64) views, FP32 accumulation. Query t sees keys [lo[t], t]; key t is seen by queries [t, hi[t]]
(bounds() derives both from game starts and the window), so every tile loop spans only its block's keys
or queries instead of a block mask. The backward recomputes P; one program per block computes dq for
its queries, then dk/dv for its keys, so no atomics (deterministic). An optional per-(token, head) gate
multiplies the output in the forward epilogue and the incoming gradient in the backward prologue."""

from typing import Optional

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
FWD = (64, 64, 4, 2, True)  # BM, BN, warps, stages, heads-fastest grid (DRAM locality: 1.53 -> 1.12 ms)
BWD = (32, 32, 2, 2, True)  # block, inner tile, warps, stages, heads-fastest grid (3.05 -> 2.22 ms)


# fmt: off
@triton.jit
def _fwd(Q, K, V, G, OUT, LSE, LO, T, qk_scale, SQ, SQH, SK, SKH, SV, SVH, SG, SGH,
         H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         HF: tl.constexpr, GATED: tl.constexpr):
    if HF:
        m0, h = tl.program_id(1) * BM, tl.program_id(0)
    else:
        m0, h = tl.program_id(0) * BM, tl.program_id(1)
    rm = m0 + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    inm = rm < T
    q = tl.load(Q + h * SQH + rm[:, None] * SQ + rd[None, :], inm[:, None], other=0.0)
    lo = tl.load(LO + rm, inm, other=0)
    m = tl.full([BM], -1e30, tl.float32)
    z = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(tl.load(LO + m0), tl.minimum(m0 + BM, T), BN):
        n = n0 + rn
        k = tl.load(K + h * SKH + n[None, :] * SK + rd[:, None], (n < T)[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        s = tl.where((n[None, :] <= rm[:, None]) & (n[None, :] >= lo[:, None]), s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, 1))
        p = tl.exp2(s - mn[:, None])
        a = tl.exp2(m - mn)
        z = z * a + tl.sum(p, 1)
        v = tl.load(V + h * SVH + n[:, None] * SV + rd[None, :], (n < T)[:, None], other=0.0)
        acc = acc * a[:, None] + tl.dot(p.to(v.dtype), v)
        m = mn
    o = acc / z[:, None]
    if GATED:
        o *= tl.load(G + rm * SG + h * SGH, inm, other=0.0).to(tl.float32)[:, None]
    tl.store(OUT + (rm[:, None] * H + h) * D + rd[None, :], o.to(OUT.dtype.element_ty), inm[:, None])
    tl.store(LSE + h * T + rm, m + tl.log2(z), inm)


@triton.jit
def _bwd(Q, K, V, DY, G, LSE, DELTA, LO, HI, DQ, DK, DV, T, qk_scale, scale,
         SQ, SQH, SK, SKH, SV, SVH, SD, SDH, SG, SGH,
         H: tl.constexpr, D: tl.constexpr, B: tl.constexpr, BI: tl.constexpr,
         HF: tl.constexpr, GATED: tl.constexpr):
    if HF:
        b0, h = tl.program_id(1) * B, tl.program_id(0)
    else:
        b0, h = tl.program_id(0) * B, tl.program_id(1)
    Q, K, V, DY, G = Q + h * SQH, K + h * SKH, V + h * SVH, DY + h * SDH, G + h * SGH
    LSE, DELTA = LSE + h * T, DELTA + h * T
    rb = b0 + tl.arange(0, B)
    ri = tl.arange(0, BI)
    rd = tl.arange(0, D)
    inb = rb < T
    out = (rb[:, None] * H + h) * D + rd[None, :]

    q = tl.load(Q + rb[:, None] * SQ + rd[None, :], inb[:, None], other=0.0)
    do = tl.load(DY + rb[:, None] * SD + rd[None, :], inb[:, None], other=0.0)
    if GATED:
        do = (do.to(tl.float32) * tl.load(G + rb * SG, inb, other=0.0).to(tl.float32)[:, None]).to(q.dtype)
    lse = tl.load(LSE + rb, inb, other=0.0)
    delta = tl.load(DELTA + rb, inb, other=0.0)
    lo = tl.load(LO + rb, inb, other=0)
    dq = tl.zeros([B, D], tl.float32)
    for n0 in range(tl.load(LO + b0), tl.minimum(b0 + B, T), BI):
        n = n0 + ri
        kt = tl.load(K + n[None, :] * SK + rd[:, None], (n < T)[None, :], other=0.0)
        vt = tl.load(V + n[None, :] * SV + rd[:, None], (n < T)[None, :], other=0.0)
        p = tl.exp2(tl.dot(q, kt) * qk_scale - lse[:, None])
        p = tl.where((n[None, :] <= rb[:, None]) & (n[None, :] >= lo[:, None]), p, 0.0)
        ds = p * (tl.dot(do, vt) - delta[:, None])
        dq = tl.dot(ds.to(kt.dtype), tl.trans(kt), dq)
    tl.store(DQ + out, (dq * scale).to(DQ.dtype.element_ty), inb[:, None])

    k = tl.load(K + rb[:, None] * SK + rd[None, :], inb[:, None], other=0.0)
    v = tl.load(V + rb[:, None] * SV + rd[None, :], inb[:, None], other=0.0)
    hi = tl.load(HI + rb, inb, other=-1)
    dk = tl.zeros([B, D], tl.float32)
    dv = tl.zeros([B, D], tl.float32)
    for m0 in range(b0, tl.load(HI + tl.minimum(b0 + B, T) - 1) + 1, BI):
        m = m0 + ri
        inm = m < T
        qt = tl.load(Q + m[None, :] * SQ + rd[:, None], inm[None, :], other=0.0)
        pt = tl.exp2(tl.dot(k, qt) * qk_scale - tl.load(LSE + m, inm, other=0.0)[None, :])
        pt = tl.where((m[None, :] >= rb[:, None]) & (m[None, :] <= hi[:, None]), pt, 0.0)
        dom = tl.load(DY + m[:, None] * SD + rd[None, :], inm[:, None], other=0.0)
        if GATED:
            dom = (dom.to(tl.float32) * tl.load(G + m * SG, inm, other=0.0).to(tl.float32)[:, None]).to(q.dtype)
        dv = tl.dot(pt.to(dom.dtype), dom, dv)
        dst = pt * (tl.dot(v, tl.trans(dom)) - tl.load(DELTA + m, inm, other=0.0)[None, :])
        dk = tl.dot(dst.to(qt.dtype), tl.trans(qt), dk)
    tl.store(DK + out, (dk * scale).to(DK.dtype.element_ty), inb[:, None])
    tl.store(DV + out, dv.to(DV.dtype.element_ty), inb[:, None])
# fmt: on


def _strides(*xs):
    for x in xs:
        assert x.stride(3) == 1 and x.shape[1] * x.stride(1) < 2**31
    return [s for x in xs for s in (x.stride(1), x.stride(2))]


def _grid(cfg, T, H):
    return (H, triton.cdiv(T, cfg[0])) if cfg[4] else (triton.cdiv(T, cfg[0]), H)


def _gate(g, q):
    return (q, 0, 0) if g is None else (g, g.stride(1), g.stride(2))


def _empty(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("allie_attn::fwd", mutates_args=())
def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: Optional[torch.Tensor],
        lo: torch.Tensor, hi: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    _, T, H, D = q.shape
    y, lse = _empty(q), q.new_empty((H, T), dtype=torch.float32)
    G, sg, sgh = _gate(g, q)
    bm, bn, warps, stages, hf = FWD
    _fwd[_grid(FWD, T, H)](q, k, v, G, y, lse, lo, T, scale * LOG2E, *_strides(q, k, v), sg, sgh,
                           H, D, bm, bn, hf, g is not None, num_warps=warps, num_stages=stages)  # fmt: skip
    return y, lse


@fwd.register_fake
def _(q, k, v, g, lo, hi, scale):
    return _empty(q), q.new_empty((q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("allie_attn::bwd", mutates_args=())
def bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dy: torch.Tensor, g: Optional[torch.Tensor],
        lse: torch.Tensor, delta: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
        scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    _, T, H, D = q.shape
    dq, dk, dv = _empty(q), _empty(k), _empty(v)
    G, sg, sgh = _gate(g, q)
    b, bi, warps, stages, hf = BWD
    _bwd[_grid(BWD, T, H)](q, k, v, dy, G, lse, delta, lo, hi, dq, dk, dv, T, scale * LOG2E, scale,
                           *_strides(q, k, v, dy), sg, sgh, H, D, b, bi, hf, g is not None,
                           num_warps=warps, num_stages=stages)  # fmt: skip
    return dq, dk, dv


@bwd.register_fake
def _(q, k, v, dy, g, lse, delta, lo, hi, scale):
    return _empty(q), _empty(k), _empty(v)


def _setup(ctx, inputs, output):
    q, k, v, g, lo, hi, ctx.scale = inputs
    ctx.save_for_backward(q, k, v, g, output[0], output[1], lo, hi)


def _backward(ctx, dy, _):
    q, k, v, g, y, lse, lo, hi = ctx.saved_tensors
    # rowsum(dO * O) with dO = dy * g and y = O * g; the gate's grad is rowsum(dy * O)
    delta = (dy.float() * y.float()).sum(-1)
    grads = bwd(q, k, v, dy, g, lse, delta[0].t().contiguous(), lo, hi, ctx.scale)
    dg = (
        None
        if g is None
        else torch.where(g == 0, 0, delta.view(g.shape) / g).to(g.dtype)
    )
    return *grads, dg, None, None, None


fwd.register_autograd(_backward, setup_context=_setup)


def bounds(starts, row, window):
    """Per-token int32 (lo, hi) from game-start flags over flattened rows (a row start is a game start)."""
    t = torch.arange(starts.numel(), device=starts.device)
    first = torch.where(starts, t, 0).cummax(0).values
    after = torch.where(starts, t, starts.numel()).flip(0).cummin(0).values.flip(0)
    last = torch.cat((after[1:], after.new_full((1,), starts.numel()))) - 1
    w = min(window, row - 1)
    return torch.maximum(first, t - w).int(), torch.minimum(last, t + w).int()


def attention(q, k, v, lo, hi, scale, gate=None):
    """softmax(scale q k^T + mask) v, times gate (broadcast over head dims) if given."""
    return fwd(q, k, v, gate, lo, hi, scale)[0]
