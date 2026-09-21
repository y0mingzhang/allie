"""FlashAttention-2 in Triton for causal, sliding-window attention within packed games: BF16 q/k/v
(1, T, H, 64) views, FP32 accumulation. Query t sees keys [lo[t], t]; key t is seen by queries [t, hi[t]]
(bounds() derives both from game starts and the window), so every tile loop spans only its block's keys
or queries instead of a block mask. The backward recomputes P; one program per block computes dq for
its queries, then dk/dv for its keys, so no atomics (deterministic)."""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
FWD = (64, 32, 4, 2, False)  # BM, BN, warps, stages, heads-fastest grid
BWD = (64, 32, 4, 2, False)  # block, inner tile, warps, stages, heads-fastest grid


@triton.jit
def _fwd(
    Q,
    K,
    V,
    OUT,
    LSE,
    LO,
    T,
    qk_scale,
    SQ,
    SQH,
    SK,
    SKH,
    SV,
    SVH,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HF: tl.constexpr,
):
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
        k = tl.load(
            K + h * SKH + n[None, :] * SK + rd[:, None], (n < T)[None, :], other=0.0
        )
        s = tl.dot(q, k) * qk_scale
        s = tl.where(
            (n[None, :] <= rm[:, None]) & (n[None, :] >= lo[:, None]), s, float("-inf")
        )
        mn = tl.maximum(m, tl.max(s, 1))
        p = tl.exp2(s - mn[:, None])
        a = tl.exp2(m - mn)
        z = z * a + tl.sum(p, 1)
        v = tl.load(
            V + h * SVH + n[:, None] * SV + rd[None, :], (n < T)[:, None], other=0.0
        )
        acc = acc * a[:, None] + tl.dot(p.to(v.dtype), v)
        m = mn
    o = acc / z[:, None]
    tl.store(
        OUT + (rm[:, None] * H + h) * D + rd[None, :],
        o.to(OUT.dtype.element_ty),
        inm[:, None],
    )
    tl.store(LSE + h * T + rm, m + tl.log2(z), inm)


@triton.jit
def _bwd(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    LO,
    HI,
    DQ,
    DK,
    DV,
    T,
    qk_scale,
    scale,
    SQ,
    SQH,
    SK,
    SKH,
    SV,
    SVH,
    SD,
    SDH,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    BI: tl.constexpr,
    HF: tl.constexpr,
):
    if HF:
        b0, h = tl.program_id(1) * B, tl.program_id(0)
    else:
        b0, h = tl.program_id(0) * B, tl.program_id(1)
    Q, K, V, DO = Q + h * SQH, K + h * SKH, V + h * SVH, DO + h * SDH
    LSE, DELTA = LSE + h * T, DELTA + h * T
    rb = b0 + tl.arange(0, B)
    ri = tl.arange(0, BI)
    rd = tl.arange(0, D)
    inb = rb < T
    out = (rb[:, None] * H + h) * D + rd[None, :]

    q = tl.load(Q + rb[:, None] * SQ + rd[None, :], inb[:, None], other=0.0)
    do = tl.load(DO + rb[:, None] * SD + rd[None, :], inb[:, None], other=0.0)
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
        pt = tl.exp2(
            tl.dot(k, qt) * qk_scale - tl.load(LSE + m, inm, other=0.0)[None, :]
        )
        pt = tl.where(
            (m[None, :] >= rb[:, None]) & (m[None, :] <= hi[:, None]), pt, 0.0
        )
        dom = tl.load(DO + m[:, None] * SD + rd[None, :], inm[:, None], other=0.0)
        dv = tl.dot(pt.to(dom.dtype), dom, dv)
        dst = pt * (
            tl.dot(v, tl.trans(dom)) - tl.load(DELTA + m, inm, other=0.0)[None, :]
        )
        dk = tl.dot(dst.to(qt.dtype), tl.trans(qt), dk)
    tl.store(DK + out, (dk * scale).to(DK.dtype.element_ty), inb[:, None])
    tl.store(DV + out, dv.to(DV.dtype.element_ty), inb[:, None])


def _strides(*xs):
    for x in xs:
        assert x.stride(3) == 1 and x.shape[1] * x.stride(1) < 2**31
    return [s for x in xs for s in (x.stride(1), x.stride(2))]


@torch.library.custom_op("allie_attn::fwd", mutates_args=())
def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, T, H, D = q.shape
    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    lse = q.new_empty((H, T), dtype=torch.float32)
    bm, bn, warps, stages, hf = FWD
    grid = (H, triton.cdiv(T, bm)) if hf else (triton.cdiv(T, bm), H)
    _fwd[grid](
        q,
        k,
        v,
        o,
        lse,
        lo,
        T,
        scale * LOG2E,
        *_strides(q, k, v),
        H,
        D,
        bm,
        bn,
        hf,
        num_warps=warps,
        num_stages=stages,
    )
    return o, lse


@fwd.register_fake
def _(q, k, v, lo, hi, scale):
    return torch.empty_like(q, memory_format=torch.contiguous_format), q.new_empty(
        (q.shape[2], q.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("allie_attn::bwd", mutates_args=())
def bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, T, H, D = q.shape
    dq, dk, dv = (
        torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v)
    )
    b, bi, warps, stages, hf = BWD
    grid = (H, triton.cdiv(T, b)) if hf else (triton.cdiv(T, b), H)
    _bwd[grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        lo,
        hi,
        dq,
        dk,
        dv,
        T,
        scale * LOG2E,
        scale,
        *_strides(q, k, v, do),
        H,
        D,
        b,
        bi,
        hf,
        num_warps=warps,
        num_stages=stages,
    )
    return dq, dk, dv


@bwd.register_fake
def _(q, k, v, do, lse, delta, lo, hi, scale):
    return tuple(
        torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v)
    )


def _setup(ctx, inputs, output):
    q, k, v, lo, hi, ctx.scale = inputs
    ctx.save_for_backward(q, k, v, output[0], output[1], lo, hi)


def _backward(ctx, do, _):
    q, k, v, o, lse, lo, hi = ctx.saved_tensors
    delta = (do.float() * o.float()).sum(-1)[0].t().contiguous()
    return *bwd(q, k, v, do, lse, delta, lo, hi, ctx.scale), None, None, None


fwd.register_autograd(_backward, setup_context=_setup)


def bounds(starts, row, window):
    """Per-token int32 (lo, hi) from game-start flags over flattened rows (a row start is a game start)."""
    t = torch.arange(starts.numel(), device=starts.device)
    first = torch.where(starts, t, 0).cummax(0).values
    after = torch.where(starts, t, starts.numel()).flip(0).cummin(0).values.flip(0)
    last = torch.cat((after[1:], after.new_full((1,), starts.numel()))) - 1
    w = min(window, row - 1)
    return torch.maximum(first, t - w).int(), torch.minimum(last, t + w).int()


def attention(q, k, v, lo, hi, scale):
    return fwd(q, k, v, lo, hi, scale)[0]
