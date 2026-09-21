"""FlashAttention-2 in Triton for causal, sliding-window attention within packed games: BF16 (1, T, H, 64)
views, FP32 accumulation. Query t sees keys [lo[t], t]; key t is seen by queries [t, hi[t]] (bounds() derives
both from game starts and the window), so every tile loop spans only its block's keys or queries instead of
a block mask. The backward recomputes P; one program per block computes dq for its queries, then dk/dv for
its keys, so no atomics (deterministic). An optional per-(token, head) gate multiplies the output in the
forward epilogue and the incoming gradient in the backward prologue.

attention_qkv() also takes the raw projection (1, T, 3H, 64) and does the model's q/k RMS norm, rotary and
value-embedding add while loading tiles, and the backward writes d(qkv) through those ops in its epilogues,
so none of them is a separate pass over memory."""

from typing import Optional

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
EPS = tl.constexpr(1.1920928955078125e-07)  # F.rms_norm's eps: FP32 math of BF16 input
# BM, BN, warps, stages, heads-fastest grid (DRAM locality): swept on L40S at 64 x 1024 rows
FWD = (64, 64, 4, 2, True)
BWD = (32, 32, 2, 2, True)  # block, inner tile, warps, stages, heads-fastest grid


# fmt: off
@triton.jit
def _qk(X, C, S, off, coff, mask, NORM: tl.constexpr, ROPE: tl.constexpr, AX: tl.constexpr,
        HALF: tl.constexpr):
    """Halves of q or k rows at X + off (head dims reduced along AX), RMS-normed then rotated by the
    cos/sin rows at coff, rounded to BF16 where the unfused model rounds."""
    a = tl.load(X + off, mask, other=0.0).to(tl.float32)
    b = tl.load(X + off + HALF, mask, other=0.0).to(tl.float32)
    if NORM:
        r = tl.expand_dims(tl.rsqrt((tl.sum(a * a, AX) + tl.sum(b * b, AX)) / (2 * HALF) + EPS), AX)
        a = (a * r).to(X.dtype.element_ty).to(tl.float32)
        b = (b * r).to(X.dtype.element_ty).to(tl.float32)
    if ROPE:
        c = tl.load(C + coff, mask, other=0.0).to(tl.float32)
        s = tl.load(S + coff, mask, other=0.0).to(tl.float32)
        a, b = a * c + b * s, b * c - a * s
    return a.to(X.dtype.element_ty), b.to(X.dtype.element_ty)


@triton.jit
def _qk_grad(ga, gb, X, C, S, off, coff, mask, NORM: tl.constexpr, ROPE: tl.constexpr,
             AX: tl.constexpr, HALF: tl.constexpr):
    """Grads of the raw halves at X + off from those of their normed, rotated halves (FP32)."""
    if ROPE:
        c = tl.load(C + coff, mask, other=0.0).to(tl.float32)
        s = tl.load(S + coff, mask, other=0.0).to(tl.float32)
        ga, gb = ga * c - gb * s, ga * s + gb * c
    if NORM:
        a = tl.load(X + off, mask, other=0.0).to(tl.float32)
        b = tl.load(X + off + HALF, mask, other=0.0).to(tl.float32)
        r = tl.expand_dims(tl.rsqrt((tl.sum(a * a, AX) + tl.sum(b * b, AX)) / (2 * HALF) + EPS), AX)
        a, b = a * r, b * r
        m = tl.expand_dims((tl.sum(ga * a, AX) + tl.sum(gb * b, AX)) / (2 * HALF), AX)
        ga, gb = r * (ga - a * m), r * (gb - b * m)
    return ga, gb


@triton.jit
def _v(V, VE, VG, off, eoff, goff, mask, gmask, VEMB: tl.constexpr, AX: tl.constexpr):
    """v rows at V + off, plus the gated value embedding (the model's v + vg * ve) if VEMB."""
    v = tl.load(V + off, mask, other=0.0)
    if VEMB:
        g = tl.expand_dims(tl.load(VG + goff, gmask, other=0.0).to(tl.float32), AX)
        v = (v.to(tl.float32) + g * tl.load(VE + eoff, mask, other=0.0).to(tl.float32)).to(v.dtype)
    return v


@triton.jit
def _fwd(Q, K, V, C, S, G, VE, VG, OUT, LSE, LO, T, qk_scale,
         SQ, SQH, SK, SKH, SV, SVH, SG, SGH, SE, SEH, SVG, SVGH, SC,
         H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, HF: tl.constexpr,
         GATED: tl.constexpr, NORM: tl.constexpr, ROPE: tl.constexpr, VEMB: tl.constexpr):
    if HF:
        m0, h = tl.program_id(1) * BM, tl.program_id(0)
    else:
        m0, h = tl.program_id(0) * BM, tl.program_id(1)
    HALF: tl.constexpr = D // 2
    Q, K, V, G, VE, VG = Q + h * SQH, K + h * SKH, V + h * SVH, G + h * SGH, VE + h * SEH, VG + h * SVGH
    rm = m0 + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    rh = tl.arange(0, HALF)
    inm = rm < T
    qa, qb = _qk(Q, C, S, rm[:, None] * SQ + rh[None, :], rm[:, None] * SC + rh[None, :], inm[:, None],
                 NORM, ROPE, 1, HALF)
    lo = tl.load(LO + rm, inm, other=0)
    m = tl.full([BM], -1e30, tl.float32)
    z = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(tl.load(LO + m0), tl.minimum(m0 + BM, T), BN):
        n = n0 + rn
        inn = n < T
        ka, kb = _qk(K, C, S, n[None, :] * SK + rh[:, None], n[None, :] * SC + rh[:, None], inn[None, :],
                     NORM, ROPE, 0, HALF)
        s = (tl.dot(qa, ka) + tl.dot(qb, kb)) * qk_scale
        s = tl.where((n[None, :] <= rm[:, None]) & (n[None, :] >= lo[:, None]), s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, 1))
        p = tl.exp2(s - mn[:, None])
        a = tl.exp2(m - mn)
        z = z * a + tl.sum(p, 1)
        v = _v(V, VE, VG, n[:, None] * SV + rd[None, :], n[:, None] * SE + rd[None, :], n * SVG,
               inn[:, None], inn, VEMB, 1)
        acc = acc * a[:, None] + tl.dot(p.to(v.dtype), v)
        m = mn
    o = acc / z[:, None]
    if GATED:
        o *= tl.load(G + rm * SG, inm, other=0.0).to(tl.float32)[:, None]
    tl.store(OUT + (rm[:, None] * H + h) * D + rd[None, :], o.to(OUT.dtype.element_ty), inm[:, None])
    tl.store(LSE + h * T + rm, m + tl.log2(z), inm)


@triton.jit
def _bwd(Q, K, V, C, S, DY, Y, G, VE, VG, LSE, LO, HI, DQ, DK, DV, DG, DVE, DVG, T, qk_scale, scale,
         SQ, SQH, SK, SKH, SV, SVH, SD, SDH, SY, SYH, SG, SGH, SE, SEH, SVG, SVGH, SC,
         SDQ, SDQH, SDK, SDKH, SDV, SDVH,
         H: tl.constexpr, D: tl.constexpr, B: tl.constexpr, BI: tl.constexpr, HF: tl.constexpr,
         GATED: tl.constexpr, NORM: tl.constexpr, ROPE: tl.constexpr, VEMB: tl.constexpr):
    """dq (and the gate's grad) of block b's queries, then dk, dv (and the value embedding's grads) of
    its keys. rowsum(dO * O) = rowsum(dy * y) comes from the gated output y, so no separate pass."""
    if HF:
        b0, h = tl.program_id(1) * B, tl.program_id(0)
    else:
        b0, h = tl.program_id(0) * B, tl.program_id(1)
    HALF: tl.constexpr = D // 2
    Q, K, V, DY, Y, G = Q + h * SQH, K + h * SKH, V + h * SVH, DY + h * SDH, Y + h * SYH, G + h * SGH
    VE, VG, DVE, DVG, DG = VE + h * SEH, VG + h * SVGH, DVE + h * SEH, DVG + h * SVGH, DG + h * SGH
    LSE, DQ, DK, DV = LSE + h * T, DQ + h * SDQH, DK + h * SDKH, DV + h * SDVH
    rb = b0 + tl.arange(0, B)
    ri = tl.arange(0, BI)
    rd = tl.arange(0, D)
    rh = tl.arange(0, HALF)
    inb = rb < T
    bcos = rb[:, None] * SC + rh[None, :]

    qoff = rb[:, None] * SQ + rh[None, :]
    qa, qb = _qk(Q, C, S, qoff, bcos, inb[:, None], NORM, ROPE, 1, HALF)
    do = tl.load(DY + rb[:, None] * SD + rd[None, :], inb[:, None], other=0.0)
    y = tl.load(Y + rb[:, None] * SY + rd[None, :], inb[:, None], other=0.0)
    delta = tl.sum(do.to(tl.float32) * y.to(tl.float32), 1)
    if GATED:
        g = tl.load(G + rb * SG, inb, other=1.0).to(tl.float32)
        do = (do.to(tl.float32) * g[:, None]).to(do.dtype)
        tl.store(DG + rb * SG, tl.where(g == 0, 0.0, delta / g).to(DG.dtype.element_ty), inb)
    lse = tl.load(LSE + rb, inb, other=0.0)
    lo = tl.load(LO + rb, inb, other=0)
    dqa = tl.zeros([B, HALF], tl.float32)
    dqb = tl.zeros([B, HALF], tl.float32)
    for n0 in range(tl.load(LO + b0), tl.minimum(b0 + B, T), BI):
        n = n0 + ri
        inn = n < T
        ka, kb = _qk(K, C, S, n[None, :] * SK + rh[:, None], n[None, :] * SC + rh[:, None], inn[None, :],
                     NORM, ROPE, 0, HALF)
        vt = _v(V, VE, VG, n[None, :] * SV + rd[:, None], n[None, :] * SE + rd[:, None], n * SVG,
                inn[None, :], inn, VEMB, 0)
        p = tl.exp2((tl.dot(qa, ka) + tl.dot(qb, kb)) * qk_scale - lse[:, None])
        p = tl.where((n[None, :] <= rb[:, None]) & (n[None, :] >= lo[:, None]), p, 0.0)
        ds = (p * (tl.dot(do, vt) - delta[:, None])).to(ka.dtype)
        dqa = tl.dot(ds, tl.trans(ka), dqa)
        dqb = tl.dot(ds, tl.trans(kb), dqb)
    dqa, dqb = _qk_grad(dqa * scale, dqb * scale, Q, C, S, qoff, bcos, inb[:, None], NORM, ROPE, 1, HALF)
    dqoff = rb[:, None] * SDQ + rh[None, :]
    tl.store(DQ + dqoff, dqa.to(DQ.dtype.element_ty), inb[:, None])
    tl.store(DQ + dqoff + HALF, dqb.to(DQ.dtype.element_ty), inb[:, None])

    koff = rb[:, None] * SK + rh[None, :]
    ka, kb = _qk(K, C, S, koff, bcos, inb[:, None], NORM, ROPE, 1, HALF)
    eoff = rb[:, None] * SE + rd[None, :]
    v = _v(V, VE, VG, rb[:, None] * SV + rd[None, :], eoff, rb * SVG, inb[:, None], inb, VEMB, 1)
    hi = tl.load(HI + rb, inb, other=-1)
    dka = tl.zeros([B, HALF], tl.float32)
    dkb = tl.zeros([B, HALF], tl.float32)
    dv = tl.zeros([B, D], tl.float32)
    for m0 in range(b0, tl.load(HI + tl.minimum(b0 + B, T) - 1) + 1, BI):
        m = m0 + ri
        inm = m < T
        qta, qtb = _qk(Q, C, S, m[None, :] * SQ + rh[:, None], m[None, :] * SC + rh[:, None], inm[None, :],
                       NORM, ROPE, 0, HALF)
        pt = tl.exp2((tl.dot(ka, qta) + tl.dot(kb, qtb)) * qk_scale
                     - tl.load(LSE + m, inm, other=0.0)[None, :])
        pt = tl.where((m[None, :] >= rb[:, None]) & (m[None, :] <= hi[:, None]), pt, 0.0)
        dom = tl.load(DY + m[:, None] * SD + rd[None, :], inm[:, None], other=0.0)
        ym = tl.load(Y + m[:, None] * SY + rd[None, :], inm[:, None], other=0.0)
        dm = tl.sum(dom.to(tl.float32) * ym.to(tl.float32), 1)
        if GATED:
            dom = (dom.to(tl.float32) * tl.load(G + m * SG, inm, other=0.0).to(tl.float32)[:, None]).to(dom.dtype)
        dv = tl.dot(pt.to(dom.dtype), dom, dv)
        dst = (pt * (tl.dot(v, tl.trans(dom)) - dm[None, :])).to(qta.dtype)
        dka = tl.dot(dst, tl.trans(qta), dka)
        dkb = tl.dot(dst, tl.trans(qtb), dkb)
    dka, dkb = _qk_grad(dka * scale, dkb * scale, K, C, S, koff, bcos, inb[:, None], NORM, ROPE, 1, HALF)
    dkoff = rb[:, None] * SDK + rh[None, :]
    tl.store(DK + dkoff, dka.to(DK.dtype.element_ty), inb[:, None])
    tl.store(DK + dkoff + HALF, dkb.to(DK.dtype.element_ty), inb[:, None])
    tl.store(DV + rb[:, None] * SDV + rd[None, :], dv.to(DV.dtype.element_ty), inb[:, None])
    if VEMB:  # dv is also the grad of v + vg * ve
        vg = tl.load(VG + rb * SVG, inb, other=0.0).to(tl.float32)
        tl.store(DVE + eoff, (dv * vg[:, None]).to(DVE.dtype.element_ty), inb[:, None])
        ve = tl.load(VE + eoff, inb[:, None], other=0.0).to(tl.float32)
        tl.store(DVG + rb * SVG, tl.sum(dv * ve, 1).to(DVG.dtype.element_ty), inb)


def _grid(cfg, T, H):
    return (H, triton.cdiv(T, cfg[0])) if cfg[4] else (triton.cdiv(T, cfg[0]), H)


def _st(*xs):
    """(token, head) strides; absent optional tensors pass zeros."""
    assert all(x is None or x.dim() != 4 or x.shape[1] * x.stride(1) < 2**31 for x in xs)
    return [s for x in xs for s in ((0, 0) if x is None else (x.stride(1), x.stride(2)))]


def _or(x, *xs):
    return [x if y is None else y for y in xs]


def _launch_fwd(q, k, v, cos, sin, g, ve, vg, lo, scale, norm):
    _, T, H, D = q.shape
    assert D % 32 == 0 and q.stride(3) == k.stride(3) == v.stride(3) == 1
    assert cos is None or (cos.shape[0] >= T and cos.stride(1) == 1 and cos.shape[1] == D // 2)
    y = torch.empty((1, T, H, D), dtype=q.dtype, device=q.device)
    lse = q.new_empty((H, T), dtype=torch.float32)
    bm, bn, warps, stages, hf = FWD
    _fwd[_grid(FWD, T, H)](q, k, v, *_or(q, cos, sin, g, ve, vg), y, lse, lo, T, scale * LOG2E,
                           *_st(q, k, v, g, ve, vg), 0 if cos is None else cos.stride(0),
                           H, D, bm, bn, hf, g is not None, norm, cos is not None, ve is not None,
                           num_warps=warps, num_stages=stages)
    return y, lse


def _launch_bwd(q, k, v, cos, sin, dy, y, g, ve, vg, lse, lo, hi, scale, norm, dq, dk, dv):
    """Writes dq, dk, dv; returns the grads of g, ve, vg (empty when absent)."""
    _, T, H, D = q.shape
    grads = [None if x is None else _like(x, q) for x in (g, ve, vg)]
    b, bi, warps, stages, hf = BWD
    _bwd[_grid(BWD, T, H)](q, k, v, *_or(q, cos, sin), dy, y, *_or(q, g, ve, vg), lse, lo, hi, dq, dk, dv,
                           *_or(q, *grads), T, scale * LOG2E, scale,
                           *_st(q, k, v, dy, y, g, ve, vg), 0 if cos is None else cos.stride(0),
                           *_st(dq, dk, dv), H, D, b, bi, hf, g is not None, norm, cos is not None,
                           ve is not None, num_warps=warps, num_stages=stages)
    return [q.new_empty(0) if x is None else x for x in grads]


@torch.library.custom_op("allie_attn::fwd", mutates_args=())
def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: Optional[torch.Tensor],
        lo: torch.Tensor, hi: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_fwd(q, k, v, None, None, g, None, None, lo, scale, False)


@fwd.register_fake
def _(q, k, v, g, lo, hi, scale):
    return q.new_empty(q.shape), q.new_empty((q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("allie_attn::bwd", mutates_args=())
def bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dy: torch.Tensor, y: torch.Tensor,
        g: Optional[torch.Tensor], lse: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
        scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = (torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in (q, k, v))
    dg = _launch_bwd(q, k, v, None, None, dy, y, g, None, None, lse, lo, hi, scale, False, dq, dk, dv)[0]
    return dq, dk, dv, dg


@bwd.register_fake
def _(q, k, v, dy, y, g, lse, lo, hi, scale):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape), _like(g, q)


def _like(x, q):
    """A grad buffer with x's strides (the kernel writes it with them), or an empty stand-in."""
    return q.new_empty(0) if x is None else torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)


@torch.library.custom_op("allie_attn::fwd_qkv", mutates_args=())
def fwd_qkv(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, g: Optional[torch.Tensor],
            ve: Optional[torch.Tensor], vg: Optional[torch.Tensor], lo: torch.Tensor, hi: torch.Tensor,
            scale: float, norm: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_fwd(*qkv.chunk(3, 2), cos, sin, g, ve, vg, lo, scale, norm)


@fwd_qkv.register_fake
def _(qkv, cos, sin, g, ve, vg, lo, hi, scale, norm):
    _, T, H3, D = qkv.shape
    return qkv.new_empty((1, T, H3 // 3, D)), qkv.new_empty((H3 // 3, T), dtype=torch.float32)


@torch.library.custom_op("allie_attn::bwd_qkv", mutates_args=())
def bwd_qkv(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dy: torch.Tensor, y: torch.Tensor,
            g: Optional[torch.Tensor], ve: Optional[torch.Tensor], vg: Optional[torch.Tensor],
            lse: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, scale: float,
            norm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dqkv = torch.empty(qkv.shape, dtype=qkv.dtype, device=qkv.device)
    grads = _launch_bwd(*qkv.chunk(3, 2), cos, sin, dy, y, g, ve, vg, lse, lo, hi, scale, norm, *dqkv.chunk(3, 2))
    return dqkv, *grads


@bwd_qkv.register_fake
def _(qkv, cos, sin, dy, y, g, ve, vg, lse, lo, hi, scale, norm):
    return qkv.new_empty(qkv.shape), _like(g, qkv), _like(ve, qkv), _like(vg, qkv)


def _setup(ctx, inputs, output):
    q, k, v, g, lo, hi, ctx.scale = inputs
    ctx.save_for_backward(q, k, v, g, *output, lo, hi)


def _backward(ctx, dy, _):
    q, k, v, g, y, lse, lo, hi = ctx.saved_tensors
    dq, dk, dv, dg = bwd(q, k, v, dy, y, g, lse, lo, hi, ctx.scale)
    return dq, dk, dv, None if g is None else dg, None, None, None


def _setup_qkv(ctx, inputs, output):
    qkv, cos, sin, g, ve, vg, lo, hi, ctx.scale, ctx.norm = inputs
    ctx.save_for_backward(qkv, cos, sin, g, ve, vg, *output, lo, hi)


def _backward_qkv(ctx, dy, _):
    qkv, cos, sin, g, ve, vg, y, lse, lo, hi = ctx.saved_tensors
    dqkv, dg, dve, dvg = bwd_qkv(qkv, cos, sin, dy, y, g, ve, vg, lse, lo, hi, ctx.scale, ctx.norm)
    none = lambda x, dx: None if x is None else dx  # noqa: E731
    return dqkv, None, None, none(g, dg), none(ve, dve), none(vg, dvg), None, None, None, None


fwd.register_autograd(_backward, setup_context=_setup)
fwd_qkv.register_autograd(_backward_qkv, setup_context=_setup_qkv)


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


def attention_qkv(qkv, cos, sin, lo, hi, scale, gate=None, ve=None, vgate=None, norm=True):
    """attention() of the model's q, k, v from its raw projection qkv (1, T, 3H, D): q, k RMS-normed
    (if norm) and rotated by the rows of cos/sin (T, D/2), v plus vgate * ve (if given)."""
    return fwd_qkv(qkv, cos, sin, gate, ve, vgate, lo, hi, scale, norm)[0]
