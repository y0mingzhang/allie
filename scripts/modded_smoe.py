"""Triton kernels of the MoE layer (modded_moe): routed SwiGLU experts, dropless, and the router's
top-ks.

routed() runs the experts of T tokens' k routes, sorted by expert (order: the routes, token * k +
slot, in expert order; se: their experts; offsets: cumulative routes per expert):
  forward   up:        pre = x[route // k] @ up[e] (BF16, sorted rows), y = silu(a) * b, a | b = pre
            scatter:   y @ down[e] into route order, then the gated combine over each token's k slots
  backward  dgates:    expanded @ grad (bmm), expanded recomputed by the scatter unless saved
            down_wgrad: y^T @ (gate * grad[route // k]), per expert
            dx:        d pre = SwiGLU backward at pre of (gate * grad[route // k]) @ down[e]^T
            up_wgrad:  x[route // k]^T @ d pre, per expert
            scatter:   d pre @ up[e]^T into route order, summed over the k slots
up and dx give each CTA rows of one expert (tile_prefix maps the launch grid onto expert-aligned
row tiles); every weight tile of the wgrads is written by one CTA, zeros for unused experts. No
atomics: deterministic. Each output element's dot is a chain of k16 MMAs from +0 over the same BF16
operands whatever the tiling, and BF16 rounding happens where the unfused path (git history)
stored BF16, so the fused SwiGLU epilogues are bitwise that path. The scatter GEMM is vendored from
ScatterMoE (github.com/shawntan/scattermoe @ 47b5e15, Apache-2.0, (c) Shawn Tan et al.).

route() is torch.topk(s + bias, k).indices and torch.topk(s, k + 1), bit for bit with the tie
order, for FP32 (T, E <= 128) scores in one pass. ATen (sbtopk, then SmallBitonicSort): gatherTopK
keeps every element above the k-th largest in the radix order (+0 above -0, NaN largest) in index
order, then the k-th value's ties in index order; sortKeyValueInplace sorts those k slots with a
32-slot bitonic network (GTOp: NaN first, else >). Here k max extractions pick the same set, a rank
sort lays it out in gather order, and the bitonic network, reduced to the comparators between its
k valid wires, sorts it.
"""

import functools
import operator

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages); BLOCK_N per half for UP. sm_89 at d1536 h512: up 241
# registers, 96 KB shared; dx 128 registers, 48 KB (two CTAs per SM); no spills
UP = (128, 128, 64, 8, 3)
DX = (64, 128, 64, 8, 3)
SCATTER = (128, 128, 32, 4, 4)
WGRAD = (32, 128, 128, 4, 4)
WIDE = dict(up=UP, dx=DX, scatter=SCATTER, up_wgrad=WGRAD)
# hidden widths whose last 128-wide hidden tile would be at most half used (h192 pads to 256), tuned at sm_89
# d1536 h192 E256 top16, 64K tokens, every candidate bitwise: up and dx 64-wide hidden tiles (8.75 -> 7.21 ms,
# 6.53 -> 6.13), scatter BLOCK_K 64 (two forwards and a backward 21.1 -> 19.4), up wgrad BLOCK_N 256 (12.8 -> 8.0;
# BLOCK_M, its row chain, stays 32). The down wgrad keeps WGRAD: no candidate was faster
NARROW = dict(up=(256, 64, 64, 8, 2), dx=(64, 64, 128, 8, 2), scatter=(128, 128, 64, 4, 3),
              up_wgrad=(32, 256, 128, 8, 3))


def tiles(h):
    """The tile tuples for expert hidden width h."""
    return NARROW if -h % 128 >= 64 else WIDE


# fmt: off
@triton.jit
def _prefix(OFF, OUT, E: tl.constexpr, BM: tl.constexpr, EB: tl.constexpr):
    e = tl.arange(0, EB)
    stop = tl.load(OFF + e, e < E, other=0)
    start = tl.load(OFF + e - 1, (e > 0) & (e < E), other=0)
    count = tl.where(e < E, (stop - start + BM - 1) // BM, 0)
    tiles = tl.cumsum(count, 0)
    tl.store(OUT + e, tiles, e < E)


@torch.library.custom_op("allie_smoe::tile_prefix", mutates_args={"out"})
def prefix_op(offsets: torch.Tensor, out: torch.Tensor, bm: int) -> None:
    _prefix[(1,)](offsets, out, offsets.numel(), bm, triton.next_power_of_2(offsets.numel()), num_warps=4)


def tile_prefix(offsets, bm):
    """Cumulative BM-row tiles per expert."""
    out = torch.empty_like(offsets)
    prefix_op(offsets, out, bm)
    return out


@triton.jit
def _rows(TILES, OFF, tile, E: tl.constexpr, BM: tl.constexpr, SEARCH: tl.constexpr):
    """Expert of an expert-aligned row tile, its rows and their mask."""
    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), E, tl.int32)
    for _ in range(SEARCH):
        mid = (lo + hi) // 2
        right = tile >= tl.load(TILES + mid, mid < E, other=2147483647)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    first = tl.load(TILES + lo - 1, lo > 0, other=0)
    rows = tl.load(OFF + lo - 1, lo > 0, other=0) + (tile - first) * BM + tl.arange(0, BM)
    return lo, rows, rows < tl.load(OFF + lo)


@triton.jit
def _swiglu(a, b):
    return a * tl.sigmoid(a) * b


@triton.jit
def _swiglu_bwd(dy, a, b):
    s = 1 / (libdevice.exp(-a) + 1.0) * 1.0
    return dy * b * s * (a * (1.0 - s) + 1.0), dy * (a * tl.sigmoid(a))


@triton.jit
def _up(X, W, ORDER, OFF, TILES, PRE, Y,
        SX0: tl.constexpr, SX1: tl.constexpr,
        SW0: tl.constexpr, SW1: tl.constexpr, SW2: tl.constexpr,
        E: tl.constexpr, D: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr,
        SAVE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        SEARCH: tl.constexpr):
    tile, n = tl.program_id(0) // tl.cdiv(H, BN), tl.program_id(0) % tl.cdiv(H, BN)
    if tile < tl.load(TILES + E - 1):
        e, rows, valid = _rows(TILES, OFF, tile, E, BM, SEARCH)
        xi = tl.load(ORDER + rows, valid, other=0) // TOPK
        c = n * 2 * BN + tl.arange(0, 2 * BN)  # gate and value columns interleaved
        h = c // 2
        acc = tl.zeros((BM, 2 * BN), tl.float32)
        for k in range(0, D, BK):
            kk = k + tl.arange(0, BK)
            x = tl.load(X + xi[:, None] * SX0 + kk[None, :] * SX1, valid[:, None] & (kk < D)[None, :], other=0)
            w = tl.load(W + e.to(tl.int64) * SW0 + kk[:, None] * SW1 + (h + c % 2 * H)[None, :] * SW2,
                        (kk < D)[:, None] & (h < H)[None, :], other=0)
            acc = tl.dot(x, w, acc)
        a, b = tl.split(tl.reshape(acc, (BM, BN, 2)))
        a, b = a.to(Y.dtype.element_ty), b.to(Y.dtype.element_ty)
        h = n * BN + tl.arange(0, BN)
        mask = valid[:, None] & (h < H)[None, :]
        if SAVE:
            p = PRE + rows[:, None] * (2 * H) + h[None, :]
            tl.store(p, a, mask)
            tl.store(p + H, b, mask)
        y = _swiglu(a.to(tl.float32), b.to(tl.float32))
        tl.store(Y + rows[:, None] * H + h[None, :], y, mask)


@triton.jit
def _dx(DY, W, GATES, ORDER, OFF, TILES, PRE, OUT,
        SY0: tl.constexpr, SY1: tl.constexpr,
        SW0: tl.constexpr, SW1: tl.constexpr, SW2: tl.constexpr,
        E: tl.constexpr, D: tl.constexpr, H: tl.constexpr, FAN: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SEARCH: tl.constexpr):
    tile, n = tl.program_id(0) // tl.cdiv(H, BN), tl.program_id(0) % tl.cdiv(H, BN)
    if tile < tl.load(TILES + E - 1):
        e, rows, valid = _rows(TILES, OFF, tile, E, BM, SEARCH)
        route = tl.load(ORDER + rows, valid, other=0)
        gate = tl.load(GATES + route, valid, other=0).to(tl.float32)
        h = n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, D, BK):
            kk = k + tl.arange(0, BK)
            x = tl.load(DY + (route // FAN)[:, None] * SY0 + kk[None, :] * SY1,
                        valid[:, None] & (kk < D)[None, :], other=0)
            x = (x.to(tl.float32) * gate[:, None]).to(DY.dtype.element_ty)
            w = tl.load(W + e.to(tl.int64) * SW0 + kk[:, None] * SW1 + h[None, :] * SW2,
                        (kk < D)[:, None] & (h < H)[None, :], other=0)
            acc = tl.dot(x, w, acc)
        mask = valid[:, None] & (h < H)[None, :]
        p = rows[:, None] * (2 * H) + h[None, :]
        a = tl.load(PRE + p, mask, other=0).to(tl.float32)
        b = tl.load(PRE + p + H, mask, other=0).to(tl.float32)
        da, db = _swiglu_bwd(acc.to(OUT.dtype.element_ty).to(tl.float32), a, b)
        tl.store(OUT + p, da, mask)
        tl.store(OUT + p + H, db, mask)


@triton.jit
def _expert_block(E_idx, E_mask, M_in_idx, N_block, N_mask, X_ptr, stride_xm, stride_xk,
                  W_ptr, stride_we, stride_wk, stride_wn, K, acc, no_k_mask, BLOCK_K):
    K_block = tl.arange(0, BLOCK_K)
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we
    for K_block_id in range(tl.cdiv(K, BLOCK_K)):
        if no_k_mask:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc = tl.dot(x, w, acc, allow_tf32=True)
    return acc


@triton.jit
def _scatter(X_ptr, stride_xm: tl.constexpr, stride_xk: tl.constexpr,
             W_ptr, stride_we, stride_wk: tl.constexpr, stride_wn: tl.constexpr,
             Y_ptr, stride_ym: tl.constexpr, stride_yn: tl.constexpr,
             grouped_idx_ptr, expert_idxs_ptr, M, K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Y[order[i]] = X[i] @ W[se[i]] over expert-sorted rows i; a row tile straddling experts
    loops over them with the other experts' rows masked to exact zeros."""
    pid = tl.program_id(axis=0)
    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block = pid // N_BLOCK_COUNT * BLOCK_M + tl.arange(0, BLOCK_M)
    N_block = pid % N_BLOCK_COUNT * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    M_boundary_mask = M_block < M
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)
    no_k_mask = K % BLOCK_K == 0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask).to(tl.int32)
    for E_idx in range(E_first_idx, E_last_idx + 1):
        acc = _expert_block(E_idx, E_idxs == E_idx, M_block, N_block, N_mask, X_ptr, stride_xm, stride_xk,
                            W_ptr, stride_we, stride_wk, stride_wn, K, acc, no_k_mask, BLOCK_K)
    Y_blk_ptrs = Y_ptr + (M_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=M_boundary_mask[:, None] & N_mask[None, :])


@triton.jit
def _down_wgrad(DY, X, GATES, ORDER, OFF, OUT,
                SY0: tl.constexpr, SY1: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr,
                SO0: tl.constexpr, SO1: tl.constexpr, SO2: tl.constexpr,
                K: tl.constexpr, N: tl.constexpr, FAN: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """OUT[e] = X[rows of e]^T @ (gate * DY[route // FAN]), the gate multiply rounded to BF16."""
    p0, p1 = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), 4)
    e = p0 // tl.cdiv(K, BK)
    ki = p0 % tl.cdiv(K, BK) * BK + tl.arange(0, BK)
    ni = p1 * BN + tl.arange(0, BN)
    start = tl.load(OFF + e - 1, e > 0, other=0).to(tl.int32)
    end = tl.load(OFF + e).to(tl.int32)
    rr = start + tl.arange(0, BM)
    acc = tl.zeros((BK, BN), tl.float32)
    for block in range(tl.cdiv(end - start, BM)):
        rows = rr + block * BM
        route = tl.load(ORDER + rows, rows < end, other=0)
        gate = tl.load(GATES + route, rows < end, other=0).to(tl.float32)
        xx = tl.load(X + ki[:, None] * SX1 + rows[None, :] * SX0, (ki < K)[:, None] & (rows < end)[None, :], other=0)
        yy = tl.load(DY + (route // FAN)[:, None] * SY0 + ni[None, :] * SY1,
                     (rows < end)[:, None] & (ni < N)[None, :], other=0)
        yy = (yy.to(tl.float32) * gate[:, None]).to(DY.dtype.element_ty)
        acc += tl.dot(xx, yy, out_dtype=tl.float32, allow_tf32=True)
    ptr = OUT + e.to(tl.int64) * SO0 + ki[:, None].to(tl.int64) * SO1 + ni[None, :].to(tl.int64) * SO2
    tl.store(ptr, acc, (ki < K)[:, None] & (ni < N)[None, :])


@triton.jit
def _up_wgrad(DY, X, ORDER, OFF, OUT,
              SY0: tl.constexpr, SY1: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr,
              SO0: tl.constexpr, SO1: tl.constexpr, SO2: tl.constexpr,
              K: tl.constexpr, N: tl.constexpr, FAN: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """OUT[e] = X[route // FAN]^T @ DY[rows of e]: the token inputs gathered, not copied k times."""
    p0, p1 = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), 4)
    e = p0 // tl.cdiv(K, BK)
    ki = p0 % tl.cdiv(K, BK) * BK + tl.arange(0, BK)
    ni = p1 * BN + tl.arange(0, BN)
    start = tl.load(OFF + e - 1, e > 0, other=0).to(tl.int32)
    end = tl.load(OFF + e).to(tl.int32)
    rr = start + tl.arange(0, BM)
    acc = tl.zeros((BK, BN), tl.float32)
    for block in range(tl.cdiv(end - start, BM)):
        rows = rr + block * BM
        xi = tl.load(ORDER + rows, rows < end, other=0) // FAN
        xx = tl.load(X + ki[:, None] * SX1 + xi[None, :] * SX0, (ki < K)[:, None] & (rows < end)[None, :], other=0)
        yy = tl.load(DY + rows[:, None] * SY0 + ni[None, :] * SY1, (rows < end)[:, None] & (ni < N)[None, :], other=0)
        acc += tl.dot(xx, yy, out_dtype=tl.float32, allow_tf32=True)
    ptr = OUT + e.to(tl.int64) * SO0 + ki[:, None].to(tl.int64) * SO1 + ni[None, :].to(tl.int64) * SO2
    tl.store(ptr, acc, (ki < K)[:, None] & (ni < N)[None, :])


@torch.library.custom_op("allie_smoe::up", mutates_args={"pre", "y"})
def up_op(x: torch.Tensor, w: torch.Tensor, order: torch.Tensor, offsets: torch.Tensor,
          tiles: torch.Tensor, pre: torch.Tensor, y: torch.Tensor, topk: int,
          bm: int, bn: int, bk: int, warps: int, stages: int) -> None:
    e, h = w.shape[0], y.shape[1]
    grid = ((triton.cdiv(order.numel(), bm) + e - 1) * triton.cdiv(h, bn),)
    _up[grid](x, w, order, offsets, tiles, pre if pre.numel() else y, y, *x.stride(), *w.stride(),
              e, x.shape[1], h, topk, pre.numel() > 0, bm, bn, bk, e.bit_length() + 1,
              num_warps=warps, num_stages=stages)


@torch.library.custom_op("allie_smoe::dx", mutates_args={"out"})
def dx_op(dy: torch.Tensor, w: torch.Tensor, gates: torch.Tensor, order: torch.Tensor,
          offsets: torch.Tensor, tiles: torch.Tensor, pre: torch.Tensor, out: torch.Tensor,
          bm: int, bn: int, bk: int, warps: int, stages: int) -> None:
    e, h = w.shape[0], w.shape[-1]
    grid = ((triton.cdiv(order.numel(), bm) + e - 1) * triton.cdiv(h, bn),)
    _dx[grid](dy, w, gates, order, offsets, tiles, pre, out, *dy.stride(), *w.stride(),
              e, dy.shape[1], h, gates.shape[1], bm, bn, bk, e.bit_length() + 1,
              num_warps=warps, num_stages=stages)


@torch.library.custom_op("allie_smoe::scatter", mutates_args={"out"})
def scatter_op(x: torch.Tensor, w: torch.Tensor, se: torch.Tensor, order: torch.Tensor,
               out: torch.Tensor, bm: int, bn: int, bk: int, warps: int, stages: int) -> None:
    grid = (triton.cdiv(order.numel(), bm) * triton.cdiv(out.shape[1], bn),)
    _scatter[grid](x, *x.stride(), w, *w.stride(), out, *out.stride(), order, se,
                   x.shape[0], x.shape[1], out.shape[1], w.shape[0], bm, bn, bk,
                   num_warps=warps, num_stages=stages)


@torch.library.custom_op("allie_smoe::down_wgrad", mutates_args={"out"})
def down_wgrad_op(dy: torch.Tensor, x: torch.Tensor, gates: torch.Tensor, order: torch.Tensor,
                  offsets: torch.Tensor, out: torch.Tensor) -> None:
    bm, bn, bk, warps, stages = WGRAD
    grid = (offsets.numel() * triton.cdiv(x.shape[-1], bk), triton.cdiv(dy.shape[-1], bn))
    _down_wgrad[grid](dy, x, gates, order, offsets, out, *dy.stride(), *x.stride(), *out.stride(),
                      x.shape[-1], dy.shape[-1], gates.shape[1], bm, bn, bk,
                      num_warps=warps, num_stages=stages)


@torch.library.custom_op("allie_smoe::up_wgrad", mutates_args={"out"})
def up_wgrad_op(dy: torch.Tensor, x: torch.Tensor, order: torch.Tensor, offsets: torch.Tensor,
                out: torch.Tensor, fan: int, bm: int, bn: int, bk: int, warps: int, stages: int) -> None:
    grid = (offsets.numel() * triton.cdiv(x.shape[-1], bk), triton.cdiv(dy.shape[-1], bn))
    _up_wgrad[grid](dy, x, order, offsets, out, *dy.stride(), *x.stride(), *out.stride(),
                    x.shape[-1], dy.shape[-1], fan, bm, bn, bk, num_warps=warps, num_stages=stages)
# fmt: on


def up(x, w, order, offsets, k, save=True):
    """pre = x[order // k] @ w[expert] (BF16, rows in expert-sorted order) and y = silu(a) * b,
    a | b = pre; w [E, D, 2H]. save=False skips pre (returned empty)."""
    n, h2 = order.numel(), w.shape[-1]
    pre, y = x.new_empty((n, h2) if save else 0), x.new_empty(n, h2 // 2)
    cfg = tiles(h2 // 2)["up"]
    up_op(x, w, order, offsets, tile_prefix(offsets, cfg[0]), pre, y, k, *cfg)
    return pre, y


def dx(dy, w, gates, order, offsets, pre):
    """d pre from the combine's output grad dy through the gated down input grad (w [E, D, H])
    and the SwiGLU backward at pre."""
    gates = gates.contiguous()
    assert pre.is_contiguous() and pre.shape == (order.numel(), 2 * w.shape[-1])
    out = torch.empty_like(pre)
    cfg = tiles(w.shape[-1])["dx"]
    dx_op(dy, w, gates, order, offsets, tile_prefix(offsets, cfg[0]), pre, out, *cfg)
    return out


def scatter(x, w, se, order, h):
    """Rows x in expert order times their expert's w [E, K, N], written back in route order; h:
    the experts' hidden width (K is h or 2h)."""
    out = x.new_empty((order.numel(), w.shape[-1]))
    scatter_op(x, w, se, order, out, *tiles(h)["scatter"])
    return out


def down_wgrad(dy, y, gates, order, offsets):
    """The down weight's grad [E, H, D], in its own layout (AccumulateGrad keeps it as is)."""
    gates = gates.contiguous()
    out = dy.new_empty((offsets.numel(), y.shape[-1], dy.shape[-1]))
    down_wgrad_op(dy, y, gates, order, offsets, out)
    return out


def up_wgrad(dpre, x, order, offsets, k):
    """The transposed up weight's grad [E, D, 2H], a view of the up weight's [E, 2H, D] layout."""
    out = dpre.new_empty((offsets.numel(), dpre.shape[-1], x.shape[-1])).transpose(1, 2)
    up_wgrad_op(dpre, x, order, offsets, out, k, *tiles(dpre.shape[-1] // 2)["up_wgrad"])
    return out


def bf16_round(x):
    """FP32 x rounded to BF16 (nearest, ties to even) but kept FP32, in bit arithmetic: inductor
    drops a cast to BF16 and back inside a fused kernel. Finite x."""
    b = x.view(torch.int32)
    return ((b + 0x7FFF + (b >> 16 & 1)) & -65536).view(torch.float32)


def combine(expanded, gates):
    """(gates.unsqueeze(1) @ expanded).squeeze(1) as FP32 adds over the top-k slots in order, rounded
    to BF16 once, as a gemv epilogue does (BF16 products are exact in FP32, so the order fixes every
    bit): a pointwise sum inductor fuses with the shared-expert add and the residual."""
    g = bf16_round(gates.float())
    terms = (expanded[:, i].float() * g[:, i, None] for i in range(gates.shape[1]))
    return bf16_round(functools.reduce(operator.add, terms)).to(expanded.dtype)


class Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up_t, down, k, se, order, offsets, gates, remat):
        pre, y = up(x, up_t, order, offsets, k)
        expanded = scatter(y, down, se, order, y.shape[1]).view(*gates.shape, down.shape[-1])
        saved = x, up_t, down, se, order, offsets, gates, pre, y
        ctx.save_for_backward(*saved, *(() if remat else (expanded,)))
        ctx.k = k
        return combine(expanded, gates)

    @staticmethod
    def backward(ctx, grad):
        x, up_t, down, se, order, offsets, gates, pre, y, *kept = ctx.saved_tensors
        expanded = kept[0] if kept else scatter(y, down, se, order, y.shape[1]).view(*gates.shape, -1)
        dgates = (expanded @ grad.unsqueeze(-1)).squeeze(-1)
        del expanded, kept
        ddown = down_wgrad(grad, y, gates, order, offsets)
        dpre = dx(grad, down.permute(0, 2, 1), gates, order, offsets, pre)
        dup = up_wgrad(dpre, x, order, offsets, ctx.k)
        dh = scatter(dpre, up_t.permute(0, 2, 1), se, order, y.shape[1])
        dh = dh.view(x.shape[0], ctx.k, dh.shape[-1]).sum(-2)
        return dh, dup, ddown, None, None, None, None, dgates, None


def routed(x, up_t, down, k, se, order, offsets, gates, remat=True):
    """Routed SwiGLU experts, combined: x [T, D], up_t [E, D, 2H] (the up weight transposed), down
    [E, H, D], gates [T, k] BF16. remat: the backward reruns the down scatter (bit for bit) for
    the gates' grad instead of saving its [T, k, D] output."""
    if torch.is_grad_enabled():
        return Routed.apply(x, up_t, down, k, se, order, offsets, gates, remat)
    y = up(x, up_t, order, offsets, k, save=False)[1]
    return combine(scatter(y, down, se, order, y.shape[1]).view(*gates.shape, -1), gates)


MIN = tl.constexpr(-(2**31))


def _stages(n=32):
    """SortUtils.cuh bitonicSort over n slots, n / 2 threads: (a, a + stride, dir) per stage."""
    stages, size = [], 2
    while size <= n:
        for stride in (size >> i for i in range(1, size.bit_length())):
            pos = [2 * t - (t & (stride - 1)) for t in range(n // 2)]
            stages.append([(p, p + stride, size < n and p & size != 0) for p in pos])
        size *= 2
    return stages


@functools.cache
def network(k, n=32):
    """bitonicSwap's comparators between valid slots, as (wire a, wire b, dir): swap iff
    GTOp(a, b) == dir. A valid-invalid swap depends on dir alone (invalid entries sort to the end),
    so which wire sits in which slot is data independent; the k wires end in slots 0..k-1."""
    wire, net = list(range(k)) + [None] * (n - k), []
    for stage in _stages(n):
        for a, b, d in stage:
            if wire[a] is not None and wire[b] is not None:
                net.append((wire[a], wire[b], d))
            elif (wire[b] is None) == d:
                wire[a], wire[b] = wire[b], wire[a]
    assert wire[:k] == list(range(k))
    return net


@functools.cache
def _net(k, device):
    return torch.tensor(network(k), dtype=torch.int32, device=device).flatten()


@triton.jit
def _pick(cols, tile, c):
    return tl.sum(tl.where(cols[None, :] == c, tile, 0), 1)


# fmt: off
@triton.jit
def _topk(bits, alive, NET, BT: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
          NC: tl.constexpr, EP: tl.constexpr):
    """Rows' torch.topk of FP32 bits (where alive): value bits and indices, [BT, KP]."""
    lanes = tl.arange(0, EP)
    cols = tl.arange(0, KP)
    key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    key = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0x7FFFFFFF, key)
    key = tl.where(alive, key, MIN)
    sk = tl.full((BT, KP), MIN, tl.int32)
    sb = tl.zeros((BT, KP), tl.int32)
    si = tl.zeros((BT, KP), tl.int32)
    for j in tl.static_range(K):  # largest key, lowest index first: gatherTopK's set
        m = tl.max(key, 1)
        lane = tl.min(tl.where(key == m[:, None], lanes[None, :], EP), 1)
        hit = lanes[None, :] == lane[:, None]
        at = cols[None, :] == j
        sk = tl.where(at, m[:, None], sk)
        sb = tl.where(at, tl.sum(tl.where(hit, bits, 0), 1)[:, None], sb)
        si = tl.where(at, lane[:, None], si)
        key = tl.where(hit, MIN, key)
    # gather order: above the k-th value by index, then its ties by index
    kth = tl.min(tl.where(cols[None, :] < K, sk, 0x7FFFFFFF), 1)
    order = tl.where(sk == kth[:, None], EP, 0) + si
    order = tl.where(cols[None, :] < K, order, 2 * EP + cols[None, :])
    rank = tl.sum((order[:, None, :] < order[:, :, None]).to(tl.int32), 2)
    at = rank[:, :, None] == cols[None, None, :]
    vb = tl.sum(tl.where(at, sb[:, :, None], 0), 1)
    vi = tl.sum(tl.where(at, si[:, :, None], 0), 1)
    for c in tl.static_range(NC):
        a = tl.load(NET + 3 * c)
        b = tl.load(NET + 3 * c + 1)
        up = tl.load(NET + 3 * c + 2) != 0
        ba, ia = _pick(cols, vb, a), _pick(cols, vi, a)
        bb, ib = _pick(cols, vb, b), _pick(cols, vi, b)
        fa, fb = ba.to(tl.float32, bitcast=True), bb.to(tl.float32, bitcast=True)
        nan_a, nan_b = (ba & 0x7FFFFFFF) > 0x7F800000, (bb & 0x7FFFFFFF) > 0x7F800000
        swap = (((nan_a & ~nan_b) | (fa > fb)) == up)[:, None]
        ca, cb = (cols[None, :] == a) & swap, (cols[None, :] == b) & swap
        vb = tl.where(ca, bb[:, None], tl.where(cb, ba[:, None], vb))
        vi = tl.where(ca, ib[:, None], tl.where(cb, ia[:, None], vi))
    return vb, vi


@triton.jit
def _route(S, BIAS, IDX, VAL, TOP, NETK, NETS, T, E: tl.constexpr, EP: tl.constexpr,
           K: tl.constexpr, KP: tl.constexpr, NK: tl.constexpr, SP: tl.constexpr,
           NS: tl.constexpr, STATS: tl.constexpr, BT: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    lanes = tl.arange(0, EP)
    alive = (rows < T)[:, None] & (lanes < E)[None, :]
    s = tl.load(S + rows[:, None].to(tl.int64) * E + lanes[None, :], alive, other=0.0)
    b = tl.load(BIAS + lanes, lanes < E, other=0.0)
    x = (s + b[None, :]).to(tl.int32, bitcast=True)
    _, vi = _topk(x, alive, NETK, BT, K, KP, NK, EP)
    cols = tl.arange(0, KP)
    out = rows[:, None].to(tl.int64) * K + cols[None, :]
    tl.store(IDX + out, vi.to(tl.int64), (rows < T)[:, None] & (cols < K)[None, :])
    cols = tl.arange(0, SP)
    out = rows[:, None].to(tl.int64) * (K + 1) + cols[None, :]
    mask = (rows < T)[:, None] & (cols < K + 1)[None, :]
    if STATS:
        vb, vi = _topk(s.to(tl.int32, bitcast=True), alive, NETS, BT, K + 1, SP, NS, EP)
        tl.store(VAL + out, vb.to(tl.float32, bitcast=True), mask)
        tl.store(TOP + out, vi.to(tl.int64), mask)
    else:
        tl.store(VAL + out, tl.zeros((BT, SP), tl.float32), mask)
        tl.store(TOP + out, tl.zeros((BT, SP), tl.int64), mask)
# fmt: on


def route(s, bias, k, stats, rows=8, warps=4):
    """(torch.topk(s + bias, k).indices, *torch.topk(s, k + 1)) for contiguous FP32 s (T, E),
    the pair zeros unless stats."""
    t, e = s.shape
    assert s.is_contiguous() and s.dtype == bias.dtype == torch.float32 and k < e <= 128
    idx = s.new_empty(t, k, dtype=torch.long)
    val, top = s.new_empty(t, k + 1), s.new_empty(t, k + 1, dtype=torch.long)
    netk, nets = _net(k, s.device), _net(k + 1, s.device)
    _route[(triton.cdiv(t, rows),)](
        s, bias, idx, val, top, netk, nets, t, e, triton.next_power_of_2(e), k,
        triton.next_power_of_2(k), len(netk) // 3, triton.next_power_of_2(k + 1), len(nets) // 3,
        bool(stats), rows, num_warps=warps,
    )  # fmt: skip
    return idx, val, top


# Quantile Balancing bins (modded_moe.qb_*): |x| in [2^-24, 16) by its FP32 bits' exponent and top
# QB_MANT mantissa bits, a log scale of 2^QB_MANT bins per octave (bin width <= 2^-QB_MANT |x|),
# mirrored for x < 0: bin QB_LEVELS + level for x >= 0, else QB_LEVELS - 1 - level. |x| below 2^-24
# counts as 2^-24 (the two centre bins span 1.0625 x 2^-24 = 6.3e-8, about half an ulp of 1.0), 16 or
# above as just below 16 (sigmoid scores keep |x| below 1 + the bias range).
QB_MANT = 4
QB_LEVELS = 28 << QB_MANT
QB_BINS = 2 * QB_LEVELS
QB_LO, QB_HI = 0x33800000, 0x41800000 - 1  # FP32 bits of 2^-24, of 16 minus an ulp


# fmt: off
@triton.jit
def _qb_hist(S, BIAS, IDX, HIST, T, reps, E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr,
             MANT: tl.constexpr, LEVELS: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr,
             BT: tl.constexpr):
    """HIST += reps x modded_moe.qb_counts, by integer atomics: deterministic."""
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    lanes = tl.arange(0, EP)
    live = rows < T
    alive = live[:, None] & (lanes < E)[None, :]
    s = tl.load(S + rows[:, None].to(tl.int64) * E + lanes[None, :], alive, other=0.0)
    b = s + tl.load(BIAS + lanes, lanes < E, other=0.0)[None, :]
    sel = tl.zeros((BT, EP), tl.int32)
    for j in tl.static_range(K):
        i = tl.load(IDX + rows.to(tl.int64) * K + j, live, other=-1).to(tl.int32)
        sel = sel | (lanes[None, :] == i[:, None]).to(tl.int32)
    sel = sel != 0
    x = tl.max(tl.where(sel | ~alive, float("-inf"), b), 1)[:, None] - b
    a = tl.minimum(tl.maximum(tl.abs(x).to(tl.int32, bitcast=True), LO), HI)
    level = (a - LO) >> (23 - MANT)
    key = tl.where(x < 0, LEVELS - 1 - level, LEVELS + level)
    add = tl.zeros((BT, EP), tl.int32) + reps
    tl.atomic_add(HIST + lanes[None, :] * (2 * LEVELS) + key, add, alive, sem="relaxed")
# fmt: on


def qb_hist(s, bias, idx, hist, reps):
    """hist += reps x modded_moe.qb_counts(s, bias, idx) for contiguous FP32 s (T, E), int32 hist."""
    t, e = s.shape
    assert s.is_contiguous() and idx.is_contiguous() and hist.dtype == torch.int32
    assert hist.shape == (e, QB_BINS) and s.dtype == bias.dtype == torch.float32
    ep = triton.next_power_of_2(e)
    bt = max(1, 1024 // ep)
    _qb_hist[(triton.cdiv(t, bt),)](
        s, bias, idx, hist, t, reps, e, ep, idx.shape[1], QB_MANT, QB_LEVELS, QB_LO, QB_HI, bt
    )  # fmt: skip
