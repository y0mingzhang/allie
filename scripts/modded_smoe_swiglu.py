"""Routed SwiGLU experts with the activation fused into the expert GEMMs (--moe-swiglu-epilogue).

scatter-dualgather computes pre = x @ up^T (BF16), y = silu(a) * b (inductor, a | b = pre),
the down GEMM, and in backward dy = gated down input grad (BF16) then inductor's cat(da, db). Here
the up GEMM's epilogue stores pre and y, and the down input-grad GEMM's epilogue applies the SiLU
backward to its BF16-rounded tile, reading pre once and writing d pre: the separate elementwise
kernels and the dy round trip are gone, and the up GEMM still runs once per forward.

Bitwise the unfused path: each output element's dot is the same chain of k16 MMAs from +0 over
the same BF16 operands whatever the tiling (the ref kernel's masked passes over other experts add
exact zeros), BF16 rounding happens where the unfused kernels store BF16, and the pointwise math
is inductor's generated code, op for op (libdevice exp in aten.silu_backward's decomposition,
tl.sigmoid elsewhere, FP32 intermediates, FMA contraction on in both). Weight grads, gate grads,
the down GEMM and the combine are the unfused path's own calls.
"""

import modded_smoe_aligned_linear as aligned
import modded_smoe_gated as gated
import modded_smoe_gather_wgrad as gather_wgrad
import torch
import triton
import triton.language as tl
from modded_smoe_aligned import tile_prefix
from triton.language.extra import libdevice

EPILOGUE = False  # modded_train --moe-swiglu-epilogue
# (BLOCK_M, BLOCK_N per half, BLOCK_K, warps, stages); any tiles give the same bits. sm_89 at
# d1536 h512: up 241 registers, 96 KB shared; dx 128 registers, 48 KB (two CTAs per SM); no spills
UP = (128, 128, 64, 8, 3)
DX = (64, 128, 64, 8, 3)


@triton.jit
def _rows(TILES, OFF, tile, E: tl.constexpr, BM: tl.constexpr, SEARCH: tl.constexpr):
    """Expert of an expert-aligned row tile (modded_smoe_aligned), its rows and their mask."""
    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), E, tl.int32)
    for _ in range(SEARCH):
        mid = (lo + hi) // 2
        right = tile >= tl.load(TILES + mid, mid < E, other=2147483647)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    first = tl.load(TILES + lo - 1, lo > 0, other=0)
    rows = (
        tl.load(OFF + lo - 1, lo > 0, other=0) + (tile - first) * BM + tl.arange(0, BM)
    )
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
        SEARCH: tl.constexpr):  # fmt: skip
    tile, n = tl.program_id(0) // tl.cdiv(H, BN), tl.program_id(0) % tl.cdiv(H, BN)
    if tile < tl.load(TILES + E - 1):
        e, rows, valid = _rows(TILES, OFF, tile, E, BM, SEARCH)
        xi = tl.load(ORDER + rows, valid, other=0) // TOPK
        c = n * 2 * BN + tl.arange(0, 2 * BN)  # gate and value columns interleaved
        h = c // 2
        acc = tl.zeros((BM, 2 * BN), tl.float32)
        for k in range(0, D, BK):
            kk = k + tl.arange(0, BK)
            x = tl.load(
                X + xi[:, None] * SX0 + kk[None, :] * SX1,
                valid[:, None] & (kk < D)[None, :],
                other=0,
            )
            w = tl.load(
                W
                + e.to(tl.int64) * SW0
                + kk[:, None] * SW1
                + (h + c % 2 * H)[None, :] * SW2,
                (kk < D)[:, None] & (h < H)[None, :],
                other=0,
            )
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
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SEARCH: tl.constexpr):  # fmt: skip
    tile, n = tl.program_id(0) // tl.cdiv(H, BN), tl.program_id(0) % tl.cdiv(H, BN)
    if tile < tl.load(TILES + E - 1):
        e, rows, valid = _rows(TILES, OFF, tile, E, BM, SEARCH)
        route = tl.load(ORDER + rows, valid, other=0)
        gate = tl.load(GATES + route, valid, other=0).to(tl.float32)
        h = n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, D, BK):
            kk = k + tl.arange(0, BK)
            x = tl.load(
                DY + (route // FAN)[:, None] * SY0 + kk[None, :] * SY1,
                valid[:, None] & (kk < D)[None, :],
                other=0,
            )
            x = (x.to(tl.float32) * gate[:, None]).to(DY.dtype.element_ty)
            w = tl.load(
                W + e.to(tl.int64) * SW0 + kk[:, None] * SW1 + h[None, :] * SW2,
                (kk < D)[:, None] & (h < H)[None, :],
                other=0,
            )
            acc = tl.dot(x, w, acc)
        mask = valid[:, None] & (h < H)[None, :]
        p = rows[:, None] * (2 * H) + h[None, :]
        a = tl.load(PRE + p, mask, other=0).to(tl.float32)
        b = tl.load(PRE + p + H, mask, other=0).to(tl.float32)
        da, db = _swiglu_bwd(acc.to(OUT.dtype.element_ty).to(tl.float32), a, b)
        tl.store(OUT + p, da, mask)
        tl.store(OUT + p + H, db, mask)


@torch.library.custom_op("allie_swiglu::up", mutates_args={"pre", "y"})
def up_op(x: torch.Tensor, w: torch.Tensor, order: torch.Tensor, offsets: torch.Tensor,
          tiles: torch.Tensor, pre: torch.Tensor, y: torch.Tensor, topk: int,
          bm: int, bn: int, bk: int, warps: int, stages: int) -> None:  # fmt: skip
    e, h = w.shape[0], y.shape[1]
    grid = ((triton.cdiv(order.numel(), bm) + e - 1) * triton.cdiv(h, bn),)
    _up[grid](x, w, order, offsets, tiles, pre if pre.numel() else y, y, *x.stride(), *w.stride(),
              e, x.shape[1], h, topk, pre.numel() > 0, bm, bn, bk, e.bit_length() + 1,
              num_warps=warps, num_stages=stages)  # fmt: skip


@torch.library.custom_op("allie_swiglu::dx", mutates_args={"out"})
def dx_op(dy: torch.Tensor, w: torch.Tensor, gates: torch.Tensor, order: torch.Tensor,
          offsets: torch.Tensor, tiles: torch.Tensor, pre: torch.Tensor, out: torch.Tensor,
          bm: int, bn: int, bk: int, warps: int, stages: int) -> None:  # fmt: skip
    e, h = w.shape[0], w.shape[-1]
    grid = ((triton.cdiv(order.numel(), bm) + e - 1) * triton.cdiv(h, bn),)
    _dx[grid](dy, w, gates, order, offsets, tiles, pre, out, *dy.stride(), *w.stride(),
              e, dy.shape[1], h, gates.shape[1], bm, bn, bk, e.bit_length() + 1,
              num_warps=warps, num_stages=stages)  # fmt: skip


def up(x, w, order, offsets, k, save=True, config=None):
    """pre = x[order // k] @ w[expert] (BF16, rows in expert-sorted order) and y = silu(a) * b,
    a | b = pre; w [E, D, 2H]. save=False skips pre (returned empty)."""
    config = config or UP
    n, h2 = order.numel(), w.shape[-1]
    pre, y = x.new_empty((n, h2) if save else 0), x.new_empty(n, h2 // 2)
    up_op(x, w, order, offsets, tile_prefix(offsets, config[0]), pre, y, k, *config)
    return pre, y


def dx(dy, w, gates, order, offsets, pre, config=None):
    """d pre from the combine's output grad dy through the gated down input grad (w [E, D, H])
    and the SwiGLU backward at pre."""
    config = config or DX
    gates = gates.contiguous()
    assert pre.is_contiguous() and pre.shape == (order.numel(), 2 * w.shape[-1])
    out = torch.empty_like(pre)
    dx_op(
        dy, w, gates, order, offsets, tile_prefix(offsets, config[0]), pre, out, *config
    )
    return out


def combine(expanded, gates):
    if aligned.COMBINE is None:
        return (gates.unsqueeze(1) @ expanded).squeeze(1)
    return aligned.combine(expanded, gates, aligned.COMBINE)


class Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up_t, down, k, se, order, offsets, gates):
        pre, y = up(x, up_t, order, offsets, k)
        expanded = aligned.matmul(y, down, se, order, offsets, 1, True, False)
        expanded = expanded.view(*gates.shape, expanded.shape[-1])
        ctx.save_for_backward(
            x, up_t, down, se, order, offsets, gates, pre, y, expanded
        )
        ctx.k = k
        return combine(expanded, gates)

    @staticmethod
    def backward(ctx, grad):
        x, up_t, down, se, order, offsets, gates, pre, y, expanded = ctx.saved_tensors
        dgates = (expanded @ grad.unsqueeze(-1)).squeeze(-1)
        ddown = gated.weight_grad(grad, y, gates, order, offsets)
        dpre = dx(grad, down.permute(0, 2, 1), gates, order, offsets, pre)
        dup = gather_wgrad.backward(dpre, x, order, offsets, ctx.k)
        dh = aligned.matmul(
            dpre, up_t.permute(0, 2, 1), se, order, offsets, 1, True, False
        )
        dh = dh.view(x.shape[0], ctx.k, dh.shape[-1]).sum(-2)
        return dh, dup, ddown, None, None, None, None, dgates


def routed(x, up_t, down, k, se, order, offsets, gates):
    """scatter-dualgather's routed SwiGLU experts, combined: x [T, D], up_t [E, D, 2H] (the up
    weight transposed), down [E, H, D], gates [T, k] BF16."""
    if torch.is_grad_enabled():
        return Routed.apply(x, up_t, down, k, se, order, offsets, gates)
    y = up(x, up_t, order, offsets, k, save=False)[1]
    expanded = aligned.matmul(y, down, se, order, offsets, 1, True, False)
    return combine(expanded.view(*gates.shape, -1), gates)
