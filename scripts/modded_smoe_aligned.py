"""Experimental expert-aligned ScatterMoE GEMM; no production dispatch yet.

Each CTA owns rows from one expert. A GPU prefix sum maps a bounded static
launch grid to expert-local row tiles, avoiding masked GEMM repeats when a
route tile straddles expert boundaries. Output route order remains unchanged.
No atomics, dropped routes, CPU synchronization, or padded activation buffer.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _prefix(OFF, OUT, E: tl.constexpr, BM: tl.constexpr, EB: tl.constexpr):
    e = tl.arange(0, EB)
    stop = tl.load(OFF + e, e < E, other=0)
    start = tl.load(OFF + e - 1, (e > 0) & (e < E), other=0)
    count = tl.where(e < E, (stop - start + BM - 1) // BM, 0)
    tiles = tl.cumsum(count, 0)
    tl.store(OUT + e, tiles, e < E)


@torch.library.custom_op('allie_aligned::tile_prefix', mutates_args={'out'})
def prefix_op(offsets: torch.Tensor, out: torch.Tensor, bm: int) -> None:
    _prefix[(1,)](offsets, out, offsets.numel(), bm,
                   triton.next_power_of_2(offsets.numel()), num_warps=4)


def tile_prefix(offsets, bm):
    out = torch.empty_like(offsets)
    prefix_op(offsets, out, bm)
    return out


@triton.jit
def _aligned(X, W, ORDER, OFF, TILES, OUT,
             SX0: tl.constexpr, SX1: tl.constexpr,
             SW0: tl.constexpr, SW1: tl.constexpr, SW2: tl.constexpr,
             SO0: tl.constexpr, SO1: tl.constexpr,
             E: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
             TOPK: tl.constexpr, XG: tl.constexpr, YG: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             SEARCH: tl.constexpr):
    ntiles = tl.cdiv(N, BN)
    tile = tl.program_id(0) // ntiles
    ntile = tl.program_id(0) % ntiles
    total = tl.load(TILES + E - 1)
    if tile < total:
        # First expert whose cumulative tile count exceeds this tile.
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), E, tl.int32)
        for _ in range(SEARCH):
            mid = (lo + hi) // 2
            bound = tl.load(TILES + mid, mid < E, other=2147483647)
            right = tile >= bound
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        expert = lo
        begin = tl.load(OFF + expert - 1, expert > 0, other=0)
        end = tl.load(OFF + expert)
        tile_begin = tl.load(TILES + expert - 1, expert > 0, other=0)
        rows = begin + (tile - tile_begin) * BM + tl.arange(0, BM)
        valid = rows < end
        route = tl.load(ORDER + rows, valid, other=0)
        xi = rows if XG else route // TOPK
        oi = rows if YG else route
        ns = ntile * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            kk = start * BK + ks
            a = tl.load(X + xi[:, None] * SX0 + kk[None, :] * SX1,
                        valid[:, None] & (kk < K)[None, :], other=0)
            b = tl.load(W + expert.to(tl.int64) * SW0 + kk[:, None] * SW1 + ns[None, :] * SW2,
                        (kk < K)[:, None] & (ns < N)[None, :], other=0)
            acc = tl.dot(a, b, acc, allow_tf32=True)
        tl.store(OUT + oi[:, None] * SO0 + ns[None, :] * SO1,
                 acc, valid[:, None] & (ns < N)[None, :])


@torch.library.custom_op('allie_aligned::gemm', mutates_args={'out'})
def gemm_op(x: torch.Tensor, w: torch.Tensor, order: torch.Tensor,
            offsets: torch.Tensor, tiles: torch.Tensor, out: torch.Tensor,
            topk: int, xg: bool, yg: bool, bm: int, bn: int, bk: int,
            warps: int, stages: int) -> None:
    # sum ceil(count_e/BM) <= ceil(sum count_e/BM) + E - 1.
    grid = ((triton.cdiv(order.numel(), bm) + w.shape[0] - 1) *
            triton.cdiv(w.shape[-1], bn),)
    _aligned[grid](x, w, order, offsets, tiles, out,
        *x.stride(), *w.stride(), *out.stride(),
        w.shape[0], x.shape[-1], w.shape[-1], topk, xg, yg,
        bm, bn, bk, w.shape[0].bit_length() + 1,
        num_warps=warps, num_stages=stages)


def linear(x, w, order, offsets, topk, xg=False, yg=False,
           config=(128, 128, 32, 4, 3), tiles=None, out=None):
    bm, bn, bk, warps, stages = config
    if tiles is None:
        tiles = tile_prefix(offsets, bm)
    if out is None:
        out = x.new_empty((order.numel(), w.shape[-1]))
    gemm_op(x, w, order, offsets, tiles, out, topk, xg, yg,
            bm, bn, bk, warps, stages)
    return out
