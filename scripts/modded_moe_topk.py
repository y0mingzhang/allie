"""torch.topk(x, k) over the last dim of a CUDA (T, E <= 128) FP32 tensor, bit for bit with the tie
order, as one Triton pass; the MoE router's two top-ks (routing on s + bias, stats on s) share it.

ATen (sbtopk, then SmallBitonicSort): gatherTopK keeps every element above the k-th largest in the
radix order (+0 above -0, NaN largest) in index order, then the k-th value's ties in index order;
sortKeyValueInplace sorts those k slots with a 32-slot bitonic network (GTOp: NaN first, else >).
Here k max extractions pick the same set, a rank sort lays it out in gather order, and the bitonic
network, reduced to the comparators between its k valid wires, sorts it.
"""

import functools

import torch
import triton
import triton.language as tl

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


@triton.jit
def _topk(bits, alive, NET, BT: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
          NC: tl.constexpr, EP: tl.constexpr):  # fmt: skip
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
           NS: tl.constexpr, STATS: tl.constexpr, BT: tl.constexpr):  # fmt: skip
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
