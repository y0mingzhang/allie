# Derived from KellerJordan/modded-nanogpt ecbb586296d3dac36fd206211f25d63bad4a6b35.
# MIT license: modded_medium_LICENSE.
import os
from collections import defaultdict
from dataclasses import dataclass

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint

import triton
import triton.language as tl
from torch import Tensor, nn

from chess_vocab import INCREMENTS_ID, SECONDS_ID
import modded_shard
from modded_arch import swiglu_hidden
from modded_moe import MoE, replay_context

dynamo.config.recompile_limit = 64
# set by modded_medium.configure: the device, args (num_layers, num_iterations) and attention backend
device = args = medium_attention = medium_attention_qkv = None


# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out


# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(
    dynamic=False, fullgraph=True
)  # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor, split_baddbmm: bool = False):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Select batched vs unbatched
    if split_baddbmm:
        BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
    else:
        aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

        # Referencing X twice causes pytorch to make a defensive copy,
        # resulting in a cudaMemcpyAsync in baddbmm.
        # For large matrices (i.e., the mlp weights), it's faster to split
        # the operation into two kernels to avoid this.
        if split_baddbmm:
            BX_matmul(B, X, out=C)  # C = B @ X
            C.add_(X, alpha=a)  # C = C + a*X  (in-place, X only read)
        else:
            aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# -----------------------------------------------------------------------------
# Compiled helpers for NorMuon by @chrisjmccormick


@torch.compile(dynamic=False, fullgraph=True)
def cautious_wd_and_update_inplace(p, v, wd_tensor, lr_tensor):
    """Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors."""
    mask = (v * p) >= 0
    wd_factor = wd_tensor.to(p.dtype)
    lr_factor = lr_tensor.to(p.dtype)
    p.copy_(p - (p * mask * wd_factor * lr_factor) - (v * lr_factor))


@torch.compile(dynamic=False, fullgraph=True)
def apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """NorMuon variance reduction. Algebraically fuses the normalization steps to minimize memory ops."""
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


# -----------------------------------------------------------------------------
# BF16 weights with FP32 master shards (ZeRO-2): each attention/MLP matrix's micro-batch grads are
# averaged over ranks straight into its NorMuon owner's FP32 row during backward

CKPT = "none"  # "eager": each training block runs its compiled graph under an eager checkpoint
CKPT_LAYERS = (
    1 << 30
)  # blocks with layer_idx < CKPT_LAYERS are checkpointed (--ckpt-frac)
DEFER = (
    False  # eager checkpoints: NorMuon's param broadcasts finish under the next forward
)
_inflight = []  # (work, param, grad) reduces of the running backward
INFLIGHT_BYTES = 1 << 29  # BF16 grads whose reduce compute has not waited on yet
_bcast = {}  # block index (-1: outside blocks) -> the param broadcasts it waits on


def sync_params(*blocks):
    """Wait for the deferred param broadcasts of these blocks (all if none given)."""
    for b in blocks or list(_bcast):
        for w in _bcast.pop(b, ()):
            w.wait()


@torch.no_grad()
def _finish(item):
    work, p, g = item
    work.wait()
    if p.main_grad is not None:  # this rank owns p: accumulate into its FP32 row
        p.main_grad.copy_(g) if p.fresh else p.main_grad.add_(g)
    p.fresh = False


@torch.no_grad()
def _drain():
    while _inflight:
        _finish(_inflight.pop(0))


@torch.no_grad()
def reduce_to_owner(p):
    """Post-accumulate hook: average this micro-batch's BF16 grad over ranks straight into its
    owner's FP32 row, then drop it. Up to INFLIGHT_BYTES of reduces in flight overlap the rest of
    the backward: a block's 6-7 grads arrive together, so a count cap made compute wait on reduces
    it had just issued."""
    if not _inflight:
        torch.autograd.Variable._execution_engine.queue_callback(_drain)
    g, p.grad = p.grad, None
    # one rank has nothing to reduce: no collective, no stream sync per param
    work = _Done()
    if dist.get_world_size() > 1:
        work = dist.reduce(g, p.owner, op=dist.ReduceOp.AVG, async_op=True)
    _inflight.append((work, p, g))
    # never the reduce just issued: a grad above the cap (width 2048's expert up grads) would stall on it at once
    while len(_inflight) > 1 and sum(x[2].nbytes for x in _inflight) > INFLIGHT_BYTES:
        _finish(_inflight.pop(0))


# -----------------------------------------------------------------------------
# NorMuon optimizer


class _Done:
    def wait(self):
        pass


class NorMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Differences from standard Muon:
    - Newton-Shulz is replaced with Polar Express for the orthogonalization step
    - NorMuon adds a low-rank variance estimator similar to Adafactor. https://arxiv.org/pdf/2510.05491
    - small 1D parameters handled here instead of in Adam
    - Cautious weight decay, a gated version of decoupled weight decay

    One param group per label, its stacked params split into chunk_size rows per rank. BF16 weights
    (p.master) keep this rank's rows as an FP32 master, their grads arrive reduced into FP32 rows
    (reduce_to_owner) and their BF16 params are rows of one buffer the owners publish into. local: this
    rank's own params (expert shards, modded_shard), all owned here, their grads reduced into their rows by
    modded_shard, updates published in place.
    """

    def __init__(
        self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, local=False
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2
        )
        self.local = local
        self.world_size = 1 if local else dist.get_world_size()
        # params whose broadcast from their owner waits for launch(), after the step's collectives
        self._deferred = [] if DEFER and self.world_size > 1 else None
        groups = defaultdict(list)
        for p in params:
            groups[p.label].append(p)
        super().__init__(
            [
                dict(params=ps, chunk_size=-(-len(ps) // self.world_size))
                for ps in groups.values()
            ],
            defaults,
        )
        rank = 0 if local else dist.get_rank()
        self._flat, self._pflat = {}, {}
        for i, group in enumerate(self.param_groups):
            ps, n = group["params"], group["chunk_size"]
            if not getattr(ps[0], "master", False):
                continue
            lo = rank * n
            if lo < len(ps):
                group["master"] = torch.stack([p.fp32 for p in ps[lo : lo + n]]).to(
                    ps[0].device
                )
            if not local:  # local rows are built at the step (step())
                self._flat[i] = torch.zeros(
                    (n, *ps[0].shape), dtype=torch.float32, device=ps[0].device
                )
            self._pflat[i] = torch.zeros(
                (n * self.world_size, *ps[0].shape),
                dtype=ps[0].dtype,
                device=ps[0].device,
            )
            for k, p in enumerate(ps):
                self._pflat[i][k].copy_(p.detach())
                p.data, p.fresh, p.owner = self._pflat[i][k], True, k // n
                owned = lo <= k < lo + n and not local
                p.main_grad = self._flat[i][k - lo] if owned else None
                del p.fp32
                if not local:
                    p.register_post_accumulate_grad_hook(reduce_to_owner)
        # by index: load_state_dict replaces the group dicts
        self._group_of = {
            p: i for i, g in enumerate(self.param_groups) for p in g["params"]
        }

    def launch(self):
        """Broadcast the updated rows from their owners in block order. Call after the step's last
        collective: later ones on this communicator would queue behind the broadcasts."""
        for p in sorted(self._deferred, key=lambda p: getattr(p, "block", -1)):
            work = dist.broadcast(p.data, p.owner, async_op=True)
            _bcast.setdefault(getattr(p, "block", -1), []).append(work)
        self._deferred.clear()
        sync_params(-1)

    @torch.no_grad()
    def step(self):
        # Efficient distributed step by @YouJiacheng, @KonstantinWilleke, @alexrgilbert,
        # @adricarda, @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal, @varunneal, @chrisjmccormick
        if (
            self._deferred is not None
        ):  # the last step's broadcasts, if a block has not run
            sync_params()
        if self.local:  # the shards' grads (modded_shard.finish: BF16, FP32 once a second micro-batch adds) as FP32 rows
            modded_shard.reset()
            for i, g in enumerate(self.param_groups):
                ps = g["params"]
                self._flat[i] = ps[0].new_empty((len(ps), *ps[0].shape), dtype=torch.float32)
                for k, p in enumerate(ps):
                    self._flat[i][k].copy_(p.part)
                    p.part = None
        rank = 0 if self.local else dist.get_rank()
        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            chunk_size = group["chunk_size"]
            flat = self._flat.get(self._group_of[params[0]])
            if flat is not None:  # already averaged into this rank's rows
                _drain()
                for p in params:
                    p.fresh = True
                # No collective remains for these rows. After step(), fresh=True
                # makes the next backward overwrite the consumed gradient rows.
                group_infos.append(dict(grad_source=flat, reduce_future=_Done()))
                continue
            padded_num_params = chunk_size * self.world_size

            stacked_grads = torch.empty(
                (padded_num_params, *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device,
            )
            for i, p in enumerate(params):
                stacked_grads[i].copy_(p.grad, non_blocking=True)
            if len(params) < padded_num_params:
                stacked_grads[len(params) :].zero_()

            grad_chunk = torch.empty_like(stacked_grads[:chunk_size])

            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            )

            group_infos.append(dict(grad_chunk=grad_chunk, reduce_future=reduce_future))

        all_gather_infos = []
        # Second pass: wait for gradients, compute updates for the local shard of parameters,
        # and launch all async all_gather operations.
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            owned = "grad_source" in info
            # BF16 weights: momentum and update math on the FP32 grad rows and master, in place
            grad_chunk = info["grad_source"] if owned else info["grad_chunk"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            num_params = min(
                chunk_size, max(0, len(params) - start_idx)
            )  # num params for this rank

            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(grad_chunk[:num_params])
            momentum_buffer = group["momentum_buffer"]
            # Apply momentum update to the persistent momentum buffer in-place
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(
                momentum_buffer, group["momentum"]
            )

            grad_shape = updated_grads.shape
            if params[module_idx].label == "attn":
                for p in params[module_idx : module_idx + num_params]:
                    assert p.label == "attn"

                updated_grads = updated_grads.view(
                    4 * grad_shape[0], grad_shape[1] // 4, grad_shape[2]
                )

            ref_param = params[module_idx]
            param_shape = ref_param.shape

            # The below shape-based heuristic assumes that matrices have their input along the
            # row dimension and their output along the columns. Gates are an exception.
            is_gate = "gate" in ref_param.label

            if "second_momentum_buffer" not in group:
                if is_gate:
                    group["second_momentum_buffer"] = torch.zeros_like(
                        updated_grads[..., :, :1]
                    )
                else:
                    group["second_momentum_buffer"] = (
                        torch.zeros_like(updated_grads[..., :, :1])
                        if param_shape[-2] >= param_shape[-1]
                        else torch.zeros_like(updated_grads[..., :1, :])
                    )
            second_momentum_buffer = group["second_momentum_buffer"]

            if "param_lr_cpu" not in group:
                # Define multipliers for ALL params in this group (global, not per-shard)
                lr_mults = []
                wd_mults = []
                for p in params:
                    # Increase learning rate for modules with larger inputs than outputs.
                    # This shape check also assumes rows=input, columns=output, so take care
                    # when changing memory layouts. @chrisjmccormick
                    shape = p.shape
                    if len(shape) >= 2:
                        shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5
                    else:
                        shape_mult = 1.0
                    lr_mults.append(shape_mult * getattr(p, "lr_mul", 1.0))
                    wd_mults.append(getattr(p, "wd_mul", 1.0))
                # Define as cpu tensors to enable Inductor constant folding
                group["param_lr_cpu"] = torch.tensor(
                    lr_mults, dtype=torch.float32, device="cpu"
                )
                group["param_wd_cpu"] = torch.tensor(
                    wd_mults, dtype=torch.float32, device="cpu"
                )

            eff_lr_all = group["param_lr_cpu"] * group["lr"]
            eff_wd_all = group["param_wd_cpu"] * group["weight_decay"] * group["lr"]

            # Slice the portion corresponding to this rank's shard
            eff_lr_cpu = eff_lr_all[module_idx : module_idx + num_params]
            eff_wd_cpu = eff_wd_all[module_idx : module_idx + num_params]

            # Compute zeropower for the entire chunk in a single, batched call.
            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(  # 3D params (MoE experts): one matrix per leading index
                    updated_grads.flatten(0, -3),
                    split_baddbmm=(ref_param.label == "mlp"),
                ).view(updated_grads.shape)

            # Note that the head orientation in O is transposed relative to QKV, so red_dim
            # is 'incorrect' for O. However, correcting this showed no improvement. @chrisjmccormick
            red_dim = -1 if (is_gate or param_shape[-2] >= param_shape[-1]) else -2

            v_chunk = apply_normuon_variance_reduction(
                v_chunk, second_momentum_buffer, group["beta2"], red_dim
            )

            v_chunk = v_chunk.view(grad_shape)

            # # "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
            flat = self._pflat.get(self._group_of[params[0]])
            direct_publish = owned and (self._deferred is not None or self.local)
            updated_params = (
                flat[start_idx : start_idx + chunk_size]
                if direct_publish else torch.empty_like(grad_chunk, dtype=params[0].dtype)
            )  # fmt: skip
            if num_params > 0:
                # Work on a stacked copy to avoid touching original params; BF16 weights update the
                # persistent FP32 master of this rank's shard instead (ZeRO-1)
                param_chunk = (
                    group["master"]
                    if owned
                    else torch.stack(params[module_idx : module_idx + num_params])
                )

                for local_idx in range(num_params):
                    cautious_wd_and_update_inplace(
                        param_chunk[local_idx],
                        v_chunk[local_idx],
                        eff_wd_cpu[local_idx],
                        eff_lr_cpu[local_idx],
                    )
            else:
                param_chunk = torch.zeros_like(v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size and not direct_publish:
                updated_params[num_params:].zero_()

            # These workspaces are not inputs to the parameter collective.
            # In ZeRO-2 no group_infos entry owns them after this point.
            del grad_chunk, updated_grads, v_chunk

            if (
                direct_publish
            ):  # owners broadcast in launch(); padding rows stay untouched
                if not self.local:
                    self._deferred += params
                continue
            stacked_params = (
                torch.empty(
                    (padded_num_params, *param_shape),
                    dtype=updated_params.dtype,
                    device=updated_params.device,
                )
                if flat is None
                else flat
            )

            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            )

            all_gather_infos.append(
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        # Final pass: wait for all_gather to complete and copy results back into original parameter tensors.
        for info in all_gather_infos:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            if stacked_params is self._pflat.get(self._group_of[orig_params[0]]):
                continue  # gathered in place
            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)
        if self.local:
            self._flat.clear()


class DistAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        label_order: list[str],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        # Group by label, with explicit ordering for execution control.
        params_by_label = defaultdict(list)
        for p in params:
            params_by_label[getattr(p, "label", None)].append(p)
        param_groups = []
        for label in label_order:
            if label in params_by_label:
                param_groups.append(dict(params=params_by_label[label]))
        # include any unlabeled params at the end (processed last)
        if None in params_by_label:
            param_groups.append(dict(params=params_by_label[None]))
        super().__init__(param_groups, defaults)
        # init state: small params (numel < 1024) use full-sized state, others use sharded
        rank = dist.get_rank() if dist.is_initialized() else 0
        for p in params:
            n = p.size(0) // self.world_size
            chunk = p if p.numel() < 1024 else p[rank * n : (rank + 1) * n]
            dtype = torch.float32 if getattr(p, "fp32_state", False) else torch.bfloat16
            exp_avg = torch.zeros_like(chunk, dtype=dtype, device=p.device)
            self.state[p] = dict(
                step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg)
            )
            if getattr(p, "adam_master", False):  # FP32 copy of this rank's rows
                self.state[p]["master"] = chunk.float()
        # DistributedAdam implementation by @vagrawal, @akash5474
        self.should_sync = (
            False  # set for the last micro-batch of the steps this optimizer takes
        )
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        self.register_backward_hooks()

    def register_backward_hooks(self):
        for group in self.param_groups:
            for param in group["params"]:
                self._reduce_scatter_hooks.append(
                    param.register_post_accumulate_grad_hook(self._sync_gradient)
                )

    @torch.no_grad()
    def _sync_gradient(self, param):
        if not self.should_sync:
            return

        grad = param.grad
        if param.numel() < 1024:
            # Small params: use all_reduce (no scatter/gather needed)
            self._reduce_scatter_futures[param] = (
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True),
                grad,
            )
        else:
            rank_size = grad.shape[0] // self.world_size
            grad_slice = torch.empty_like(grad[:rank_size])
            self._reduce_scatter_futures[param] = (
                dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ),
                grad_slice,
            )

    def copy_lm_to_embed(self):
        # by label, not group position: other Adam groups (e.g. the board) may follow
        by_label = {p.label: p for g in self.param_groups for p in g["params"]}
        lm_head, embed = by_label["lm_head"], by_label["embed"]
        lm_head_state = self.state[lm_head]
        embed_state = self.state[embed]
        embed_state["step"] = lm_head_state["step"]
        for k in ("exp_avg", "exp_avg_sq", "master"):
            if k in lm_head_state:
                embed_state[k] = lm_head_state[k].clone()
        embed.data.copy_(lm_head.data)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        all_gather_futures = []  # c10d Works: Work.wait() also releases NCCL's stashed tensors

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for param in group["params"]:
                if param not in self._reduce_scatter_futures:
                    continue

                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()

                is_small = param.numel() < 1024
                if is_small:
                    # Small params: g_slice is actually full grad, p_slice is full param
                    p_slice = param
                else:
                    rank_size = param.shape[0] // self.world_size
                    p_slice = param[rank * rank_size : (rank + 1) * rank_size]

                lr = group["lr"] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2**0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                # lr as weight decay schedule
                eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                master = state.get("master")
                w = p_slice if master is None else master
                mask = (update * w) > 0
                update.addcmul_(w, mask, value=eff_weight_decay * lr)

                if getattr(param, "decoupled_wd", 0.0):  # AdamW: w -= lr * wd * w
                    w.mul_(1 - lr * param.decoupled_wd)
                w.add_(other=update, alpha=-1.0)
                if master is not None:
                    p_slice.copy_(master)

                if not is_small:
                    all_gather_futures.append(
                        dist.all_gather_into_tensor(param, p_slice, async_op=True)
                    )

        self._reduce_scatter_futures.clear()
        for work in all_gather_futures:
            work.wait()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()  # @Grad62304977 and others

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))


# yarn implementation @classiclarryd; the WSD schedule's windows never change, so neither do the tables
class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        quarter = head_dim // 4
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=quarter, dtype=torch.float32, device=device
        )
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(quarter)])
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().bfloat16(), persistent=False)
        self.sin = nn.Buffer(theta.sin().bfloat16(), persistent=False)
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)


@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float


# Attention backend is supplied by modded_medium.py.


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = self.dim**-0.5
        bound = (3**0.5) * std  # improved init scale by @YouJiacheng
        # merged QKVO weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # Simplified layout by @chrisjmccormick
        self.qkvo_w = nn.Parameter(torch.empty(self.dim * 4, self.hdim))
        # label all modules for explicit optimizer grouping
        self.qkvo_w.label = "attn"

        with torch.no_grad():
            self.qkvo_w[: self.dim * 3].uniform_(-bound, bound)  # init QKV weights
            self.qkvo_w[self.dim * 3 :].zero_()  # init O weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(16, num_heads)
        self.attn_gate.weight.label = "attn_gate"
        self.attn_gate.weight.lr_mul = 0.1

        # only include gates on layers with value embeds used on forward pass
        if layer_idx < min(
            5, args.num_layers // 2
        ) or layer_idx >= args.num_layers - min(5, args.num_layers // 2):
            self.value_embed_gate = CastedLinear(16, num_heads)
            self.value_embed_gate.weight.label = "value_embed_gate"
            self.value_embed_gate.weight.lr_mul = 0.1

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = (
            attn_args.seqlens,
            attn_args.attn_scale,
            attn_args.bm_size,
        )

        qkv = F.linear(x, sa_lambdas[0] * self.qkvo_w[: self.dim * 3].type_as(x)).view(
            B, T, 3 * self.num_heads, self.head_dim
        )
        gate = lambda: torch.sigmoid(
            self.attn_gate(x[..., : self.attn_gate.weight.size(-1)])
        ).view(B, T, self.num_heads, 1)
        ve_gate = lambda: (
            2
            * torch.sigmoid(
                self.value_embed_gate(x[..., : self.value_embed_gate.weight.size(-1)])
            ).view(B, T, self.num_heads, 1)
        )
        if seqlens.backend == "triton":
            # q/k norm, rotary, value embeddings and the output gate inside the attention kernels
            g = gate()
            vg = None if ve is None else ve_gate()
            y = medium_attention_qkv(
                qkv, seqlens, bm_size, attn_scale, cos, sin, g,
                None if ve is None else ve.view(B, T, self.num_heads, self.head_dim), vg,
            )  # fmt: skip
            return F.linear(
                y.view(B, T, self.dim),
                sa_lambdas[1] * self.qkvo_w[self.dim * 3 :].type_as(y),
            )
        q, k, v = qkv.chunk(3, dim=-2)
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = v + ve_gate() * ve.view_as(v)  # @ KoszarskyB & @Grad62304977

        y = medium_attention(q, k, v, seqlens, bm_size, attn_scale)
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * gate()
        y = y.contiguous().view(
            B, T, self.num_heads * self.head_dim
        )  # re-assemble all head outputs side by side
        # sa_lambdas[1] pre-multiplied to O @shenberg
        return F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3 :].type_as(y))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = swiglu_hidden(dim)
        # Transposed layout to match attention weights; c_fc rows: gate then value
        self.c_fc = nn.Parameter(torch.empty(2 * hdim, dim))
        self.c_proj = nn.Parameter(torch.empty(hdim, dim))
        # label all modules for explicit optimizer grouping; NorMuon stacks a label's params, so
        # the differently shaped projection gets its own label
        self.c_fc.label = "mlp"
        self.c_proj.label = "mlp_proj"
        self.c_proj.lr_mul = 2.0

        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std  # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        a, b = F.linear(x, self.c_fc.type_as(x)).chunk(2, dim=-1)
        return F.linear(F.silu(a) * b, self.c_proj.T.type_as(x))


@torch.compile(dynamic=False, fullgraph=True, options={"emulate_precision_casts": True})
def _blend_fwd(*args):
    y = args[0] * args[1]
    for i in range(2, len(args), 2):
        y = y + args[i] * args[i + 1]
    return y


@torch.compile(dynamic=False, fullgraph=True)
def _blend_bwd(g, scales, ts):
    return [(g.float() * s).to(g.dtype) for s in scales], [g * t for t in ts]


class _Blend(torch.autograd.Function):
    """Eager's l0 * t0 + l1 * t1 + ... (FP32 0-dim l, BF16 t) bit for bit in one kernel that rounds
    each step as eager does (emulate_precision_casts, in these graphs only). One backward kernel
    writes the tensor grads and the products whose eager sums are the scalar grads (autograd's
    sum_to). Eager g * l keeps a CPU scalar's FP32 value but casts a CUDA 0-dim l to BF16 first."""

    @staticmethod
    def forward(ctx, *args):
        ctx.save_for_backward(*args)
        return _blend_fwd(*args)

    @staticmethod
    def backward(ctx, g):
        args = ctx.saved_tensors
        scales = [lam if lam.is_cpu else lam.to(g.dtype) for lam in args[::2]]
        grads, prods = _blend_bwd(g, scales, args[1::2])
        return tuple(v for p, gt in zip(prods, grads) for v in (p.sum(), gt))


@torch.compiler.disable
def _fused_blend(first, x, x0, x02, resid, w):
    """The residual blend before block i: resid * x + w0 * x0 + w1 * x02 (block 0: x0 is x)."""
    if first:
        return _Blend.apply(resid + w[0], x, w[1], x02)
    return _Blend.apply(resid, x, w[0], x0, w[1], x02)


@torch.compiler.disable
def _eager_block(module, x, attn_args, blend, ckpt=True):
    """The residual blend (x0, x02, resid, x0 weights) and the block's own compiled graph, under an
    eager checkpoint if ckpt: everything the graph saves (incl. opaque autograd Functions' tensors)
    is dropped and recomputed in backward, keeping only the block input."""
    # Create AND call the optimized bound method outside the outer Dynamo trace.
    # Passing module._compiled through that trace can unwrap it back to Python;
    # changing disable(recursive=...) alone does not preserve the compiled child.
    if not hasattr(module, "_compiled"):
        module._compiled = torch.compile(module._forward, dynamic=False, fullgraph=True)
    sync_params(module.layer_idx)

    def run(x, attn_args, x0, x02, resid, x0_weights):
        x = _fused_blend(module.layer_idx == 0, x, x0, x02, resid, x0_weights)
        return module._compiled(x, attn_args)

    if not ckpt:
        return run(x, attn_args, *blend)
    return torch.utils.checkpoint.checkpoint(
        run, x, attn_args, *blend, use_reentrant=False, context_fn=replay_context
    )


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        layer_idx: int,
        moe: tuple | None = None,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads, layer_idx)
        self.layer_idx = layer_idx
        self.mlp = MoE(dim, *moe) if moe else MLP(dim)
        if isinstance(self.mlp, MoE):  # a checkpointed block's recompute keeps expanded briefly
            self.mlp.remat = not (CKPT == "eager" and layer_idx < CKPT_LAYERS)

    def forward(self, x: Tensor, attn_args: AttnArgs, blend=None):
        if blend is None:
            return self._forward(x, attn_args)
        # blocks past --ckpt-frac run the same compiled graph without the checkpoint
        return _eager_block(
            self, x, attn_args, blend, ckpt=self.layer_idx < CKPT_LAYERS
        )

    def _forward(self, x: Tensor, attn_args: AttnArgs):
        x = x + self.attn(norm(x), attn_args)
        return x + self.mlp(norm(x))


# -----------------------------------------------------------------------------
# The main model


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


@dataclass
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int


def header_features(tokens: Tensor, same_previous: Tensor, n: int):
    """Per-token features of each game's header, on the positions predicting its moves (from
    the header's last token on; zeros elsewhere and where the header is not a game's): mover's
    and opponent's Elo (linear, Elo / 4000), with n = 2 also the base time and increment (raw
    seconds, log like the clock features); each value as 8 sin, 8 cos, itself and a presence
    flag, padded to 128 columns."""
    t = tokens.numel()
    pos = torch.arange(t, device=tokens.device)
    start = torch.where(same_previous, 0, pos).cummax(0).values
    h = start[:, None] + torch.arange(11, device=tokens.device)
    h = tokens[h.clamp(max=t - 1)]
    game = (h[:, 0] == 2348) & (pos >= start + 10)
    d = h[:, 3:11].view(t, 2, 4)
    place = torch.tensor([1000, 100, 10, 1], device=tokens.device)
    elo = (d * place).sum(-1).masked_fill((d > 9).any(-1), 0)
    white = (pos - start) % 2 == 0
    cols = [elo[:, 0].where(white, elo[:, 1]), elo[:, 1].where(white, elo[:, 0])]
    v = [c / 4000 for c in cols]
    ok = [game & (c > 0) for c in cols]
    if n == 2:
        i, inc = h[:, 1] - 192, h[:, 2] - 10
        base = torch.where(i < 5, 15 * i, torch.where(i == 5, 90, (i - 4) * 60))
        v += [torch.log1p(base.clamp(min=0)) / 10, torch.log1p(inc.clamp(min=0)) / 10]
        ok += [game & (i >= 0) & (i < 185), game & (inc >= 0) & (inc < 181)]
    v, ok = torch.stack(v, 1).float()[..., None], torch.stack(ok, 1)[..., None]
    w = torch.pi * 2.0 ** torch.arange(8, device=tokens.device)
    f = torch.cat((torch.sin(v * w), torch.cos(v * w), v, torch.ones_like(v)), -1)
    return F.pad((f * ok).flatten(1), (0, 128 - 18 * v.shape[1]))


def mask_tc_header(tokens: Tensor):
    """The tokens with every base-time and increment token (ids 10..377, used only by game headers)
    replaced by the unknown base time and unknown increment tokens."""
    inc, base = INCREMENTS_ID["*"], SECONDS_ID["*"]
    tc = (tokens >= INCREMENTS_ID["0"]) & (tokens <= base)
    return torch.where(tc, torch.where(tokens <= inc, inc, base), tokens)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        model_dim: int,
        max_seq_len: int,
        moe: tuple | None = None,
        dense_first: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        vocab_size = next_multiple_of_n(vocab_size, n=128)

        self.smear_gate = CastedLinear(16, 1)
        self.smear_gate.weight.label = "smear_gate"
        self.smear_gate.weight.lr_mul = 0.01
        self.smear_gate.weight.wd_mul = 0.0

        self.skip_gates = nn.ModuleList([CastedLinear(16, 1) for _ in range(3)])
        for sg in self.skip_gates:
            sg.weight.label = "skip_gate"
            sg.weight.lr_mul = 0.01
            sg.weight.wd_mul = 0.0

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList(
            [
                nn.Embedding(vocab_size, model_dim)
                for _ in range(min(5, num_layers // 2))
            ]
        )
        for embed in self.value_embeds:
            nn.init.zeros_(embed.weight)
        for ve in self.value_embeds:
            ve.weight.label = "value_embed"
        # MoE (modded_arch.moe_dims) after dense_first dense blocks, as in DeepSeek-V3
        moes = [moe if i >= dense_first else None for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim, num_heads, i, m) for i, m in enumerate(moes)]
        )
        self.yarn = Yarn(head_dim, max_seq_len)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, vocab_size)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)
        self.lm_head.weight.label = "lm_head"

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.embed.weight.label = "embed"

        self.embed2 = nn.Embedding(vocab_size, model_dim)
        self.embed2.weight.label = "embed2"

        # x0_lambdas separated out for different optimizer treatment (no beta smoothing)
        self.x0_lambdas = nn.Parameter(torch.zeros(2 * num_layers))
        self.x0_lambdas.label = "x0_lambdas"
        self.x0_lambdas.lr_mul = 5.0
        self.x0_lambdas.wd_mul = 0.0
        self.use_x0 = True

        pad = (
            -num_layers * 3 - 5
        ) % dist.get_world_size()  # updated: 3*num_layers instead of 4*
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    1.05
                    * torch.ones(
                        num_layers
                    ),  # resid lambdas. 1.05 init such that layer i weight is i^(num_layers-i).
                    *[
                        torch.tensor([0.5, 1.0]) for _ in range(num_layers)
                    ],  # SA lambdas
                    torch.zeros(1),  # smear_lambda
                    0.5 * torch.ones(1),  # backout_lambda
                    -1.5 * torch.ones(3),  # skip_lambdas -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )

        self.scalars.label = "scalars"
        # set learning rates
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.0
            param.wd_mul = 5.0
        for param in self.embed.parameters():
            param.wd_mul = 150.0
        for param in self.embed2.parameters():
            param.lr_mul = 75.0
            param.wd_mul = 5.0
        for param in self.lm_head.parameters():
            param.wd_mul = 150.0
        self.scalars.lr_mul = 5.0
        self.scalars.wd_mul = 0.0

        # start training with tied embed/lm_head
        self.split_embed = False

        # the retired clock and Elo input tables' init draws, so every later init keeps its values
        for _ in range(2):
            torch.empty(64, model_dim).normal_()
        # Continuous clock features (use_feats): three raw-seconds channels -> 54 log-Fourier
        # features -> model_dim, zero init; rows 54..63 pad dim 0 for optimizer sharding.
        self.use_feats = False
        self.feat_embed = nn.Embedding(64, model_dim)
        nn.init.zeros_(self.feat_embed.weight)
        self.feat_embed.weight.label = "embed2"
        self.feat_embed.weight.lr_mul = 75.0
        self.feat_embed.weight.wd_mul = 5.0
        self.board = None  # modded_board branch, added at every position
        self.header_feats = 0  # header_features on every move position (header_embed)
        self.tc_header = True  # False: mask_tc_header on every input

    def train(
        self, mode=True
    ):  # eval forwards skip the eager checkpoints that wait per block
        sync_params()
        return super().train(mode)

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        seqlens: Tensor,
        schedule_cfg: ForwardScheduleConfig,
        feat_seq: Tensor | None = None,
    ):
        assert input_seq.ndim == 1
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long

        # set configs
        skip_connections = []
        skip_in = [i * self.num_layers // 16 for i in (2, 4, 6)]
        skip_out = [9 * self.num_layers // 16 + i for i in range(3)]
        x_backout = None
        backout_layer = skip_out[-1]

        # set lambdas
        resid_lambdas = self.scalars[: 1 * self.num_layers]
        x0_lambdas = self.x0_lambdas.view(-1, 2)
        if not self.use_x0:  # drop x0 re-injection; keep the x02 column
            x0_lambdas = torch.stack((x0_lambdas[:, 0] * 0, x0_lambdas[:, 1]), 1)
        sa_lambdas = self.scalars[1 * self.num_layers : 3 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[3 * self.num_layers]
        backout_lambda = self.scalars[3 * self.num_layers + 1]
        skip_lambdas = self.scalars[3 * self.num_layers + 2 : 3 * self.num_layers + 5]

        # attention windows in 128-token blocks: long on 4 layers, short elsewhere
        long_layers = {round(i * (self.num_layers - 1) / 15) for i in (0, 4, 11, 15)}
        bm_sizes = [
            (ws_long if i in long_layers else ws_short) * 128
            for i in range(self.num_layers)
        ]

        if not self.tc_header:
            input_seq = mask_tc_header(input_seq)
        # weight-tied: use lm_head.weight for embedding lookup (or separate embed after split)
        if self.split_embed:
            x = self.embed(input_seq)
        else:
            x = F.embedding(input_seq, self.lm_head.weight)
        if self.use_feats and feat_seq is not None:
            t = feat_seq[:, : self.use_feats].float()
            v = torch.log1p(t.clamp(min=0))[..., None] / 10
            w = torch.pi * 2.0 ** torch.arange(8, device=t.device)
            f = torch.cat(
                (torch.sin(v * w), torch.cos(v * w), v, torch.ones_like(v)), -1
            )
            f = (f * (t >= 0)[..., None]).flatten(1)  # absent values (-1) give zeros
            f = F.pad(f, (0, 64 - f.shape[1]))
            x = x + f.type_as(x) @ self.feat_embed.weight.type_as(x)
        if self.header_feats:
            f = header_features(input_seq, seqlens.same_previous, self.header_feats)
            x = x + f.type_as(x) @ self.header_embed.weight.type_as(x)
        x = x + self.board(seqlens.board, x.dtype)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        # dropping first layer updates this to .12 ... 012
        ve = ve + [None] * (self.num_layers - 2 * len(ve)) + ve

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(
            self.smear_gate(x[1:, : self.smear_gate.weight.size(-1)])
        )
        x = torch.cat(
            [x[:1], x[1:] + smear_gate_out * x[:-1] * seqlens.same_previous[1:, None]]
        )
        x = x0 = norm(x[None])
        x02 = norm(self.embed2(input_seq)[None])

        cos, sin = self.yarn.cos, self.yarn.sin
        skip_idx = 0
        for i in range(self.num_layers):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                cos=cos,
                sin=sin,
                attn_scale=self.yarn.attn_scale,
            )
            if i in skip_out:
                skip_gate_out = (
                    torch.sigmoid(skip_lambdas[skip_idx])
                    * 2
                    * torch.sigmoid(
                        self.skip_gates[skip_idx](
                            x0[..., : self.skip_gates[skip_idx].weight.size(-1)]
                        )
                    )
                )
                skip_idx += 1
                x = x + skip_gate_out * skip_connections.pop()
            if (
                CKPT == "eager" and self.training
            ):  # the blend runs (and is recomputed) in the block
                x = self.blocks[i](
                    x, attn_args, blend=(x0, x02, resid_lambdas[i], x0_lambdas[i])
                )
            elif i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0, 0]) * x + x0_lambdas[0, 1] * x02
                x = self.blocks[i](x, attn_args)
            else:
                x = (
                    resid_lambdas[i] * x
                    + x0_lambdas[i, 0] * x0
                    + x0_lambdas[i, 1] * x02
                )
                x = self.blocks[i](x, attn_args)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # back out contributions from first 2/3 layers that are only required for downstream context and not direct prediction
        x -= backout_lambda * x_backout
        x = norm(x)

        logits = self.lm_head(x)
        return 23 * torch.sigmoid((logits + 5) / 7.5)


# -----------------------------------------------------------------------------
# Optimizers and schedule


class TrainingManager:
    """
    Manages three optimizers for Adam embed/lm_head, Adam scalars, and Muon weight matrices.
    Notable Features:
        1. Scalars are given higher momentum terms to smooth learning @ChrisJMcCormick
        2. Adam optimizers are only stepped on odd steps @classiclarryd
        3. Adam optimizers have hooks to start gradient communication during backwards pass @akash5474
        4. Embed/lm_head weights and optimizer state split at the schedule's split step @classiclarryd
    The schedule (modded_wsd.Schedule) sets the learning rates, Muon momentum and multi-token
    prediction weights; the batch size and attention windows are constant.
    """

    def __init__(self, model, schedule, decay_start=-1, end_step=None):
        self.model, self.schedule = model, schedule
        self.decay_start, self.end_step = decay_start, end_step
        self.mtp_weights_schedule = [
            torch.tensor(schedule.mtp(s), device=device)
            for s in range(args.num_iterations + 1)
        ]
        self.split_step = schedule.split_step
        self.batch_size = schedule.batch_rows * 1024
        self.ws_short, self.ws_long = (
            11,
            23,
        )  # full history within each original 1024-token row

        adam_labels = [
            "lm_head",
            "value_embed",
            "smear_gate",
            "skip_gate",
            "x0_lambdas",
            "embed2",
            "embed",
            "board",
            "router",
        ]
        scalar_labels = ["scalars"]
        muon_labels = ["attn_gate", "value_embed_gate", "attn", "mlp", "mlp_proj"]
        muon_labels += ["moe", "moe_up", "mlp_shared", "mlp_shared_up"]
        self.moe = [m for m in model.modules() if isinstance(m, MoE)]
        adam_params = [
            p for p in model.parameters() if getattr(p, "label", None) in adam_labels
        ]
        scalar_params = [
            p for p in model.parameters() if getattr(p, "label", None) in scalar_labels
        ]
        muon_params = [
            p for p in model.parameters() if getattr(p, "label", None) in muon_labels
            and not getattr(p, "local", False)
        ]  # fmt: skip
        assert set(getattr(p, "label", None) for p in model.parameters()) <= set(
            adam_labels + scalar_labels + muon_labels
        ), "All params must have label"

        self.adam_opt = DistAdam(
            adam_params,
            adam_labels,
            lr=0.004,
            betas=(0.8, 0.95),
            eps=1e-8,
            weight_decay=0.005,
        )
        self.scalar_opt = DistAdam(
            scalar_params,
            scalar_labels,
            lr=0.008,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.005,
        )
        muon = dict(lr=0.015, momentum=0.95, beta2=0.95, weight_decay=1.2)
        self.muon_opt = NorMuon(muon_params, **muon)
        self.optimizers = [self.adam_opt, self.scalar_opt, self.muon_opt]
        local = [p for p in model.parameters() if getattr(p, "local", False)]
        if local:  # expert shards: the same NorMuon, rank-local
            self.optimizers.append(NorMuon(local, **muon, local=True))
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        self.adam_every = False  # Adam optimizers step on every step, not odd ones only

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights=self.mtp_weights, ws_short=self.ws_short, ws_long=self.ws_long
        )

    def advance_schedule(self, step: int):
        self.mtp_weights = self.mtp_weights_schedule[step]

    def activate_hooks(self, step: int):
        """Before the last micro-batch: the Adam optimizers reduce grads on the steps they take."""
        self.adam_opt.should_sync = self.scalar_opt.should_sync = (
            self.adam_every or step % 2 == 1
        )

    def step_optimizers(self, step: int):
        step_lr = self.schedule.lr(step, self.decay_start, self.end_step)
        muons = [opt for opt in self.optimizers if isinstance(opt, NorMuon)]
        for group in (g for opt in muons for g in opt.param_groups):
            group["momentum"] = self.schedule.momentum(step)

        for opt in self.optimizers:
            # Adam on odd steps only, unless adam_every
            if opt in muons or self.adam_every or step % 2 == 1:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * step_lr
                opt.step()
                opt.zero_grad(set_to_none=True)
        self.adam_opt.should_sync = self.scalar_opt.should_sync = False

        # prop bias rule: full speed for the first 80% of training, then linearly to 0 at the end
        # (DeepSeek-V3 stops bias updates in its final phase)
        if self.moe:
            n = args.num_iterations
            scale = min(1.0, max(0.0, (n - step) / (0.2 * n)))
            for m in self.moe:
                m.rebalance(scale)

        if step == self.split_step:
            self.adam_opt.copy_lm_to_embed()
            self.model.split_embed = True

        if self.muon_opt._deferred:
            self.muon_opt.launch()
