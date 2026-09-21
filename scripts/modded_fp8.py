"""--fp8 dense: training forwards of the dense matmuls (attention, MLP, shared experts) in FP8 e4m3 with dynamic
tensorwise scales and FP32 accumulation; the backward is the BF16 matmul's (straight-through). dense-dgrad also runs
the input gradient in FP8, dense-all the weight gradient too."""
import torch
from torch.nn import functional as F

DENSE = False
BWD = ()
E4M3 = torch.finfo(torch.float8_e4m3fn).max


def round_e4m3(t):
    """FP32 RNE onto the finite E4M3 grid before the hardware cast.

    Triton on sm89 can double-round FP32 through BF16, even with rtne.
    Grid values cast exactly. The caller supplies finite values in [-448,448].
    Normal values use the same integer rounding as the expert quantizer;
    subnormals round to multiples of 2^-9, retaining signed zero.
    """
    bits = t.view(torch.int32)
    sign = bits & -0x80000000
    magnitude = bits & 0x7fffffff
    normal = ((magnitude + 0x7ffff + ((magnitude >> 20) & 1)) & -0x100000) | sign
    sub = (torch.round(t.abs() * 512.0) / 512.0).view(torch.int32) | sign
    return torch.where(t.abs() < 0.015625, sub, normal).view(torch.float32)


def quantize(t):
    s = t.abs().amax().float().clamp(min=1e-12) / E4M3
    v = (t.float() / s).clamp(-E4M3, E4M3)
    return round_e4m3(v).to(torch.float8_e4m3fn).contiguous(), s  # _scaled_mm: row-major x, column-major w.T


def mm(a, b):
    """a @ b.T for 2D a, b in FP8."""
    (aq, sa), (bq, sb) = quantize(a), quantize(b)
    return torch._scaled_mm(aq, bq.T, scale_a=sa, scale_b=sb, out_dtype=a.dtype)


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return mm(x.flatten(0, -2), w).view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g2, x2 = g.flatten(0, -2), x.flatten(0, -2)
        dx = mm(g2, w.T).view_as(x) if "dgrad" in BWD else g @ w
        dw = mm(g2.T, x2.T) if "wgrad" in BWD else g2.T @ x2
        return dx, dw


def linear(x, w, training):
    """F.linear, or its FP8 forward in training when the shapes suit _scaled_mm (multiples of 16)."""
    if DENSE and training and x.shape[-1] % 16 == 0 and w.shape[0] % 16 == 0:
        return Linear.apply(x, w)
    return F.linear(x, w)
