"""--fp8 dense: training forwards of the dense matmuls (attention, MLP, shared experts) in FP8 e4m3 with dynamic
tensorwise scales and FP32 accumulation; the backward is the BF16 matmul's (straight-through)."""
import torch
from torch.nn import functional as F

DENSE = False
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


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        (xq, sx), (wq, sw) = quantize(x.flatten(0, -2)), quantize(w)
        y = torch._scaled_mm(xq, wq.T, scale_a=sx, scale_b=sw, out_dtype=x.dtype)
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        return g @ w, g.flatten(0, -2).T @ x.flatten(0, -2)


def linear(x, w, training):
    """F.linear, or its FP8 forward in training when the shapes suit _scaled_mm (multiples of 16)."""
    if DENSE and training and x.shape[-1] % 16 == 0 and w.shape[0] % 16 == 0:
        return Linear.apply(x, w)
    return F.linear(x, w)
