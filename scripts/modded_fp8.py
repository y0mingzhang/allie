"""--fp8 dense: training forwards of the dense matmuls (attention, MLP, shared experts) in FP8 e4m3 with dynamic
tensorwise scales and FP32 accumulation; the backward is the BF16 matmul's (straight-through)."""
import torch
from torch.nn import functional as F

DENSE = False
E4M3 = torch.finfo(torch.float8_e4m3fn).max


def quantize(t):
    s = t.abs().amax().float().clamp(min=1e-12) / E4M3
    return (t.float() / s).to(torch.float8_e4m3fn).contiguous(), s  # _scaled_mm: row-major x, column-major w.T


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
