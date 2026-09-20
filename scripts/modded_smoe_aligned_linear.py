"""Expert-aligned GEMMs with the reference ScatterMoE autograd contract.

Experimental, default off; BF16 materialized gradients, suitable for ZeRO-2.
No direct optimizer-buffer mutation. Tile selections are offline and shape
specific; unsupported shapes retain the tuned reference. Gates and weight
gradients use exactly the tuned path's operations and rounding boundaries.
"""
import functools
import operator

import torch
import modded_smoe_tuned as tuned
import modded_smoe_aligned as aligned
import modded_smoe_gather_wgrad as gather_wgrad
import modded_smoe_gated as gated_backward
from modded_smoe import group

# Gated combine's FP32 add order over the top-k slots (nested tuples, each a left fold: (0, 1, 2, 3)
# is ((p0 + p1) + p2) + p3), a pointwise sum inductor fuses with the shared-expert add and the
# residual; None: the cuBLAS bmm. test_moe_nongemm.py orders finds the one equal to cuBLAS.
COMBINE = None


def bf16_round(x):
    """FP32 x rounded to BF16 (nearest, ties to even) but kept FP32, in bit arithmetic: inductor
    drops a cast to BF16 and back inside a fused kernel. Finite x."""
    b = x.view(torch.int32)
    return ((b + 0x7FFF + (b >> 16 & 1)) & -65536).view(torch.float32)


def combine(expanded, gates, order):
    """(gates.unsqueeze(1) @ expanded).squeeze(1) as FP32 adds in order, rounded to BF16 once, as
    a gemv epilogue does: BF16 products are exact in FP32, so the add order fixes every bit."""
    g = bf16_round(gates.float())

    def term(o):
        if isinstance(o, int):
            return expanded[:, o].float() * g[:, o, None]
        return functools.reduce(operator.add, map(term, o))

    return bf16_round(term(order)).to(expanded.dtype)


def config(x,w,order,k,xg,yg):
    if x.dtype!=torch.bfloat16 or w.shape[0] not in (96,128):
        return None
    rows=order.numel()
    key=(x.shape[1],w.shape[-1],k,xg,yg)
    if rows==98304:
        # Down/up-input-grad use the tuned fallback: the exploratory aligned
        # sweep used the wrong weight strides for those two operators.
        return {
            (2048,910,6,False,True):(128,256,64,8,3),
            (2048,455,1,True,True):(128,256,64,8,3),
        }.get(key)
    if rows==131072:  # d1536 E96 top-4 at 32K micro (tune_moe_shape, main-kernels/tune-d1536-*-partial.json)
        return {
            (1536,1024,4,False,True):(128,256,64,8,3),
            (512,1536,1,True,False):(64,128,64,4,3),
        }.get(key)
    if rows==393216:
        return {
            (2048,910,6,False,True):(128,256,64,8,3),
            (2048,455,1,True,True):(128,256,64,8,3),
        }.get(key)
    return None


def matmul(x,w,se,order,offsets,k,xg,yg,out=None):
    c=config(x,w,order,k,xg,yg)
    if c is None:
        return tuned.scatter2scatter(x,w,se,order,k,x_grouped=xg,y_grouped=yg,out=out)
    return aligned.linear(x,w,order,offsets,k,xg,yg,c,out=out)


class AlignedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weights,k,se,order,offsets,gates,grouped_in,grouped_out,gather,gated):
        output=matmul(x,weights,se,order,offsets,k,grouped_in,grouped_out)
        if gates is not None:
            expanded=output.view(gates.size(0),gates.size(1),output.size(-1))
            output=((gates.unsqueeze(1) @ expanded).squeeze(1) if COMBINE is None else
                    combine(expanded,gates,COMBINE))
        else:
            expanded=None
        ctx.save_for_backward(x,weights,se,order,offsets,gates,expanded)
        ctx.k=k;ctx.grouped_in=grouped_in;ctx.grouped_out=grouped_out
        ctx.gather=gather
        ctx.gated=gated
        return output

    @staticmethod
    def backward(ctx,grad_out):
        x,weights,se,order,offsets,gates,expanded=ctx.saved_tensors
        if gates is not None:
            d_gates=(expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
            if ctx.gated:
                assert ctx.grouped_in and not ctx.grouped_out and ctx.k==1
                dw=gated_backward.weight_grad(grad_out,x,gates,order,offsets)
                dx=gated_backward.input_grad(grad_out,weights.permute(0,2,1),gates,order,offsets)
                return dx,dw,None,None,None,None,d_gates,None,None,None,None
            gates_flat=gates.flatten();fan=gates.size(1)
            # Reusing a saved activation here makes AOTAutograd emit a full
            # 1.6GB clone/self-copy at64K before the overwrite. A fresh output
            # is fully written by group(), with identical values and no copy.
            grouped_grad=None
        else:
            d_gates=None;gates_flat=None;fan=1;grouped_grad=None
        if ctx.grouped_out:
            grouped_grad=grad_out
        else:
            grouped_grad=group(grad_out,order,fan_out=fan,coeff=gates_flat,out=grouped_grad)
        if ctx.gather and not ctx.grouped_in:
            dw=gather_wgrad.backward(grouped_grad,x,order,offsets,ctx.k)
            dx=None
        elif ctx.grouped_in:
            grouped_x=x;dx=None
        else:
            grouped_x=group(x,order,fan_out=ctx.k);dx=grouped_x
        if not (ctx.gather and not ctx.grouped_in):
            dw,_=tuned.group_bwd_W(grouped_grad,grouped_x,offsets,weights.size(0))
        dx=matmul(grouped_grad,weights.permute(0,2,1),se,order,offsets,
                  1,True,ctx.grouped_in,out=dx)
        if ctx.k!=1:
            dx=dx.view(x.size(0),ctx.k,dx.size(-1)).sum(-2)
        return dx,dw,None,None,None,None,d_gates,None,None,None,None


def parallel_linear(inputs,expert_weights,k,sorted_expert_idxs,sorted_scattered_idxs,
                    expert_offsets,expert_biases=None,gates=None,
                    grouped_in=False,grouped_out=False,gather=False,gated=False):
    if expert_biases is not None:
        return tuned.parallel_linear(inputs,expert_weights,k,sorted_expert_idxs,
                                     sorted_scattered_idxs,expert_offsets,expert_biases,
                                     gates,grouped_in,grouped_out)
    return AlignedLinear.apply(inputs,expert_weights,k,sorted_expert_idxs,
                               sorted_scattered_idxs,expert_offsets,gates,
                               grouped_in,grouped_out,gather,gated)
