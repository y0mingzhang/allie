"""Gather original token inputs inside expert weight-gradient GEMMs.

Avoid materializing k copies of the input. Reduction follows sorted route order
with the same dot/add boundaries as ScatterMoE groupXtY; one CTA writes each
weight tile, including explicit zero stores for unused experts. No atomics.
"""
import torch
import triton
import triton.language as tl
from modded_smoe_tuned import weight_config


@triton.jit
def _dw(DY,X,ORDER,OFF,OUT,
        SY0:tl.constexpr,SY1:tl.constexpr,SX0:tl.constexpr,SX1:tl.constexpr,
        SO0:tl.constexpr,SO1:tl.constexpr,SO2:tl.constexpr,
        K:tl.constexpr,N:tl.constexpr,FAN:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    p0,p1=tl.swizzle2d(tl.program_id(0),tl.program_id(1),
                       tl.num_programs(0),tl.num_programs(1),4)
    e=p0//tl.cdiv(K,BK)
    ki=p0%tl.cdiv(K,BK)*BK+tl.arange(0,BK)
    ni=p1*BN+tl.arange(0,BN)
    start=tl.load(OFF+e-1,e>0,other=0).to(tl.int32)
    end=tl.load(OFF+e).to(tl.int32)
    rr=start+tl.arange(0,BM)
    acc=tl.zeros((BK,BN),tl.float32)
    for block in range(tl.cdiv(end-start,BM)):
        rows=rr+block*BM
        xi=tl.load(ORDER+rows,rows<end,other=0)//FAN
        xx=tl.load(X+ki[:,None]*SX1+xi[None,:]*SX0,
                   (ki<K)[:,None]&(rows<end)[None,:],other=0)
        yy=tl.load(DY+rows[:,None]*SY0+ni[None,:]*SY1,
                   (rows<end)[:,None]&(ni<N)[None,:],other=0)
        acc+=tl.dot(xx,yy,out_dtype=tl.float32,allow_tf32=True)
    ptr=OUT+e.to(tl.int64)*SO0+ki[:,None].to(tl.int64)*SO1+ni[None,:].to(tl.int64)*SO2
    tl.store(ptr,acc,(ki<K)[:,None]&(ni<N)[None,:])


@torch.library.custom_op('allie_gather_wgrad::dw',mutates_args={'out'})
def dw_op(dy:torch.Tensor,x:torch.Tensor,order:torch.Tensor,offsets:torch.Tensor,
          out:torch.Tensor,fan:int,bm:int,bn:int,bk:int,warps:int,stages:int)->None:
    grid=(offsets.numel()*triton.cdiv(x.shape[-1],bk),triton.cdiv(dy.shape[-1],bn))
    _dw[grid](dy,x,order,offsets,out,*dy.stride(),*x.stride(),*out.stride(),
              x.shape[-1],dy.shape[-1],fan,bm,bn,bk,num_warps=warps,num_stages=stages)


def backward(dy,x,order,offsets,fan,config=None,out=None):
    if config is None:config=weight_config(dy,x,offsets.numel()) or (32,128,128,4,4)
    if out is None:out=dy.new_empty((offsets.numel(),dy.shape[-1],x.shape[-1])).transpose(1,2)
    dw_op(dy,x,order,offsets,out,fan,*config)
    return out
