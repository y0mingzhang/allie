"""Research: gather and gate output gradients inside both down-backward GEMMs.

BF16 rounding after the gate multiply is mandatory: it reproduces the existing
group() materialization. These kernels do not change the gate-gradient matmul.
"""
import torch
import triton
import triton.language as tl
from modded_smoe_aligned import tile_prefix
from modded_smoe_tiles import WEIGHT


@triton.jit
def _dw(DY,X,GATES,ORDER,OFF,OUT,
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
        route=tl.load(ORDER+rows,rows<end,other=0)
        gate=tl.load(GATES+route,rows<end,other=0).to(tl.float32)
        xx=tl.load(X+ki[:,None]*SX1+rows[None,:]*SX0,
                   (ki<K)[:,None]&(rows<end)[None,:],other=0)
        yy=tl.load(DY+(route//FAN)[:,None]*SY0+ni[None,:]*SY1,
                   (rows<end)[:,None]&(ni<N)[None,:],other=0)
        yy=(yy.to(tl.float32)*gate[:,None]).to(DY.dtype.element_ty)
        acc+=tl.dot(xx,yy,out_dtype=tl.float32,allow_tf32=True)
    ptr=OUT+e.to(tl.int64)*SO0+ki[:,None].to(tl.int64)*SO1+ni[None,:].to(tl.int64)*SO2
    tl.store(ptr,acc,(ki<K)[:,None]&(ni<N)[None,:])


@triton.jit
def _dx(DY,W,GATES,ORDER,OFF,TILES,OUT,
        SY0:tl.constexpr,SY1:tl.constexpr,
        SW0:tl.constexpr,SW1:tl.constexpr,SW2:tl.constexpr,
        E:tl.constexpr,D:tl.constexpr,H:tl.constexpr,FAN:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SEARCH:tl.constexpr):
    ntiles=tl.cdiv(H,BN)
    tile=tl.program_id(0)//ntiles;ntile=tl.program_id(0)%ntiles
    if tile<tl.load(TILES+E-1):
        lo=tl.full((),0,tl.int32);hi=tl.full((),E,tl.int32)
        for _ in range(SEARCH):
            mid=(lo+hi)//2
            bound=tl.load(TILES+mid,mid<E,other=2147483647)
            right=tile>=bound
            lo=tl.where(right,mid+1,lo);hi=tl.where(right,hi,mid)
        e=lo
        start=tl.load(OFF+e-1,e>0,other=0);end=tl.load(OFF+e)
        first_tile=tl.load(TILES+e-1,e>0,other=0)
        rows=start+(tile-first_tile)*BM+tl.arange(0,BM)
        valid=rows<end
        route=tl.load(ORDER+rows,valid,other=0)
        gate=tl.load(GATES+route,valid,other=0).to(tl.float32)
        hs=ntile*BN+tl.arange(0,BN);ks=tl.arange(0,BK)
        acc=tl.zeros((BM,BN),tl.float32)
        for block in range(tl.cdiv(D,BK)):
            kk=block*BK+ks
            x=tl.load(DY+(route//FAN)[:,None]*SY0+kk[None,:]*SY1,
                      valid[:,None]&(kk<D)[None,:],other=0)
            x=(x.to(tl.float32)*gate[:,None]).to(DY.dtype.element_ty)
            w=tl.load(W+e.to(tl.int64)*SW0+kk[:,None]*SW1+hs[None,:]*SW2,
                      (kk<D)[:,None]&(hs<H)[None,:],other=0)
            acc=tl.dot(x,w,acc,allow_tf32=True)
        tl.store(OUT+rows[:,None]*H+hs[None,:],acc,valid[:,None]&(hs<H)[None,:])


@torch.library.custom_op('allie_gated::dw',mutates_args={'out'})
def dw_op(dy:torch.Tensor,x:torch.Tensor,gates:torch.Tensor,order:torch.Tensor,
          offsets:torch.Tensor,out:torch.Tensor,fan:int,
          bm:int,bn:int,bk:int,warps:int,stages:int)->None:
    grid=(offsets.numel()*triton.cdiv(x.shape[-1],bk),triton.cdiv(dy.shape[-1],bn))
    _dw[grid](dy,x,gates,order,offsets,out,*dy.stride(),*x.stride(),*out.stride(),
              x.shape[-1],dy.shape[-1],fan,bm,bn,bk,num_warps=warps,num_stages=stages)


@torch.library.custom_op('allie_gated::dx',mutates_args={'out'})
def dx_op(dy:torch.Tensor,w:torch.Tensor,gates:torch.Tensor,order:torch.Tensor,
          offsets:torch.Tensor,tiles:torch.Tensor,out:torch.Tensor,fan:int,
          bm:int,bn:int,bk:int,warps:int,stages:int)->None:
    e=w.shape[0];h=w.shape[-1]
    grid=((triton.cdiv(order.numel(),bm)+e-1)*triton.cdiv(h,bn),)
    _dx[grid](dy,w,gates,order,offsets,tiles,out,*dy.stride(),*w.stride(),
              e,dy.shape[-1],h,fan,bm,bn,bk,e.bit_length()+1,
              num_warps=warps,num_stages=stages)


def weight_grad(dy,x,gates,order,offsets,config=None,out=None):
    gates=gates.contiguous()
    # Tile lookup uses routed row count; dy here is unexpanded.
    if config is None:
        config=WEIGHT.get((offsets.numel(),order.numel(),x.shape[-1],dy.shape[-1]),
                          (32,128,128,4,4))
    # The down weight's own [E,H,D] layout: AccumulateGrad keeps it instead of a strided copy.
    if out is None:out=dy.new_empty((offsets.numel(),x.shape[-1],dy.shape[-1]))
    dw_op(dy,x,gates,order,offsets,out,gates.shape[1],*config)
    return out


def input_grad(dy,w,gates,order,offsets,config=(128,256,64,8,3),out=None):
    gates=gates.contiguous()
    tiles=tile_prefix(offsets,config[0])
    if out is None:out=dy.new_empty((order.numel(),w.shape[-1]))
    assert out.is_contiguous()
    dx_op(dy,w,gates,order,offsets,tiles,out,gates.shape[1],*config)
    return out
