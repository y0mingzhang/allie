"""Default-off ScatterMoE tiles and direct expert-gradient accumulation.

Based on modded_smoe (ScatterMoE Apache-2.0). Original scatter stays untouched.
Tile choices are offline measurements, never runtime autotuning. Direct accumulation
rounds the per-microbatch dot to BF16 before adding to FP32, matching the existing
weight-gradient -> BF16 leaf gradient -> FP32 main_grad contract.
"""
from typing import Optional
import torch
import triton
import triton.language as tl
import modded_smoe as ref
from modded_smoe import group
from modded_smoe_tiles import SCATTER, WEIGHT


def scatter_config(x,w,order,k,xg,yg):
    if x.dtype == torch.bfloat16:
        selected = SCATTER.get((w.shape[0], order.numel(), x.shape[1], w.shape[-1], k, xg, yg))
        if selected is not None:
            return selected
    # Measured on L40S, 16K tokens, E192/k6/h455. Other shapes use the reference.
    if x.dtype==torch.bfloat16 and w.shape[0]==192 and order.numel()==98304:
        key=(x.shape[1],w.shape[-1],k,xg,yg)
        return {
            (2048,910,6,False,True):(128,128,64,4,4),
            (910,2048,1,True,False):(128,256,32,8,3),
            (2048,455,1,True,True):(64,128,64,4,4),
        }.get(key)
    return None


def legacy_weight_config(dy,x,experts):
    if dy.dtype==torch.bfloat16 and experts==192 and dy.shape[0]==98304:
        return {(2048,910):(64,128,128,8,3),(455,2048):(64,64,256,8,3)}.get((x.shape[1],dy.shape[1]))
    return None


def weight_config(dy,x,experts):
    if dy.dtype == torch.bfloat16:
        selected = WEIGHT.get((experts, dy.shape[0], x.shape[1], dy.shape[1]))
        if selected is not None:
            return selected
    return legacy_weight_config(dy,x,experts)


@torch.library.custom_op('allie_tuned::scatter',mutates_args={'out'})
def scatter_op(x:torch.Tensor,w:torch.Tensor,se:torch.Tensor,order:torch.Tensor,k:int,
               xg:bool,yg:bool,out:torch.Tensor)->None:
    bm,bn,bk,nw,ns=scatter_config(x,w,order,k,xg,yg)
    grid=(triton.cdiv(order.numel(),bm)*triton.cdiv(out.shape[1],bn),)
    ref._scatter2scatter.fn[grid](x,*x.stride(),w,*w.stride(),out,*out.stride(),None,0,0,
        order,se,FAN_OUT=k,M=x.shape[0],K=x.shape[1],N=out.shape[1],E=w.shape[0],
        BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,ACC_TYPE=tl.float32,allow_tf32=True,
        x_grouped=xg,y_grouped=yg,num_warps=nw,num_stages=ns)


def scatter2scatter(X,W,sorted_expert_idxs,sorted_scattered_idxs,k,b=None,x_grouped=False,y_grouped=False,out=None):
    cfg=scatter_config(X,W,sorted_scattered_idxs,k,x_grouped,y_grouped)
    if cfg is None or b is not None:
        return ref.scatter2scatter(X,W,sorted_expert_idxs,sorted_scattered_idxs,k,b,x_grouped,y_grouped,out)
    if out is None:out=X.new_empty((sorted_scattered_idxs.numel(),W.shape[-1]))
    scatter_op(X,W,sorted_expert_idxs,sorted_scattered_idxs,k,x_grouped,y_grouped,out)
    return out


@torch.library.custom_op('allie_tuned::wgrad',mutates_args={'out'})
def wgrad_op(dy:torch.Tensor,x:torch.Tensor,offsets:torch.Tensor,out:torch.Tensor)->None:
    bm,bn,bk,nw,ns=weight_config(dy,x,out.shape[0])
    grid=(out.shape[0]*triton.cdiv(x.shape[1],bk),triton.cdiv(dy.shape[1],bn))
    ref._groupXtY.fn[grid](dy,*dy.stride(),x,*x.stride(),out,*out.stride(),None,0,0,offsets,
        M=dy.shape[0],K=x.shape[1],N=dy.shape[1],BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,
        ACC_TYPE=tl.float32,allow_tf32=True,num_warps=nw,num_stages=ns)


def group_bwd_W(DY,X,expert_offsets,E,has_bias=False,out=None):
    if has_bias or weight_config(DY,X,E) is None:
        return ref.group_bwd_W(DY,X,expert_offsets,E,has_bias,out)
    if out is None:out=DY.new_empty((E,DY.shape[-1],X.shape[-1])).transpose(1,2)
    wgrad_op(DY,X,expert_offsets,out)
    return out,None


@triton.jit
def _direct_wgrad(DY,X,OFF,OUT,
                  SY0:tl.constexpr,SY1:tl.constexpr,SX0:tl.constexpr,SX1:tl.constexpr,
                  SO0:tl.constexpr,SO1:tl.constexpr,SO2:tl.constexpr,
                  K:tl.constexpr,N:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                  FRESH:tl.constexpr):
    pid0,pid1=tl.program_id(0),tl.program_id(1)
    pid0,pid1=tl.swizzle2d(pid0,pid1,tl.num_programs(0),tl.num_programs(1),4)
    e=pid0//tl.cdiv(K,BK)
    ki=(pid0%tl.cdiv(K,BK))*BK+tl.arange(0,BK)
    ni=pid1*BN+tl.arange(0,BN)
    start=tl.load(OFF+e-1,e>0,other=0).to(tl.int32)
    end=tl.load(OFF+e).to(tl.int32)
    rr=start+tl.arange(0,BM)
    acc=tl.zeros((BK,BN),tl.float32)
    for block in range(tl.cdiv(end-start,BM)):
        rows=rr+block*BM
        xx=tl.load(X+ki[:,None]*SX1+rows[None,:]*SX0,(ki<K)[:,None]&(rows<end)[None,:],other=0)
        yy=tl.load(DY+rows[:,None]*SY0+ni[None,:]*SY1,(rows<end)[:,None]&(ni<N)[None,:],other=0)
        acc+=tl.dot(xx,yy,out_dtype=tl.float32,allow_tf32=True)
    # Preserve the reference's BF16 gradient rounding even though no BF16 tensor is stored.
    val=acc.to(DY.dtype.element_ty).to(tl.float32)
    ptr=OUT+e.to(tl.int64)*SO0+ki[:,None].to(tl.int64)*SO1+ni[None,:].to(tl.int64)*SO2
    mask=(ki<K)[:,None]&(ni<N)[None,:]
    if not FRESH:val+=tl.load(ptr,mask,other=0)
    tl.store(ptr,val,mask)


@torch.library.custom_op('allie_tuned::direct_wgrad',mutates_args={'out'})
def direct_wgrad(dy:torch.Tensor,x:torch.Tensor,offsets:torch.Tensor,out:torch.Tensor,fresh:bool)->None:
    assert dy.dtype==torch.bfloat16 and out.dtype==torch.float32
    cfg=legacy_weight_config(dy,x,out.shape[0]) or (32,128,128,4,4)
    # FP32 read/add/write needs different tiles from materialized BF16 dW.
    # Selected offline, including the actual physical optimizer-buffer layout.
    if out.shape[0]==192 and dy.shape[0]==98304:
        cfg={
            (2048,910,1,2048):(64,256,64,8,3),
            (455,2048,2048,1):(32,128,256,8,3),
        }.get((x.shape[1],dy.shape[1],out.stride(1),out.stride(2)),cfg)
    bm,bn,bk,nw,ns=cfg
    grid=(out.shape[0]*triton.cdiv(x.shape[1],bk),triton.cdiv(dy.shape[1],bn))
    _direct_wgrad[grid](dy,x,offsets,out,*dy.stride(),*x.stride(),*out.stride(),
                       K=x.shape[1],N=dy.shape[1],BM=bm,BN=bn,BK=bk,FRESH=fresh,num_warps=nw,num_stages=ns)


@torch.library.custom_op('allie_tuned::direct_wgrad_flat',mutates_args={'flat'})
def direct_wgrad_flat(dy:torch.Tensor,x:torch.Tensor,offsets:torch.Tensor,flat:torch.Tensor,
                      row:int,transposed:bool,fresh:bool)->None:
    # The whole allocation crosses the compiled boundary. Make all views here:
    # compiled backward can double offsets when rebuilding aliased input views.
    out=flat[row] if row>=0 else flat
    if transposed:out=out.transpose(1,2)
    direct_wgrad(dy,x,offsets,out,fresh)


class TunedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        x: torch.Tensor, expert_weights: torch.Tensor, k: int,
        sorted_expert_idxs: torch.Tensor, sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_biases: Optional[torch.Tensor]=None,
        gates: Optional[torch.Tensor]=None,
        grouped_in: bool =False, grouped_out: bool=False,
        main_grad: Optional[torch.Tensor]=None, fresh:bool=False, main_grad_transposed:bool=False,
        main_grad_row:int=-1,
    ):
        output = scatter2scatter(
            X=x, W=expert_weights,
            b=expert_biases, k=k,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            x_grouped=grouped_in, y_grouped=grouped_out
        )
        if gates is not None:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
        else:
            output_expanded = None

        ctx.save_for_backward(
            x, expert_weights,
            expert_biases,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            output_expanded
        )
        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k
        ctx.fresh = fresh
        # Mutable optimizer output, not a saved activation: other layers may write
        # disjoint views of the same flat allocation before this backward runs.
        ctx.main_grad = main_grad
        ctx.main_grad_transposed = main_grad_transposed
        ctx.main_grad_row = main_grad_row
        return output
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x, expert_weights, expert_biases,
         sorted_expert_idxs,
         sorted_scattered_idxs,
         expert_offsets,
         gates, output_expanded) = ctx.saved_tensors
        main_grad = ctx.main_grad
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out
        # print("backward")

        if gates is not None:
            # calculate gates gradient
            # d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
            d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            grouped_grad_out = output_expanded.flatten(0, 1) # reuse expanded buffer later
        else:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            grouped_grad_out = group(grad_out, sorted_scattered_idxs,
                                                 fan_out=gate_fan, coeff=gates_flat,
                                                 out=grouped_grad_out)
        if grouped_in:
            grouped_x = x
            d_expanded_input = None
        else:
            grouped_x = group(x, sorted_scattered_idxs, fan_out=k)
            d_expanded_input = grouped_x

        if main_grad is None:
            d_weights, d_biases = group_bwd_W(
                DY=grouped_grad_out, X=grouped_x,
                expert_offsets=expert_offsets,
                E=expert_weights.size(0),
                has_bias=expert_biases is not None,
                out=torch.empty_like(expert_weights, dtype=grouped_grad_out.dtype)
            )
        else:
            assert expert_biases is None
            if ctx.main_grad_row>=0 or ctx.main_grad_transposed:
                direct_wgrad_flat(grouped_grad_out,grouped_x,expert_offsets,main_grad,
                                 ctx.main_grad_row,ctx.main_grad_transposed,ctx.fresh)
            else:
                direct_wgrad(grouped_grad_out,grouped_x,expert_offsets,main_grad,ctx.fresh)
            d_weights, d_biases = None, None


        d_expanded_input = scatter2scatter(
            X=grouped_grad_out, x_grouped=True,
            W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1,
            y_grouped=grouped_in,
            out=d_expanded_input # Reuse grouped_x buffer
        )

        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)
        # print("backward end.")
        return (
            # x, expert_weights,
            d_input, d_weights,
            # k, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            None, None, None, None, 
            # bias, gates
            d_biases, d_gates,
            # grouped_in, grouped_out,
            None, None,
            # main_grad and fresh: side-effect buffer, not differentiable inputs
            None, None, None, None
        )


def parallel_linear(inputs,expert_weights,k,sorted_expert_idxs,sorted_scattered_idxs,expert_offsets,
                    expert_biases=None,gates=None,grouped_in=False,grouped_out=False,
                    main_grad=None,fresh=False,main_grad_transposed=False,main_grad_row=-1):
    return TunedLinear.apply(inputs,expert_weights,k,sorted_expert_idxs,sorted_scattered_idxs,
                             expert_offsets,expert_biases,gates,grouped_in,grouped_out,main_grad,fresh,main_grad_transposed,main_grad_row)


def finish_direct_accum(param):
    """Leaf post-accumulate hook: the custom backward already wrote main_grad."""
    assert param.grad is None, "direct expert path must not also return a leaf gradient"
    param.fresh = False
