"""Direct gradient accumulation: CPU algebra + CUDA fullgraph and microbatch tests."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
import modded_smoe_tuned as tuned
from modded_moe import MoE
from modded_medium_core import accumulate_fp32
from modded_smoe_tuned import finish_direct_accum


def cpu_proof():
    def linear(X,W,sorted_expert_idxs,sorted_scattered_idxs,k,b=None,x_grouped=False,y_grouped=False,out=None):
        xx=X if x_grouped else X[sorted_scattered_idxs//k]
        yy=torch.bmm(xx[:,None],W[sorted_expert_idxs]).squeeze(1)
        if not y_grouped:yy=torch.zeros_like(yy).index_copy_(0,sorted_scattered_idxs,yy)
        if out is not None:out.copy_(yy);return out
        return yy
    def group(A,order,coeff=None,fan_out=1,out=None):
        yy=A[order//fan_out]
        if coeff is not None:yy=yy*coeff[order,None]
        if out is not None:out.copy_(yy);return out
        return yy
    def wgrad(DY,X,expert_offsets,E,has_bias=False):
        starts=torch.cat([expert_offsets.new_zeros(1),expert_offsets[:-1]])
        return torch.stack([X[lo:hi].T@DY[lo:hi] for lo,hi in zip(starts,expert_offsets)]),None
    def direct(dy,x,offsets,out,fresh):
        dw=wgrad(dy,x,offsets,out.shape[0])[0]
        out.copy_(dw) if fresh else out.add_(dw)
    saved={n:getattr(tuned,n) for n in ['scatter2scatter','group','group_bwd_W','direct_wgrad']}
    tuned.scatter2scatter=linear;tuned.group=group;tuned.group_bwd_W=wgrad;tuned.direct_wgrad=direct
    try:
        for k in (1,2,4):
            torch.manual_seed(k);t,e,d,h=7,9,5,11
            ix=torch.stack([torch.randperm(e-2)[:k] for _ in range(t)])
            flat=ix.flatten();order=flat.argsort(stable=True);se=flat[order];offs=flat.bincount(minlength=e).cumsum(0)
            for down in (False,True):
                for fresh in (False,True):
                    x=torch.randn((t*k if down else t),d,dtype=torch.float64,requires_grad=True)
                    w=torch.randn(e,d,h,dtype=torch.float64,requires_grad=True)
                    gates=torch.randn(t,k,dtype=torch.float64,requires_grad=True) if down else None
                    init=torch.randn_like(w);main=init.clone()
                    actual=tuned.parallel_linear(x,w,1 if down else k,se,order,offs,gates=gates,
                                                 grouped_in=down,grouped_out=not down,main_grad=main,fresh=fresh)
                    expected=linear(x,w,se,order,1 if down else k,x_grouped=down,y_grouped=not down)
                    if down:expected=(gates[:,:,None]*expected.reshape(t,k,h)).sum(1)
                    dy=torch.randn_like(actual);ins=(x,w,gates) if down else (x,w)
                    eg=torch.autograd.grad(expected,ins,dy)
                    ag=torch.autograd.grad(actual,ins,dy,allow_unused=True)
                    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                    torch.testing.assert_close(ag[0],eg[0],atol=1e-12,rtol=1e-12)
                    if down:torch.testing.assert_close(ag[2],eg[2],atol=1e-12,rtol=1e-12)
                    assert ag[1] is None
                    torch.testing.assert_close(main,eg[1] if fresh else init+eg[1],atol=1e-12,rtol=1e-12)
        # Two backward nodes write different slices of one optimizer allocation.
        # Saving either mutable slice as an activation would invalidate the other.
        t,e,d=5,3,7
        order=torch.arange(t);se=torch.tensor([0,0,1,1,2]);offs=torch.tensor([2,4,5])
        x=torch.randn(t,d,dtype=torch.float64,requires_grad=True)
        w1=torch.randn(e,d,d,dtype=torch.float64,requires_grad=True)
        w2=torch.randn(e,d,d,dtype=torch.float64,requires_grad=True)
        flatbuf=torch.empty(2,e,d,d,dtype=torch.float64)
        y=tuned.parallel_linear(x,w1,1,se,order,offs,grouped_out=True,main_grad=flatbuf[0],fresh=True)
        z=tuned.parallel_linear(y,w2,1,se,order,offs,grouped_in=True,grouped_out=True,main_grad=flatbuf[1],fresh=True)
        yr=linear(x,w1,se,order,1,y_grouped=True)
        zr=linear(yr,w2,se,order,1,x_grouped=True,y_grouped=True)
        expected=torch.autograd.grad(zr.sum(),(x,w1,w2))
        actual=torch.autograd.grad(z.sum(),x)[0]
        torch.testing.assert_close(actual,expected[0],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(flatbuf[0],expected[1],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(flatbuf[1],expected[2],atol=1e-12,rtol=1e-12)
        print('PASS shared main_grad storage across multiple backward nodes',flush=True)
        print('PASS CPU FP64: forward, input/gate grads, accumulation, overwrite, unselected experts',flush=True)
    finally:
        for n,v in saved.items():setattr(tuned,n,v)


def make(kernel,candidate=False):
    torch.manual_seed(701)
    d,e,k,h=(2048,192,6,455) if candidate else (128,9,2,17)
    m=MoE(d,e,k,h,d,kind='swiglu',kernel=kernel,init=.006,seq=0).cuda()
    with torch.no_grad():m.down.normal_(0,.02);m.shared_down.normal_(0,.02)
    for n,p in m.named_parameters():
        if n!='router':
            p.data=p.data.bfloat16();p.main_grad=torch.empty_like(p,dtype=torch.float32);p.fresh=True
            p.acc=(p.main_grad,-1)
            p.register_post_accumulate_grad_hook(finish_direct_accum if kernel=="scatter-accum" and n in ("up","down") else accumulate_fp32)
    return m


def gpu_proof(candidate,out):
    results={};reference=None
    for kernel in ('scatter','scatter-tuned','scatter-accum'):
        m=make(kernel,candidate);f=torch.compile(m,fullgraph=True,dynamic=False)
        xs=[];ys=[]
        for micro in range(3):
            torch.manual_seed(80+micro)
            x=torch.randn(16384 if candidate else 1024,m.up.shape[-1],device='cuda',dtype=torch.bfloat16,requires_grad=True)
            y=f(x);y.float().square().sum().backward()
            xs.append(x.grad.detach().cpu());ys.append(y.detach().cpu())
        gg={n:(p.main_grad if hasattr(p,'main_grad') else p.grad).detach().cpu().clone() for n,p in m.named_parameters()}
        pack=dict(x=xs,y=ys,grad=gg)
        if reference is None:reference=pack
        errors={}
        for key,aa,bb in [(f'x{i}',v,reference['x'][i]) for i,v in enumerate(xs)]+[(f'y{i}',v,reference['y'][i]) for i,v in enumerate(ys)]+[(n,v,reference['grad'][n]) for n,v in gg.items()]:
            diff=aa.float()-bb.float();err=float(diff.norm()/bb.float().norm().clamp_min(1e-10))
            errors[key]=dict(relative=err,max_abs=float(diff.abs().max()))
            assert torch.equal(aa,bb),(kernel,key,errors[key])
        assert all(e['max_abs']==0 for e in errors.values()),errors
        results[kernel]=errors;print(kernel,json.dumps(errors),flush=True)
        if kernel=='scatter-accum':assert m.up.grad is None and m.down.grad is None
        del f,m,pack,gg,x,y;__import__('gc').collect();torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(results,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--candidate',action='store_true');p.add_argument('--out');a=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=True
    if a.gpu:gpu_proof(a.candidate,a.out)
    else:cpu_proof()

if __name__=='__main__':main()
