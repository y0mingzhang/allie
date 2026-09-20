"""CPU distributed optimizer/resume contract for direct expert gradient buffers.

Actual Triton gradients are tested by test_smoe_accum. Here inject identical,
BF16-representable per-rank gradients to isolate master/flat-buffer behavior.
"""
import os
os.environ['TORCH_COMPILE_DISABLE']='1'
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import test_modded_zero as base
from modded_medium import Config,TrainingManager,create_model,core,cpu_copy
from modded_wsd import Schedule,install


class Inject(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,g):
        ctx.p=p;ctx.save_for_backward(g)
        return (p*g).sum()
    @staticmethod
    def backward(ctx,scale):
        p=ctx.p;g=ctx.saved_tensors[0]*scale
        if p.fresh:p.main_grad.copy_(g)
        else:p.main_grad.add_(g)
        return None,None


def build(direct):
    torch.manual_seed(701)
    arch=dict(mlp='swiglu',moe=[8,2],moe_kernel='scatter-accum' if direct else 'scatter')
    cfg=Config(width=64,head_dim=16,layers=10,max_tokens=1024,scheduled_steps=8,
               extension_steps=0,initial_batch_rows=8,bf16_weights=True,arch=arch)
    m=create_model(cfg,device='cpu');mgr=TrainingManager(m,cfg);mgr.split_step=5
    return m,mgr


def train(m,mgr,steps,accum,direct):
    for step in steps:
        mgr.advance_schedule(step)
        for micro in range(accum):
            if micro==accum-1:mgr.activate_hooks(step)
            g=torch.Generator().manual_seed(1000*step+10*micro+dist.get_rank())
            loss=0
            for p in m.parameters():
                x=(.01*torch.randn(p.shape,generator=g)).bfloat16().to(p.dtype)
                loss += Inject.apply(p,x) if direct and p.label in ('moe','moe_up') else (p*x).sum()
            loss.backward()
        mgr.step_optimizers(step)


def equal(a,am,b,bm):
    for (na,p),(nb,q) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb and torch.equal(p,q),na
    for g,h in zip(am.muon_opt.param_groups,bm.muon_opt.param_groups):
        for key in ('master','momentum_buffer','second_momentum_buffer'):
            if key in g:assert torch.equal(g[key],h[key]),key


def worker(rank,world,port):
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world)
    torch.set_num_threads(1);base.patch()
    install(core,Schedule(warmup_steps=2,mtp_steps=0,split_step=5,batch_rows=8),-1,8)
    for accum in (1,3):
        ref,rm=build(False);new,nm=build(True)
        for step in range(8):
            train(ref,rm,[step],accum,False);train(new,nm,[step],accum,True)
            equal(ref,rm,new,nm)
            if step==3:
                weights=cpu_copy(new.state_dict());opt=nm.rank_state_dict()
                new,nm=build(True);new.load_state_dict(weights);nm.load_rank_state_dict(opt)
    if rank==0:print(f'PASS world{world}: direct buffer optimizer/master/moments bit-equal,1/3 microbatches,split5,resume4',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':
    for world in (1,2,4):
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        mp.spawn(worker,args=(world,port),nprocs=world,join=True)
