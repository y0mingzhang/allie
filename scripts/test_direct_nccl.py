"""Real compiled GPT + NorMuon/Adam/NCCL proof, including split and rank-state restore.

Run with torchrun --standalone --nproc_per_node=2. No CPU stand-ins or mocked
gradients: both paths train the same deterministic synthetic token rows.
"""
import os
import json
import torch
import torch.distributed as dist
from modded_medium import Config, TrainingManager, create_model, core, cpu_copy, make_context, move_losses
from modded_wsd import Schedule, install


def build(direct):
    torch.manual_seed(701)
    cfg=Config(width=128,head_dim=64,layers=8,max_tokens=1024,scheduled_steps=8,
               extension_steps=0,initial_batch_rows=8,bf16_weights=True,
               arch=dict(mlp='swiglu',moe=[16,2],moe_seq=.001,
                         moe_kernel='scatter-accum' if direct else 'scatter'))
    m=create_model(cfg,device=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    mgr=TrainingManager(m,cfg);mgr.split_step=5
    return m,mgr,torch.compile(m,fullgraph=True,dynamic=False)


def equal(a,b,path='state'):
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],f'{path}.{k}')
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}.{i}')
    else:assert a==b,(path,a,b)


def train(net,mgr,step,accum):
    mgr.advance_schedule(step);core.grad_accum_steps=accum
    total=0.
    for micro in range(accum):
        if micro==accum-1:mgr.activate_hooks(step)
        g=torch.Generator(device='cuda').manual_seed(1000*step+10*micro+dist.get_rank())
        row=torch.randint(378,2346,(1,1025),device='cuda',generator=g)
        x,y=row[:,:-1],row[:,1:]
        ctx=make_context(x,mgr.ws_short*128,mgr.ws_long*128)
        z=net(x.flatten(),y.flatten(),ctx,mgr.get_forward_args())
        loss,primary,_=move_losses(z,x,y,ctx,mgr.mtp_weights)
        (loss*(dist.get_world_size()/8)).backward()
        total+=float(primary.detach())
    mgr.step_optimizers(step)
    return total


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True
    install(core,Schedule(warmup_steps=2,mtp_steps=0,split_step=5,batch_rows=8),-1,8)
    ref,rm,rf=build(False);new,nm,nf=build(True)
    for step in range(8):
        accum=1 if step%2==0 else 3
        r=train(rf,rm,step,accum);n=train(nf,nm,step,accum)
        assert r==n,('loss',step,r,n)
        equal(dict(ref.named_parameters()),dict(new.named_parameters()),'parameters')
        for i,(ro,no) in enumerate(zip(rm.optimizers,nm.optimizers)):
            equal(ro.state_dict(),no.state_dict(),f'optimizer{i}')
        if step==3:
            weights=cpu_copy(new.state_dict());state=nm.rank_state_dict()
            del nf,new,nm
            new,nm,nf=build(True)
            new.load_state_dict(weights);nm.load_rank_state_dict(state)
        if dist.get_rank()==0:print(json.dumps(dict(step=step+1,microbatches=accum,loss=r,bit_equal=True)),flush=True)
    dist.barrier();dist.destroy_process_group()
    print('PASS NCCL compiled GPT, real gradients, optimizer states, split and restore',flush=True)


if __name__=='__main__':main()
