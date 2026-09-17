"""Train MoveLM directly on the established chess-v2 packed 1025-token rows."""
import argparse
from dataclasses import asdict
import json
import math
import os
import random
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(os.environ.get('ALLIE_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
SOURCE = Path(__file__).resolve().parent
from lm_model import Config,MoveLM,optimizers


from lm_data import Packed
from lm_checkpoint import atomic_save, rng_state, restore_rng
import signal
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def rating_at_targets(rows):
    n,l=rows.shape
    pos=np.arange(l)[None,:]
    start=np.maximum.accumulate(np.where(rows==2348,pos,0),axis=1)
    ar=np.arange(n)[:,None]
    white=sum(rows[ar,np.minimum(start+k,l-1)]*p for k,p in zip(range(3,7),[1000,100,10,1]))
    black=sum(rows[ar,np.minimum(start+k,l-1)]*p for k,p in zip(range(7,11),[1000,100,10,1]))
    return np.where((pos-start-11)%2==0,white,black)[:,1:]


@torch.inference_mode()
def evaluate(model,rows,batch=8):
    model.eval();sums=np.zeros(8,np.float64)
    for lo in range(0,len(rows),batch):
        data=rows[lo:lo+batch]
        x=torch.as_tensor(data[:,:-1],device='cuda');y=torch.as_tensor(data[:,1:],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
        mask=(y>=378)&(y<=2345)
        all_loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='none').view_as(y)
        move_loss=torch.logsumexp(logits[:,:,378:2346].float(),-1)-logits.float().gather(-1,y.unsqueeze(-1)).squeeze(-1)
        elo=torch.as_tensor(rating_at_targets(data),device='cuda')
        sums[0]+=all_loss.sum().item();sums[1]+=y.numel()
        sums[2]+=move_loss[mask].sum().item();sums[3]+=mask.sum().item()
        expert=mask&(elo>=2400);elite=mask&(elo>=2600)
        sums[4]+=move_loss[expert].sum().item();sums[5]+=expert.sum().item()
        sums[6]+=move_loss[elite].sum().item();sums[7]+=elite.sum().item()
    model.train()
    return dict(all_token_ce=sums[0]/sums[1],move_ce=sums[2]/sums[3],expert2400_ce=sums[4]/max(1,sums[5]),
                expert2600_ce=sums[6]/max(1,sums[7]),move_count=int(sums[3]),expert2400_count=int(sums[5]),expert2600_count=int(sums[7]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True);p.add_argument('--layers',type=int,default=12)
    p.add_argument('--width',type=int,default=384);p.add_argument('--heads',type=int,default=6);p.add_argument('--ff',type=int,default=1024)
    p.add_argument('--lr',type=float,default=.003);p.add_argument('--wd',type=float,default=.1);p.add_argument('--dropout',type=float,default=0.)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--accum',type=int,default=1)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--max-seconds',type=int,default=1500)
    p.add_argument('--eval-every',type=int,default=500);p.add_argument('--val-rows',type=int,default=256)
    p.add_argument('--checkpoint-every',type=int,default=500);p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--train-shards',type=int,default=0)
    p.add_argument('--data',default='/scratch/yimingz3/allie/lichess_tokens_v2')
    p.add_argument('--no-compile',action='store_true');p.add_argument('--plain',action='store_true')
    p.add_argument('--bf16-residual',action='store_true')
    p.add_argument('--mlp',choices=['swiglu','relu2'],default='swiglu')
    p.add_argument('--game-attention',action='store_true')
    p.add_argument('--deterministic',action='store_true',help='Strict recovery checks; may reduce throughput')
    p.add_argument('--resume');p.add_argument('--init-from');p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--momentum-warmup',type=int,default=300)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(4)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local_rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local_rank)
    if world>1:dist.init_process_group('nccl')
    torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=True
    torch.use_deterministic_algorithms(a.deterministic)
    out=ROOT/'results'/'pretrain'/a.name
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    cfg=Config(layers=a.layers,width=a.width,heads=a.heads,ff=a.ff,dropout=a.dropout,extras=not a.plain,bf16_residual=a.bf16_residual,mlp=a.mlp,game_attention=a.game_attention)
    model=MoveLM(cfg).cuda();opts=optimizers(model,lr=a.lr,weight_decay=a.wd)
    train=Packed(a.data,'train',a.seed,a.train_shards);val=Packed(a.data,'val')
    vidx=np.random.default_rng(20260910).choice(int(val.ends[-1]),min(a.val_rows,int(val.ends[-1])),replace=False)
    vrows=val.rows(vidx) if rank==0 else None
    initial_step=0;best=float('inf');elapsed_prior=0.;checkpoint=None
    if a.resume:
        checkpoint=torch.load(a.resume,map_location='cpu',weights_only=False)
        if checkpoint.get('format')!=2:raise ValueError('Resume requires a full format-2 checkpoint; use --init-from for weights only')
        # Old checkpoints may omit later optional fields whose defaults preserve
        # the exact architecture (e.g. game_attention=False).
        assert asdict(Config(**checkpoint['config']))==asdict(cfg)
        for key in ['batch','accum','steps','seed','lr','wd','warmup','momentum_warmup','deterministic','no_compile']:
            saved=checkpoint['args'].get(key,300 if key=='momentum_warmup' else None)
            assert saved==getattr(a,key),f'Resume changes {key}'
        assert checkpoint['world_size']==world,'Exact continuation requires the same world size'
        model.load_state_dict(checkpoint['model'])
        for opt,state in zip(opts,checkpoint['optimizers']):opt.load_state_dict(state)
        train.load_state_dict(checkpoint['data'])
        initial_step=checkpoint['step'];best=checkpoint['best_move_ce'];elapsed_prior=checkpoint['elapsed_seconds']
    elif a.init_from:
        checkpoint=torch.load(a.init_from,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model']);checkpoint=None
    elif (out/'last.pt').exists():raise ValueError('Run already exists: explicitly resume or choose a new name')
    # Count useful causal matmul work, not padded kernel work. Retain rank-local
    # context sums so exact resume does not approximate packed-game sparsity.
    dense_forward_flops=model.flop_estimate(0)['matmul_flops_per_cached_move']
    attention_coefficient=4*cfg.layers*cfg.width
    flops_start_rows=0;context_sum_local=0
    if a.resume:
        if 'attention_context_sums' in checkpoint:
            context_sum_local=checkpoint['attention_context_sums'][rank]
            flops_start_rows=checkpoint['flops_start_rows']
        elif cfg.game_attention:
            # Historical sparse checkpoints lack this counter; explicitly report
            # work since this resume rather than invent their prior context sum.
            flops_start_rows=train.seen
        else:context_sum_local=(train.seen//world)*(1024*1025//2)
    def useful_flops(context_sum):
        tokens=(train.seen-flops_start_rows)*1024
        return 3*(dense_forward_flops*tokens+attention_coefficient*context_sum)
    base=model if a.no_compile else torch.compile(model,dynamic=False)
    net=DDP(base,device_ids=[local_rank],broadcast_buffers=False) if world>1 else base
    torch.manual_seed(a.seed+rank)
    random.seed(a.seed+rank);np.random.seed(a.seed+rank)
    if checkpoint:restore_rng(checkpoint['rng'][rank])
    metadata=dict(args=vars(a),config=asdict(cfg),parameters=model.parameters_count(),flops=model.flop_estimate(),
        world_size=world,train_shards=[str(p) for p in train.paths],train_rows=int(train.ends[-1]),val_indices=vidx.tolist(),
        dataset=json.loads((ROOT/'results'/'original-data.json').read_text()),job_id=os.environ.get('SLURM_JOB_ID'))
    metadata['dataset']['local_path']=str(Path(a.data).resolve())
    metadata['compute_accounting']='Useful forward/backward matmul FLOPs; causal context pairs counted from actual training rows. Excludes optimizer, elementwise and padded kernel work.'
    metadata['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.glob('lm_*.py')}
    if rank==0:
        snapshot=out/('resume-source' if a.resume else 'source');snapshot.mkdir(exist_ok=True)
        for f in SOURCE.glob('lm_*.py'):(snapshot/f.name).write_bytes(f.read_bytes())
        (out/('resume-config.json' if a.resume else 'config.json')).write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata|{'train_shards':len(train.paths),'val_indices':len(vidx)}),flush=True)
    termination=[False]
    def request_stop(*_):termination[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
    def save(step,metrics=None):
        checkpoint_start=time.monotonic()
        states=[None]*world if rank==0 else None
        local=dict(rng=rng_state(),context_sum=context_sum_local)
        if world>1:dist.gather_object(local,states,dst=0)
        else:states=[local]
        if rank==0:
            state=dict(format=2,model=model.state_dict(),optimizers=[o.state_dict() for o in opts],config=asdict(cfg),
                args=vars(a),step=step,best_move_ce=best,elapsed_seconds=elapsed_prior+time.monotonic()-start,
                data=train.state_dict(),rng=[s['rng'] for s in states],world_size=world,metrics=metrics,source_sha256=metadata['source_sha256'],
                attention_context_sums=[s['context_sum'] for s in states],flops_start_rows=flops_start_rows,
                useful_training_flops=useful_flops(sum(s['context_sum'] for s in states)))
            atomic_save(state,out/'last.pt')
        if world>1:dist.barrier()
        if rank==0:
            record=dict(event='checkpoint',step=step,seconds=time.monotonic()-checkpoint_start,
                        bytes=(out/'last.pt').stat().st_size)
            with (out/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
    def append(file,record):
        if rank==0:
            with (out/file).open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
    interval_start=time.monotonic();interval_tokens=0;interval_steps=0;loss_sum=torch.zeros((),device='cuda')
    step=initial_step;stop_reason='steps'
    for step in range(initial_step+1,a.steps+1):
        frac=step/a.steps
        lr_mult=min(1.,step/max(1,a.warmup))*(.1+.9*.5*(1+math.cos(math.pi*max(0.,(frac-.3)/.7))))
        for oi,opt in enumerate(opts):
            for group in opt.param_groups:
                group['lr']=a.lr*lr_mult/(3 if oi else 1)
                if oi==0:group['momentum']=.85+.1*min(step/max(1,a.momentum_warmup),1.)
            opt.zero_grad(set_to_none=True)
        for micro in range(a.accum):
            rows=train.batch(a.batch,rank,world)
            if cfg.game_attention:
                positions=np.arange(1024)[None,:]
                starts=np.maximum.accumulate(np.where(rows[:,:-1]==2348,positions,0),axis=1)
                context_sum_local+=int((positions-starts+1).sum())
            else:context_sum_local+=len(rows)*(1024*1025//2)
            data=torch.from_numpy(rows).pin_memory().to('cuda',non_blocking=True)
            ctx=net.no_sync() if world>1 and micro<a.accum-1 else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(data[:,:-1],targets=data[:,1:])/a.accum
                loss.backward()
            loss_sum+=loss.detach();interval_tokens+=a.batch*world*1024
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for opt in opts:opt.step()
        interval_steps+=1
        if step%25==0 or step==a.steps:
            torch.cuda.synchronize();dt=time.monotonic()-interval_start
            if not torch.isfinite(loss_sum):raise RuntimeError('Nonfinite training loss')
            contexts=torch.tensor(context_sum_local,device='cuda',dtype=torch.int64)
            if world>1:dist.all_reduce(contexts,op=dist.ReduceOp.SUM)
            total_context=int(contexts.item())
            rec=dict(step=step,train_ce=loss_sum.item()/interval_steps,lr=a.lr*lr_mult,
                seconds=elapsed_prior+time.monotonic()-start,tokens=train.seen*1024,tokens_per_second=interval_tokens/dt,
                useful_training_flops=useful_flops(total_context),flops_start_rows=flops_start_rows,
                grad_norm=float(norm),epochs_started=train.epochs,max_memory_gb=torch.cuda.max_memory_allocated()/1e9)
            append('train.jsonl',rec);loss_sum.zero_();interval_steps=0;interval_tokens=0;interval_start=time.monotonic()
        stop=termination[0] or time.monotonic()-start>a.max_seconds or (a.stop_after and step>=a.stop_after)
        signal_stop=termination[0]
        if world>1:
            # Every rank must agree before any rank enters checkpoint collectives.
            # Poll at a shared interval to avoid a host/device sync each step.
            if step%5==0 or step%a.eval_every==0 or step%a.checkpoint_every==0 or step==a.steps or (a.stop_after and step==a.stop_after):
                # Preserve urgency when only another rank received the signal.
                # Signal stops must checkpoint before any lengthy validation.
                flag=torch.tensor(2 if signal_stop else int(stop),device='cuda')
                dist.all_reduce(flag,op=dist.ReduceOp.MAX)
                code=int(flag.item());stop=code>0;signal_stop=code==2
            else:stop=False;signal_stop=False
        metrics=None
        if not signal_stop and (step%a.eval_every==0 or step==a.steps or stop):
            contexts=torch.tensor(context_sum_local,device='cuda',dtype=torch.int64)
            if world>1:dist.all_reduce(contexts,op=dist.ReduceOp.SUM)
            validation_flops=useful_flops(int(contexts.item()))
            if rank==0:
                metrics=evaluate(model,vrows)
                append('validation.jsonl',dict(step=step,seconds=elapsed_prior+time.monotonic()-start,
                    useful_training_flops=validation_flops,flops_start_rows=flops_start_rows,**metrics))
                if metrics['move_ce']<best:
                    best=metrics['move_ce']
                    atomic_save(dict(model=model.state_dict(),config=asdict(cfg),step=step,metrics=metrics),out/'best.pt')
            if world>1:dist.barrier()
            # Keep validation/checkpoint time out of steady-state throughput windows.
            loss_sum.zero_();interval_steps=0;interval_tokens=0;interval_start=time.monotonic()
        if step%a.checkpoint_every==0 or step==a.steps or stop:
            save(step,metrics)
            # Checkpoint time must not contaminate the next throughput window.
            loss_sum.zero_();interval_steps=0;interval_tokens=0;interval_start=time.monotonic()
        if stop:
            stop_reason='signal' if signal_stop else 'stop_after' if a.stop_after and step>=a.stop_after else 'wall_clock_cap'
            break
    if rank==0:
        final=dict(step=step,seconds=elapsed_prior+time.monotonic()-start,best_move_ce=best,tokens_processed=train.seen*1024,
            unique_rows_available=int(train.ends[-1]),epochs_started=train.epochs,stop_reason=stop_reason)
        (out/'done.json').write_text(json.dumps(final,indent=2));print('DONE',json.dumps(final),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
