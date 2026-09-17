"""Distributed native-Qwen driver draft; requires actual proof before quality use.

Preserves global-batch objective. Checkpoints are published only after every rank
has saved optimizer, RNG and original data state. No world-size migration.
"""
import argparse,hashlib,json,math,os,random,signal,time,uuid
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scaled_native_qwen import Shape,build
from lm_data import Packed
from lm_checkpoint import atomic_save,rng_state,restore_rng
SOURCE=Path(__file__).resolve().parent
FILES=('train_native_distributed.py','scaled_native_qwen.py','historical_qwen_runtime.py','native_fp32_vector_adam.py','lm_data.py','lm_checkpoint.py')

def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu(v) for v in value)
 return value

def lr_at(step,steps,warmup,peak,shape):
 if step<warmup:return peak*(step+1)/warmup
 fraction=(step-warmup+1)/(steps-warmup)
 decay=1-fraction if shape=='linear' else .5*(1+math.cos(math.pi*fraction))
 return peak*(.05+.95*decay)

def write_json(path,value):
 temporary=path.with_name(path.name+'.partial')
 temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--out',required=True);p.add_argument('--steps',type=int,required=True)
 p.add_argument('--warmup',type=int,default=32);p.add_argument('--lr',type=float,default=.01)
 p.add_argument('--schedule',choices=('linear','cosine'),default='linear')
 p.add_argument('--global-rows',type=int,default=512);p.add_argument('--micro',type=int,default=8)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--checkpoint-every',type=int,default=256)
 p.add_argument('--max-seconds',type=int,default=1080);p.add_argument('--resume')
 p.add_argument('--data',default='/scratch/yimingz3/allie/lichess_tokens_v2')
 p.add_argument('--shape-json',default='{}');p.add_argument('--capture-first-update',action='store_true')
 a=p.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
 assert world in (1,2,4,8) and 0<a.warmup<a.steps and a.lr>0
 assert a.global_rows%(world*a.micro)==0 and a.checkpoint_every>0
 torch.set_num_threads(4);torch.cuda.set_device(local)
 dist.init_process_group('nccl',device_id=torch.device('cuda',local));torch.use_deterministic_algorithms(True)
 started=time.monotonic();out=Path(a.out)
 if rank==0:out.mkdir(parents=True,exist_ok=bool(a.resume))
 dist.barrier()
 model,net,optimizer,metadata=build(Shape(**json.loads(a.shape_json)),seed=a.seed,lr=a.lr)
 assert all(torch.isfinite(v).all() for v in model.parameters())
 assert all(torch.all(v==1) for n,v in model.named_parameters() if '.q_norm.' in n or '.k_norm.' in n)
 assert all(layer.cos.is_cuda and layer.sin.is_cuda for layer in model.decoder_layers)
 source_sha={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in FILES}
 recipe={k:v for k,v in vars(a).items() if k not in ('out','resume','max_seconds','capture_first_update')}
 metadata.update(source_sha256=source_sha,recipe=recipe,gpu=torch.cuda.get_device_name(),world_size=world,objective='2350-way all-token CE, native BF16 reduction',context='Full causal packed row',gradient_normalization='Local accumulated mean, then native FP32 DP buckets average over ranks')
 train=Packed(a.data,'train',a.seed);assert len(train.paths)==100 and int(train.ends[-1])==54368123
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 first=0;history=[]
 if a.resume:
  pointer=torch.load(a.resume,map_location='cpu',weights_only=False);assert pointer['format']=='native-distributed-pointer-v1'
  directory=Path(a.resume).parent/pointer['directory'];shared=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
  local_state=torch.load(directory/f'rank{rank}.pt',map_location='cpu',weights_only=False)
  assert shared['metadata']==metadata and local_state['rank']==rank and local_state['world_size']==world
  assert pointer['step']==shared['step']==local_state['step']
  model.load_state_dict(shared['model']);optimizer.load_state_dict(local_state['optimizer'])
  train.load_state_dict(local_state['data']);restore_rng(local_state['rng'])
  first=shared['step'];history=local_state['history'];del shared,local_state
  assert 0<=first<a.steps, 'Completed run cannot be resumed as a new allocation'
 if rank==0:write_json(out/'metadata.json',metadata)
 write_json(out/f'worker-rank{rank}.json',dict(pid=os.getpid(),rank=rank,resumed_from=first))
 stopping=[False]
 def stop(*_):stopping[0]=True
 signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
 def checkpoint(step):
  tick=time.monotonic();name=[f'step-{step:08d}-{uuid.uuid4().hex[:10]}' if rank==0 else None]
  dist.broadcast_object_list(name,src=0);directory=out/'checkpoints'/name[0]
  if rank==0:directory.mkdir(parents=True)
  dist.barrier()
  atomic_save(dict(rank=rank,world_size=world,step=step,optimizer=cpu(optimizer.state_dict()),data=train.state_dict(),rng=rng_state(),history=history),directory/f'rank{rank}.pt')
  if rank==0:atomic_save(dict(format='native-distributed-model-v1',metadata=metadata,step=step,model=cpu(model.state_dict()),tokens=train.seen*1024),directory/'model.pt')
  dist.barrier()
  if rank==0:
   assert all((directory/f'rank{r}.pt').is_file() for r in range(world))
   atomic_save(dict(format='native-distributed-pointer-v1',step=step,directory=str(directory.relative_to(out))),out/'last.pt')
   # Retain the last two fully published states; no pre-existing run is edited.
   import shutil
   completed=[x for x in (out/'checkpoints').iterdir() if (x/'model.pt').is_file() and all((x/f'rank{r}.pt').is_file() for r in range(world))]
   previous=sorted((x for x in completed if x!=directory),key=lambda x:x.name,reverse=True)
   for old in previous[1:]:shutil.rmtree(old)
  dist.barrier();return time.monotonic()-tick
 reason='steps';accum=a.global_rows//(world*a.micro)
 if a.capture_first_update and first==0:atomic_save(cpu(model.state_dict()),out/f'initial-rank{rank}.pt')
 for step in range(first,a.steps):
  tick=time.monotonic();lr=lr_at(step,a.steps,a.warmup,a.lr,a.schedule)
  for group in optimizer.param_groups:group['lr']=lr
  global_rows=train.batch(a.global_rows)
  rows=global_rows[rank*(a.global_rows//world):(rank+1)*(a.global_rows//world)]
  probe=[random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device='cuda'))]
  optimizer.zero_grad();loss_sum=torch.zeros((),device='cuda',dtype=torch.float64)
  for micro in range(accum):
   net.require_backward_grad_sync=micro==accum-1
   ids=torch.as_tensor(rows[micro*a.micro:(micro+1)*a.micro],device='cuda')
   logits=net(ids[:,:-1]);loss=F.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten())/accum
   loss.backward();loss_sum+=loss.detach().double()
  if a.capture_first_update and step==0:
   atomic_save(dict(gradients={n:cpu(v.grad) for n,v in model.named_parameters()},global_rows_sha256=hashlib.sha256(global_rows.tobytes()).hexdigest(),local_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),local_loss=float(loss_sum)),out/f'first-gradients-rank{rank}.pt')
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.5)
  dist.all_reduce(loss_sum);loss_sum/=world
  assert torch.isfinite(norm) and torch.isfinite(loss_sum)
  optimizer.step();net.reset();torch.cuda.synchronize()
  row=dict(step=step+1,loss=float(loss_sum),grad_norm=float(norm),lr=lr,rng_probe=probe,rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),global_rows_sha256=hashlib.sha256(global_rows.tobytes()).hexdigest())
  history.append(row)
  with (out/f'timing-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row|dict(seconds=time.monotonic()-tick,peak_bytes=torch.cuda.max_memory_allocated()))+'\n')
  if rank==0:write_json(out/'progress.json',dict(step=step+1))
  flag=torch.tensor([int(stopping[0]),int(time.monotonic()-started>a.max_seconds)],device='cuda')
  dist.all_reduce(flag,op=dist.ReduceOp.MAX)
  flags=flag.tolist();reason='signal' if flags[0] else 'wall_clock_cap' if flags[1] else 'steps'
  if (step+1)%a.checkpoint_every==0 or step+1==a.steps or reason!='steps':
   duration=checkpoint(step+1)
   if rank==0:
    with (out/'checkpoint-times.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,seconds=duration))+'\n')
  if reason!='steps':break
 if reason=='steps' and rank==0:
  model.eval();val=Packed(a.data,'val');assert int(val.ends[-1])==5371
  with torch.inference_mode():
   ids=torch.as_tensor(val.rows(np.arange(8)),device='cuda');scores=model(ids[:,:-1]).float();targets=ids[:,1:]
   valid=(targets>=378)&(targets<2346);values=scores[...,378:2346]
   nll=values.logsumexp(-1)-values.gather(-1,(targets-378).clamp(0,1967)[...,None]).squeeze(-1)
   metric=nll[valid].double().mean().item()
  write_json(out/'proof-validation.json',dict(rows=8,move_ce=metric,move_count=int(valid.sum()),scope='Recovery correctness panel, not quality selection'))
 dist.barrier()
 if rank==0:write_json(out/'done.json',dict(step=step+1,stop_reason=reason,launcher_seconds=time.monotonic()-started,tokens=train.seen*1024))
 dist.destroy_process_group()
if __name__=='__main__':main()
