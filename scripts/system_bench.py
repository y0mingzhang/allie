"""Isolated named system benchmark; never overwrites or submits a Slurm job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modelexp as mx

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--experts', type=int, default=96)
p.add_argument('--micro', type=int, default=64)
p.add_argument('--nproc', type=int, default=8)
p.add_argument('--layers', type=int, default=24)
p.add_argument('--rows', type=int, default=512)
p.add_argument('--steps', type=int, default=100)
p.add_argument('--ckpt-frac', type=float, default=1.)
p.add_argument('--ckpt-blend', action='store_true')
p.add_argument('--fused-blend', action='store_true')
p.add_argument('--fp8', choices=('', 'dense'), default='')
p.add_argument('--snapshot', action='store_true')
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
assert a.name.startswith('csys-') and all(c.isalnum() or c in '-_' for c in a.name)
assert a.rows % (a.nproc*a.micro) == 0
root = Path('/home/yimingz3/src/allie')
source = Path(__file__).resolve().parent
study = root/'results/recipe10x/moe-v1-round1'
plan = json.loads((study/'plan.json').read_text())
r = dict(next(r for r in plan['runs'] if r['v'] == 'moe64k8'))
r |= dict(name=a.name, layers=a.layers, width=2048,
          schedule=r['schedule'] | dict(batch_rows=a.rows))
r['arch'] = (r['arch'] | dict(moe=[a.experts, 6], moe_kernel='scatter-dualgather', moe_shard=False)
             if a.experts else {k:v for k,v in r['arch'].items() if not k.startswith('moe')})
args = list(map(str, mx.train_args(study, r)))
for flag, value in (('--micro-batch',a.micro), ('--initial-batch-rows',a.rows)):
    args[args.index(flag)+1] = str(value)
args += ['--bf16-weights', '--zero2', '--ckpt', 'eager', '--ckpt-frac', str(a.ckpt_frac),
         '--stop-after', str(a.steps), '--max-seconds', '3600']
if a.fp8: args += ['--fp8', a.fp8]
if a.ckpt_blend: args += ['--ckpt-blend']
if a.fused_blend: args += ['--fused-blend']
out = root/'results/pretrain'/a.name
reports = root/'results/recipe10x/moe-perf/codex-systems'
log, receipt = reports/(a.name+'.log'), reports/(a.name+'.json')
assert not out.exists() and not log.exists() and not receipt.exists(), 'run name already used'
env = os.environ | dict(ALLIE_PROJECT_ROOT=str(root), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
    TORCHINDUCTOR_COMPILE_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_MODULE_LOADING='LAZY',
    DISABLE_FP8='1', NCCL_CUMEM_HOST_ENABLE='0', NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='1')
if a.snapshot: env['MEMSNAP'] = str(reports/(a.name+'.pickle'))
else: env.pop('MEMSNAP',None)
cmd = [sys.executable, '-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.nproc}',
       str(source/'modded_train.py'),*args]
meta = dict(args=vars(a),command=cmd,source=str(source),job_id=env.get('SLURM_JOB_ID'),
            snapshot=env.get('MEMSNAP'),status='prepared')
if a.dry_run:
    print(json.dumps(meta,indent=2));sys.exit(0)
reports.mkdir(parents=True,exist_ok=True)
receipt.write_text(json.dumps(meta,indent=2)+'\n')
start = time.monotonic()
with log.open('x') as f:
    rc = subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root,env=env).returncode
meta.update(status='finished',returncode=rc,wall_seconds=time.monotonic()-start)
rows=[]
for line in log.read_text().splitlines():
    if line.startswith('{"step"') and '"tokens_per_second"' in line:
        rows.append(json.loads(line))
if len(rows)>=2:
    lo,hi=rows[-2:];dt=hi['seconds']-lo['seconds']
    meta['measured_window']=dict(start_step=lo['step'],end_step=hi['step'],
        tokens_per_second=(hi['tokens']-lo['tokens'])/dt,max_memory_gb=hi.get('max_memory_gb'),
        mfu=(hi['useful_training_flops']-lo['useful_training_flops'])/dt/(362e12*a.nproc))
receipt.write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta),flush=True)
sys.exit(rc)
