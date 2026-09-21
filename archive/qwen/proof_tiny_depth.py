"""Actual single-GPU signal/recovery for the depth-parameterized trainer."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/yimingz3/src/allie')
OUT=ROOT/'results/recipe10x/tiny-scaling-v1'
SOURCE=OUT/'source-ours'

def load(run):
    import torch
    p=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    d=run/p['directory']
    return torch.load(d/'model.pt',map_location='cpu',weights_only=False),torch.load(d/'rank0.pt',map_location='cpu',weights_only=False)

def compare(a,b):
    import numpy as np,torch
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b,(a,b)

def main():
    out=OUT/'proof';out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists()
    assert json.loads((OUT/'cpu-check.json').read_text())['passed']
    python=subprocess.check_output([sys.executable,str(SOURCE/'modded_runtime_stage.py')],text=True).strip()
    # Fully verified original corpus; cache stage is charged inside the proof.
    stage=ROOT/'results/recipe10x/selected-curvature-v1/selected128-third/stage_corpus.py'
    subprocess.run([sys.executable,str(stage)],check=True,timeout=600)
    job=os.environ['SLURM_JOB_ID'];prefix=f'tiny-depth-proof-{job}'
    schedule=dict(warmup_steps=1,mtp_steps=2,split_step=5,batch_rows=8,plateau=1.,final_lr=.2,decay_shape='linear')
    base=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(SOURCE/'modded_train.py'),
          '--layers','8','--width','128','--head-dim','64','--steps','16','--extension-steps','0','--initial-batch-rows','8',
          '--micro-batch','1','--eval-every','16','--checkpoint-every','3','--val-rows','4','--deterministic','--keep-checkpoints','2',
          '--max-seconds','1200','--data','/scratch/yimingz3/allie/lichess_tokens_v2','--wsd-schedule',json.dumps(schedule),
          '--wsd-end-step','16','--wsd-decay-start','1']
    names={k:prefix+'-'+k for k in ('full','signal')}
    def run(kind,extra):
        with (out/(kind+'.log')).open('w') as log:
            subprocess.run(base+extra,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1400)
    run('full',['--name',names['full']])
    split=ROOT/'results/pretrain'/names['signal']
    with (out/'signal.log').open('w') as log:
        proc=subprocess.Popen(base+['--name',names['signal']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+900
        while not (split/'last.pt').exists():
            assert proc.poll() is None,'Trainer failed before checkpoint'
            assert time.monotonic()<deadline,'Checkpoint timeout'
            time.sleep(.1)
        todo=[proc.pid];target=None
        while todo:
            pid=todo.pop();p=Path('/proc')/str(pid)
            try:
                for task in (p/'task').iterdir():todo.extend(map(int,(task/'children').read_text().split()))
                argv=(p/'cmdline').read_bytes().split(b'\0')
                if any(x.endswith(b'/modded_train.py') for x in argv) and b'LOCAL_RANK=0' in (p/'environ').read_bytes().split(b'\0'):target=pid
            except (FileNotFoundError,ProcessLookupError):pass
        assert target is not None
        os.kill(target,signal.SIGUSR1);assert proc.wait(timeout=240)==0
    done=json.loads((split/'done.json').read_text());assert done['stop_reason']=='signal' and 3<=done['step']<16
    run('resume',['--name',names['signal'],'--resume',str(split/'last.pt')])
    full,fr=load(ROOT/'results/pretrain'/names['full']);resumed,rr=load(split)
    for k in ('model','inference','metrics','best_move_ce','useful_training_flops','tokens'):compare(full[k],resumed[k])
    compare(fr,rr)
    # Check the other changed-depth shape through actual compiled optimizer steps.
    smoke=base.copy();smoke[smoke.index('--layers')+1]='12';smoke[smoke.index('--width')+1]='256'
    with (out/'depth12.log').open('w') as log:
        subprocess.run(smoke+['--name',prefix+'-depth12'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1400)
    state,_=load(ROOT/'results/pretrain'/(prefix+'-depth12'));assert state['step']==16
    report=dict(passed=True,production_driver=True,world_size=1,actual_process_signal=True,
        saved_step=done['step'],bitwise_exact_model_optimizer_rng_data_flops=True,exact_validation=True,
        compiled_depth12_optimizer_smoke=True,quality_ineligible=True,job_id=int(job),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.glob('*.py')})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'accepted.json').write_text(json.dumps(dict(passed=True,report_sha256=hashlib.sha256((out/'report.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
