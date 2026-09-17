"""Freeze the user-approved width/depth grid. Preparation never submits jobs."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/recipe10x'
OUT=BASE/'tiny-scaling-v1'
STOP=Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')
SHAPES={128:8,256:12,512:16}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)


def batch(path,name,gpu,hours,command,proof=False):
    partition,qos=('general','normal') if gpu=='L40S' else ('dei-group','dei_group_qos')
    exclude='#SBATCH --exclude=babel-o5-20,babel-o5-24,babel-n5-32,babel-q5-32\n' if gpu=='L40S' else ''
    mins=round(hours*60)
    path.write_text(f'''#!/bin/bash
#SBATCH --account=dippolit
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:{gpu}:1
#SBATCH --time={mins//60:02}:{mins%60:02}:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --job-name={name}
#SBATCH --output={ROOT}/logs/%x-%j.out
{exclude}set -euo pipefail
cd {ROOT}
export ALLIE_PROJECT_ROOT={ROOT}
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_MODULE_LOADING=LAZY DISABLE_FP8=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR=/scratch/yimingz3/allie/tiny-scaling-v1-inductor
export TRITON_CACHE_DIR=/scratch/yimingz3/allie/tiny-scaling-v1-triton
{command}
''')
    subprocess.run(['bash','-n',str(path)],check=True)


def source():
    assert not OUT.exists(),'Never overwrite a frozen package'
    OUT.mkdir()
    old=BASE/'schedule-quality-v1/source';dest=OUT/'source-ours';shutil.copytree(old,dest,ignore=shutil.ignore_patterns('__pycache__'))
    p=dest/'modded_medium.py';t=p.read_text()
    t=replace(t,'assert cfg.layers == 16 and','assert cfg.layers in (8, 12, 16) and')
    t=replace(t,'core.args = SimpleNamespace(','core.args = SimpleNamespace(\n        num_layers=cfg.layers,')
    p.write_text(t)
    p=dest/'modded_medium_core.py';t=p.read_text()
    t=replace(t,'if custom_sizing and dist.get_world_size()==8:',"if custom_sizing and dist.get_world_size()==8 and args.num_layers == 16:")
    t=replace(t,'if layer_idx in [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]:',
        'if layer_idx < min(5, args.num_layers//2) or layer_idx >= args.num_layers-min(5, args.num_layers//2):')
    t=replace(t,'nn.Embedding(vocab_size, model_dim) for _ in range(5)',
        'nn.Embedding(vocab_size, model_dim) for _ in range(min(5, num_layers//2))')
    t=replace(t,'skip_in = [2, 4, 6]','skip_in = [i*self.num_layers//16 for i in (2, 4, 6)]')
    t=replace(t,'skip_out = [9, 10, 11]','skip_out = [9*self.num_layers//16+i for i in range(3)]')
    t=replace(t,'backout_layer = 11','backout_layer = skip_out[-1]')
    start=t.index('        bm_sizes = [long_bm, short_bm, short_bm, short_bm, long_bm,')
    end=t.index('        assert len(bm_sizes)',start)
    t=t[:start]+'''        long_layers = {round(i*(self.num_layers-1)/15) for i in (0, 4, 11, 15)}
        bm_sizes = [long_bm if i in long_layers else short_bm for i in range(self.num_layers)]
'''+t[end:]
    t=replace(t,'ve = [ve[0], ve[1], ve[2], ve[3], ve[4]] + [None] * (self.num_layers - 10) + [ve[0], ve[1], ve[2], ve[3], ve[4]]',
        've = ve + [None] * (self.num_layers - 2*len(ve)) + ve')
    p.write_text(t)
    p=dest/'modded_train.py';t=p.read_text()
    t=replace(t,"p.add_argument('--width', type=int, default=512)","p.add_argument('--width', type=int, default=512)\n    p.add_argument('--layers', type=int, default=16, choices=(8,12,16))")
    t=replace(t,'cfg = Config(width=a.width, head_dim=a.head_dim,','cfg = Config(width=a.width, layers=a.layers, head_dim=a.head_dim,')
    t=replace(t,"for key in ('width', 'head_dim', 'extension_steps'","for key in ('width', 'layers', 'head_dim', 'extension_steps'")
    p.write_text(t)
    for f in dest.glob('*.py'):compile(f.read_text(),str(f),'exec')
    write(OUT/'source-change.json',dict(original=str(old),frozen=str(dest),
        changes={f.name:dict(before=sha(old/f.name),after=sha(f)) for f in dest.glob('*.py') if sha(old/f.name)!=sha(f)},
        intent='Parameterize depth; preserve16-layer topology exactly. Scale skips/window positions and value-embedding placement for8/12 layers. Training objective/optimizer unchanged.'))


def prepare(counts):
    assert not STOP.exists() and not (OUT/'prepared.json').exists()
    proposal=read(BASE/'tiny-scaling-proposal-20260917/proposal.json')
    ours_template=BASE/'selected-curvature-v1/selected128-third'
    q_template=BASE/'native-curvature-v1/native128-third'
    source_dir=OUT/'source-ours'
    for template,kind in [(ours_template,'ours'),(q_template,'qwen_recipe')]:
        p=read(template/'plan.json')
        for name,h in p['source_sha256'].items():
            assert sha((Path(p['source']) if kind=='ours' else template)/name)==h
    corpus=Path('/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2')
    manifest=read(corpus/'manifest.json');assert len(manifest['files'])==200
    for entry in manifest['files']:assert (corpus/entry['path']).stat().st_size==entry['size']
    design=dict(at=datetime.now(timezone.utc).isoformat(),goal='High-fidelity L(N,D) for ours and Qwen; plain additive Chinchilla law.',
        shapes=SHAPES,token_horizons=[s*512*1024 for s in (256,512,1024,2048)],
        new_cells=24,parallelism=dict(L40S=8,A6000=16),
        bound_gpu_hours=dict(scaling_wsd=26.,science_a6000=28.5),proof_bound_A6000_gpu_hours=.5,
        source_change_sha256=sha(OUT/'source-change.json'),cpu_check_sha256=sha(OUT/'cpu-check.json'),
        training_rule='Frozen optimizer and schedule; independent cooldown at every endpoint; no board/search/distillation.',
        analysis_rule='Fit first3 tiny horizons, predict fourth before refitting all. Report leave-size-out checks, residuals and model-form limitations; no interaction term.',
        original_dataset_revision=manifest['revision'],corpus_manifest_sha256=sha(corpus/'manifest.json'),
        old_anchor_scope='128M and483M fixed16-layer controls are extra validation points; explicitly flag changed aspect ratio when pooling.',
        final_tests_accessed=False,final_pool_used=False)
    write(OUT/'design.json',design)
    prepared={}
    for cell in proposal['cells']:
        recipe,width,steps=cell['recipe'],cell['width'],cell['steps'];layers=SHAPES[width]
        key=f"{'ours' if recipe=='ours' else 'qwen'}-l{layers}-w{width}-s{steps}"
        dest=OUT/key;dest.mkdir();ev=dest/'evaluator';ev.mkdir()
        gpu=cell['gpu_type'];phase='scaling_wsd' if gpu=='L40S' else 'science_a6000'
        hours=cell['max_training_hours'];params=counts[recipe][str(width)]
        old=ours_template if recipe=='ours' else q_template;p=read(old/'plan.json')
        p.update(at=design['at'],purpose='tiny-scaling-v1 '+key,phase=phase,gpu_type=gpu,gpus_per_trial=1,
            layers=layers,width=width,parameters=params,steps=steps,end_step=steps,tokens=steps*512*1024,
            tokens_per_trial=steps*512*1024,max_seconds=int(hours*3600),max_hours_per_trial=hours,
            training_envelope_gpu_hours=hours,matching_envelope_gpu_hours=.25,design_sha256=sha(OUT/'design.json'))
        trial='quality';runname='tiny-v1-'+key
        if recipe=='ours':
            p.update(source=str(source_dir),source_sha256={f.name:sha(f) for f in source_dir.iterdir() if f.is_file()},
                trials={trial:dict(name=runname,schedule=dict(warmup_steps=32,mtp_steps=64,split_step=65,batch_rows=512,plateau=4.,final_lr=.2,decay_shape='linear'))},
                recovery_report=str(OUT/'proof/report.json'),recovery_report_sha256=None)
            t=(old/'run.py').read_text()
            t=replace(t,"'--width',str(plan['width']),'--head-dim'","'--width',str(plan['width']),'--layers',str(plan['layers']),'--head-dim'")
            t=replace(t,"assert digest(proof)==plan['recovery_report_sha256']","assert digest(proof)==json.loads((PACKAGE.parent/'proof/accepted.json').read_text())['report_sha256']")
            t=replace(t,"report['actual_rank1_signal']","report['actual_process_signal']")
            dry="    if a.dry_run:\n        print(json.dumps(command(plan,trial,plan['python'],plan['max_seconds'])))\n        return\n"
            t=replace(t,dry,'');t=replace(t,"    proof = Path(plan['recovery_report'])",dry+"    proof = Path(plan['recovery_report'])")
            t=replace(t,"entry['phase']=='scaling_wsd' and entry['gpus']==1 and entry['gpu_type']=='L40S'","entry['phase']==plan['phase'] and entry['gpus']==1 and entry['gpu_type']==plan['gpu_type']")
            t=replace(t,"account['committed_gpu_hours']['scaling_wsd']<=LIMITS['scaling_wsd']","account['committed_gpu_hours'][plan['phase']]<=LIMITS[plan['phase']]")
            t=replace(t,"account['scheduled_gpus']<=16","account['scheduled_gpus']<=24")
            (dest/'run.py').write_text(t);shutil.copyfile(old/'stage_corpus.py',dest/'stage_corpus.py')
        else:
            for f in p['worker_files']+['stage_corpus.py']:shutil.copyfile(old/f,dest/f)
            p.update(shape=dict(layers=layers,width=width,ff={128:448,256:896,512:1728}[width],heads=width//128,kv_heads=max(1,width//256),head_dim=128),
                     trials={trial:dict(name=runname,lr=.01,schedule='linear')},micro=16)
            t=(old/'run.py').read_text()
            t=replace(t,"entry['phase'] == 'scaling_wsd' and entry['gpus'] == plan['gpus_per_trial'] and entry['gpu_type'] == 'L40S'","entry['phase'] == plan['phase'] and entry['gpus'] == 1 and entry['gpu_type'] == plan['gpu_type']")
            assert "account['scheduled_gpus'] <= account['max_concurrent_gpus']" in t
            t=replace(t,"account['committed_gpu_hours']['scaling_wsd'] <= LIMITS['scaling_wsd']","account['committed_gpu_hours'][plan['phase']] <= LIMITS[plan['phase']]")
            t=replace(t,"assert not (ROOT/'controller/STOP').exists()",f"assert not Path('{STOP}').exists()")
            (dest/'run.py').write_text(t)
        # Preserve training/stop logic; only launch metadata and shape arguments change.
        batch(dest/'run.sbatch','tiny-'+key,gpu,hours,f'exec .venv/bin/python {dest}/run.py --trial quality')
        if recipe=='ours':p['launcher_sha256']={f:sha(dest/f) for f in ('run.py','run.sbatch','stage_corpus.py')}
        else:p['source_sha256']={f.name:sha(f) for f in dest.iterdir() if f.is_file()}
        write(dest/'plan.json',p)
        ep=read(old/'evaluator/plan.json')
        for name,h in ep['source_sha256'].items():
            assert sha(old/'evaluator'/name)==h
            shutil.copyfile(old/'evaluator'/name,ev/name)
        t=(ev/'run.py').read_text()
        if recipe=='ours':
            t=replace(t,"entry['phase'] == 'scaling_wsd' and entry['gpus'] == 1","entry['phase'] == plan['phase'] and entry['gpus'] == 1")
            t=replace(t,"entry['gpu_type'] == 'L40S'","entry['gpu_type'] == plan['gpu_type']")
            command=f'exec .venv/bin/python {ev}/run.py'
        else:
            t=replace(t,"entry['phase']=='scaling_wsd' and entry['gpus']==1 and entry['gpu_type']=='L40S' and entry['max_hours']<=.5","entry['phase']==plan['phase'] and entry['gpus']==1 and entry['gpu_type']==plan['gpu_type'] and entry['max_hours']<=.25")
            t=replace(t,'1740-(time.monotonic()-started)','840-(time.monotonic()-started)')
            pp=ev/'prepare.py';pp.write_text(replace(pp.read_text(),"identity['tensors']==179","identity['tensors']==11*training['shape']['layers']+3"))
            command=f'.venv/bin/python {ev}/prepare.py\nexec .venv/bin/python {ev}/run.py'
        (ev/'run.py').write_text(t)
        batch(ev/'run.sbatch','tiny-'+key+'-eval',gpu,.25,command)
        ep.update(at=design['at'],phase=phase,gpu_type=gpu,max_hours=.25,training_plan=str(dest/'plan.json'),training_plan_sha256=sha(dest/'plan.json'),
                  source_sha256={f.name:sha(f) for f in ev.iterdir() if f.is_file()})
        write(ev/'plan.json',ep)
        for folder in (dest,ev):
            for f in folder.glob('*.py'):compile(f.read_text(),str(f),'exec')
        subprocess.run([sys.executable,str(dest/'run.py'),'--trial','quality','--dry-run'],check=True,stdout=subprocess.DEVNULL)
        prepared[key]=dict(recipe=recipe,width=width,layers=layers,steps=steps,parameters=params,gpu_type=gpu,phase=phase,
            training_hours=hours,evaluation_hours=.25,plan_sha256=sha(dest/'plan.json'),eval_plan_sha256=sha(ev/'plan.json'))
    write(OUT/'prepared.json',dict(passed=True,design_sha256=sha(OUT/'design.json'),packages=prepared,jobs_submitted=False))
    print('Prepared',len(prepared),'cells')


if __name__=='__main__':
    assert not STOP.exists()
    if sys.argv[1]=='source':source()
    elif sys.argv[1]=='packages':prepare(read(OUT/'cpu-check.json')['parameters'])
    else:raise ValueError(sys.argv)
