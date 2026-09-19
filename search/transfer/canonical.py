"""Read-only original-evaluator baseline on unchanged golden rows, own outputs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
BASE=Path('/data/group_data/dei-group/yimingz3/allie')
MODELS={'large':('msh-3e17-model-pf115h-s42','model-v1-ship3e17'),
        'small':('mo1-3e16-dense-pf052h-s42','moe-v1-round1')}


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('size',choices=MODELS);a=ap.parse_args()
    run,study=MODELS[a.size];source=BASE/'results/recipe10x'/study/'source-ours'
    out=ROOT/'transfer-v1'/('canonical-'+a.size);out.mkdir(exist_ok=True)
    if (out/'results.json').exists():return
    start=time.monotonic();pointer=BASE/'results/pretrain'/run/'last.pt'
    ptr=torch.load(pointer,map_location='cpu',weights_only=False);path=pointer.parent/ptr['directory']/'model.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    for name,expected in state['source_sha256'].items():assert sha(source/name)==expected,name
    official=BASE/'results/lm-eval'/run/'strat-v1.json';report=json.loads(official.read_text())
    assert sha(path)==report['model_sha256'] and sha(pointer)==report['checkpoint_sha256']
    sys.path.insert(0,str(source))
    from modded_medium import Config,create_model,make_context,core
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.use_deterministic_algorithms(state['args'].get('deterministic',False));torch.cuda.set_device(0)
    dist.init_process_group('nccl');cfg=Config(**state['config']);model=create_model(cfg)
    model.scalars.data=state['model']['scalars'].to(device='cuda',dtype=model.scalars.dtype)
    model.load_state_dict(state['model']);inf=state['inference'];model.split_embed=inf['split_embed']
    for k in ('angular_freq','cos','sin'):getattr(model.yarn,k).copy_(inf['yarn'][k].to('cuda'))
    model.yarn.attn_scale=inf['yarn']['attn_scale'];model.eval()
    net=torch.compile(model,dynamic=False,fullgraph=True)
    schedule=core.ForwardScheduleConfig(None,inf['ws_short'],inf['ws_long'])
    gold=BASE/'strat-eval-v1';manifest=json.loads((gold/'manifest.json').read_text())
    assert sha(gold/'strat.npz')==manifest['sha256']==report['strat_sha256']
    assert sha(gold/'feats.npz')==manifest['feats_sha256']
    with np.load(gold/'strat.npz') as f:rows=f['rows'];labels=f['labels'][:,1:]
    with np.load(gold/'feats.npz') as f:feats=f['feats']
    sample=json.loads((ROOT/'golden-balanced-v1/sample.json').read_text())['positions']
    sr=np.array([r['row'] for r in sample]);sc=np.array([r['column']-1 for r in sample])
    logits=np.zeros((len(sample),2432),np.float32);sums=np.zeros(16);counts=np.zeros(16,np.int64)
    batch=16;assert cfg.max_tokens//1024>=batch
    for lo in range(0,len(rows),batch):
        if (ROOT/'STOP').exists() or (BASE/'controller/STOP').exists():raise RuntimeError('STOP')
        dst=out/f'{lo:06d}.npz';hi=min(lo+batch,len(rows));take=np.flatnonzero((sr>=lo)&(sr<hi))
        if not dst.exists():
            with torch.inference_mode():
                ids=torch.tensor(rows[lo:hi,:-1].astype(np.int64),device='cuda')
                fs=torch.tensor(feats[lo:hi,:-1],device='cuda')
                if hi-lo<batch:
                    ids=torch.cat((ids,ids.new_full((batch-(hi-lo),1024),2348)))
                    fs=torch.cat((fs,fs.new_full((batch-(hi-lo),1024,3),-1)))
                ctx=make_context(ids,schedule.ws_short*128,schedule.ws_long*128)
                z=net(ids.flatten(),ids.flatten(),ctx,schedule,feat_seq=fs.flatten(0,1)).reshape(batch,1024,-1)[:hi-lo]
                lab=torch.tensor(labels[lo:hi],device='cuda');valid=lab>=0
                move=z[...,378:2346].float();target=torch.tensor(rows[lo:hi,1:].astype(np.int64),device='cuda')-378
                loss=torch.logsumexp(move[valid],-1)-move[valid].gather(1,target[valid].unsqueeze(1)).squeeze(1)
                ls=loss.cpu().numpy();ll=lab[valid].cpu().numpy()
                bs=np.bincount(ll,weights=ls,minlength=16);bc=np.bincount(ll,minlength=16)
                zz=z[torch.tensor(sr[take]-lo,device='cuda'),torch.tensor(sc[take],device='cuda')].float().cpu().numpy()
            temp=dst.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,logits=zz,indices=take,sums=bs,counts=bc)
            temp.replace(dst)
        with np.load(dst) as f:
            np.testing.assert_array_equal(f['indices'],take);logits[take]=f['logits'];sums+=f['sums'];counts+=f['counts']
        if lo%(batch*16)==0:print('canonical',a.size,hi,len(rows),time.monotonic()-start,flush=True)
    means=sums/counts;expected=np.array(list(report['cells'].values()))
    assert np.array_equal(counts,np.array(list(report['counts'].values())))
    delta=means-expected
    # A hardware change may change BF16 rounding, but must not materially move CE.
    assert abs(delta).max()<.003,delta
    np.savez_compressed(out/'sample.npz',logits=logits,game=[r['game'] for r in sample],ply=[r['ply'] for r in sample])
    result=dict(run=run,step=state['step'],model_sha256=sha(path),source_sha256=state['source_sha256'],
        official_report_sha256=sha(official),sample_sha256=sha(ROOT/'golden-balanced-v1/sample.json'),
        strat_sha256=manifest['sha256'],feats_sha256=manifest['feats_sha256'],torch=torch.__version__,
        macro=float(means.mean()),expert_macro=float(means[3::4].mean()),cells=means.tolist(),counts=counts.tolist(),
        cell_delta_vs_official=delta.tolist(),seconds=time.monotonic()-start,job=os.environ.get('SLURM_JOB_ID'))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('CANONICAL DONE',a.size,result['macro'],result['expert_macro'],flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
