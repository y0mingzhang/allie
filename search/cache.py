"""One-GPU packed-prefix oracle and resumable model-only child-value cache."""
import hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import chess
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/search-v1'
BASE=Path('/data/group_data/dei-group/yimingz3/allie')
SOURCE=BASE/'results/recipe10x/data-v1-round2/source-ours'
CKPT=BASE/'results/pretrain/r2-3e16-control-t20-w20-pf052h-s42/last.pt'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def atomic_npz(path,**arrays):
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f: np.savez(f,**arrays)
    os.replace(tmp,path)

def main():
    torch.backends.cuda.matmul.allow_tf32=True
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    dist.init_process_group('nccl')
    pointer=torch.load(CKPT,map_location='cpu',weights_only=False)
    modelpath=CKPT.parent/pointer['directory']/'model.pt'
    state=torch.load(modelpath,map_location='cpu',weights_only=False)
    for name,h in state['source_sha256'].items(): assert sha(SOURCE/name)==h,name
    sys.path.insert(0,str(SOURCE))
    from modded_medium import Config,core,create_model,make_context
    from chess_vocab import MOVES
    cfg=Config(**state['config'])
    assert not cfg.clock and not cfg.elo,'This initial oracle requires the no-input-channel checkpoint'
    model=create_model(cfg)
    model.scalars.data=state['model']['scalars'].to('cuda',dtype=model.scalars.dtype)
    model.load_state_dict(state['model'])
    inf=state['inference']; model.split_embed=inf['split_embed']
    for k in ('angular_freq','cos','sin'): getattr(model.yarn,k).copy_(inf['yarn'][k].to('cuda'))
    model.yarn.attn_scale=inf['yarn']['attn_scale']; model.eval()
    sched=core.ForwardScheduleConfig(None,inf['ws_short'],inf['ws_long'])
    rotary_cos=model.yarn.cos.clone();rotary_sin=model.yarn.sin.clone()
    net=torch.compile(model,dynamic=False,fullgraph=True)
    data=json.loads((OUT/'dev.json').read_text()); positions=data['positions']
    identity=dict(checkpoint_sha256=sha(modelpath),data_sha256=sha(OUT/'dev.json'),
        code_sha256=sha(__file__),source=str(SOURCE),top_k=8,pack_tokens=4096,rotary='per-document canonical coordinates',alignment=128)
    manifest=OUT/'cache-identity.json'
    serve_only='--serve-only' in sys.argv
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        compare=lambda d:{k:v for k,v in d.items() if k!='code_sha256'}
        assert previous==identity or (serve_only and compare(previous)==compare(identity)),'cache identity changed'
    else: manifest.write_text(json.dumps(identity,indent=2)+'\n')
    pack_tokens=4096; stats=dict(calls=0,prefix_tokens=0,padded_tokens=0,forward_seconds=0.)
    def forward(prefixes,all_tokens=False):
        outputs=[]; batch=[]; used=0
        def flush():
            if not batch: return
            x=np.full(pack_tokens,2348,dtype=np.int64); idx=[]; cursor=0
            rotary_pos=np.zeros(pack_tokens,dtype=np.int64);actual_tokens=0
            for p in batch:
                x[cursor:cursor+len(p)]=p;rotary_pos[cursor:cursor+len(p)]=np.arange(len(p))
                idx.extend(range(cursor,cursor+len(p))) if all_tokens else idx.append(cursor+len(p)-1)
                actual_tokens+=len(p);cursor+=((len(p)+127)//128)*128
            xt=torch.as_tensor(x,device='cuda').reshape(1,-1)
            rp=torch.as_tensor(rotary_pos,device='cuda')
            model.yarn.cos[:pack_tokens].copy_(rotary_cos[rp]);model.yarn.sin[:pack_tokens].copy_(rotary_sin[rp])
            ctx=make_context(xt,sched.ws_short*128,sched.ws_long*128)
            torch.cuda.synchronize(); start=time.monotonic()
            with torch.inference_mode():
                logits=net(xt.flatten(),xt.flatten(),ctx,sched).reshape(-1,2432)
                y=logits[torch.as_tensor(idx,device='cuda')].float().cpu().numpy()
            torch.cuda.synchronize(); stats['forward_seconds']+=time.monotonic()-start
            stats['calls']+=1; stats['prefix_tokens']+=actual_tokens; stats['padded_tokens']+=pack_tokens
            outputs.extend(y)
        for p in prefixes:
            assert 11<=len(p)<=1025 and p[0]==2348 and 2348 not in p[1:]
            size=((len(p)+127)//128)*128
            if used+size>pack_tokens: flush(); batch=[]; used=0
            batch.append(p); used+=size
        flush()
        return np.array(outputs)
    started=time.monotonic()
    # BF16 rotary tables are quantized at absolute packed coordinates. Test
    # probability-level drift, rather than a max over thousands of tiny-probability logits.
    probes=positions[:32]
    reference=np.array([forward([p['prefix']])[0] for p in probes])
    packed=forward([p['prefix'] for p in probes])
    def logprob(x):
        x=x[:,378:2346].astype(np.float64); z=x-x.max(1,keepdims=True)
        return z-np.log(np.exp(z).sum(1,keepdims=True))
    ref_lp,pack_lp=logprob(reference),logprob(packed)
    target=np.array([p['target']-378 for p in probes]);ar=np.arange(len(probes))
    bias=float(np.mean(ref_lp[ar,target]-pack_lp[ar,target]))
    kl=float(np.mean(np.sum(np.exp(ref_lp)*(ref_lp-pack_lp),axis=1)))
    diff=float(np.max(np.abs(reference-packed)))
    warmup=dict(seconds=time.monotonic()-started,max_packed_logit_difference=diff,
                packed_minus_reference_ce=bias,mean_policy_kl=kl,
                caveat='Canonical per-document rotary coordinates; final baseline must also include unchanged official packing',**stats)
    (OUT/'oracle-check.json').write_text(json.dumps(warmup,indent=2)+'\n')
    print('oracle warmup',warmup,flush=True)
    assert kl<1e-5 and abs(bias)<0.001,('packing drift too large for pilot',warmup)
    stats={k:0 for k in stats}
    (OUT/'oracle-check.json').write_text(json.dumps(warmup,indent=2)+'\n')
    print('oracle warmup',warmup,flush=True)
    if serve_only:
        with np.load(OUT/'cache-00000.npz') as z:
            repeat=forward([p['prefix'] for p in positions[:128]])
            assert np.array_equal(repeat,z['root']),'Transport-only restart changed cached predictions'
        (OUT/'server-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    for lo in ([] if serve_only else range(0,len(positions),128)):
        dest=OUT/f'cache-{lo:05d}.npz'
        if dest.exists():
            with np.load(dest) as z: assert int(z['lo'])==lo and z['root'].shape[0]==min(128,len(positions)-lo)
            continue
        rows=positions[lo:lo+128]; root=forward([p['prefix'] for p in rows])
        cand=np.full((len(rows),8),-1,np.int16); terminal=np.full((len(rows),8),np.nan,np.float32)
        child_prefix=[]; where=[]
        for i,p in enumerate(rows):
            legal=np.array(p['legal']); moves=legal[np.argsort(root[i,legal])[::-1][:8]]
            cand[i,:len(moves)]=moves
            for j,t in enumerate(moves):
                board=chess.Board(p['fen']); board.push(chess.Move.from_uci(MOVES[int(t)-378]))
                if board.is_checkmate(): terminal[i,j]=1.
                elif board.is_stalemate() or board.is_insufficient_material(): terminal[i,j]=.5
                else: child_prefix.append(p['prefix']+[int(t)]); where.append((i,j))
        child=np.full((len(rows),8,3),np.nan,np.float32)
        # Child WDL is from the next mover's perspective. CPU analysis flips it.
        if child_prefix:
            child_logits=forward(child_prefix)
            for k,(i,j) in enumerate(where): child[i,j]=child_logits[k,2413:2416]
        atomic_npz(dest,lo=lo,root=root,candidates=cand,child_wdl=child,terminal=terminal)
        report=dict(gpu=torch.cuda.get_device_name(),completed=lo+len(rows),total=len(positions),
            elapsed_seconds=time.monotonic()-started,**stats)
        (OUT/'cache-progress.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    # Persistent model server: new requests reuse this exact loaded model and compiled net.
    if '--serve' in sys.argv:
        queue=OUT/'queue';queue.mkdir(exist_ok=True)
        for interrupted in queue.glob('*.running.json'):
            interrupted.replace(interrupted.with_name(interrupted.name.replace('.running.json','.request.json')))
        from rpc import make_server
        http=make_server(forward,OUT,lambda:dict(stats))
        ready=dict(url=f'http://{os.uname().nodename}:{http.server_port}',job_id=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,
                   checkpoint_sha256=identity['checkpoint_sha256'],gpu=torch.cuda.get_device_name(),pid=os.getpid())
        (OUT/'server-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
        print('Persistent oracle ready',ready,flush=True)
        next_disk_poll=0.
        while True:
            http.handle_request()
            if time.monotonic()<next_disk_poll:continue
            next_disk_poll=time.monotonic()+1.
            if (OUT/'STOP').exists():break
            work=sorted(queue.glob('*.request.json'))
            if not work: continue
            for request in work:
                running=request.with_name(request.name.replace('.request.json','.running.json'))
                request.replace(running)
                try:
                    r=json.loads(running.read_text()); operation=r['op']
                    if operation=='ping': result=ready|dict(stats=stats)
                    elif operation=='score':
                        inputs=Path(r['input']).resolve(); output=Path(r['output']).resolve()
                        assert inputs.is_relative_to(OUT.resolve()) and output.is_relative_to(OUT.resolve())
                        payload=json.loads(inputs.read_text()); begin=time.monotonic()
                        predictions=forward(payload['prefixes'])
                        atomic_npz(output,logits=predictions)
                        result=dict(output=str(output),prefixes=len(predictions),seconds=time.monotonic()-begin,
                                    input_sha256=sha(inputs),checkpoint_sha256=identity['checkpoint_sha256'])
                    else: raise ValueError('Unknown oracle operation '+operation)
                    done=running.with_name(running.name.replace('.running.json','.done.json'))
                    tmp=done.with_suffix('.partial');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(done)
                except Exception:
                    running.with_name(running.name.replace('.running.json','.error.txt')).write_text(traceback.format_exc())
                finally: running.unlink(missing_ok=True)
        (OUT/'server-ready.json').unlink(missing_ok=True)
    dist.destroy_process_group()

if __name__=='__main__': main()
