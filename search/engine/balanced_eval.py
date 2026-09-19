"""Score every preregistered method on the frozen balanced golden sample.

Immutable resumable blocks. No fitting/selection occurs in this worker. A cached
four-ply traversal also yields two-ply expectations; standalone timing is already
reported on dev. All sampled positions remain scored, including legal fallbacks.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'
GLOBAL_STOP=Path('/data/group_data/dei-group/yimingz3/allie/controller/STOP')

def atomic(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(p,rows):
    y=np.array([r['target']-378 for r in rows]);ar=np.arange(len(rows))
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1.,atol=1e-6)
    assert (p[ar,y]>0).all(),'Every played move must retain positive mass'
    return -np.log(p[ar,y]),p.argmax(1)==y,p.max(1)


def freeze():
    out=ROOT/'golden-balanced-v1';plan=json.loads((out/'plan.json').read_text())
    files=[Path(__file__), *[Path(__file__).with_name(f) for f in ('tree.py','direct.py','native_mcts.py','mcts.py','adaptive_pilot.py','adaptive_policy.py','policy.py','native_board.py')],Path(__file__).with_name('model.py'),Path(__file__).parent/'sglang_models/allie.py']
    native=ROOT/'runtime/native'
    frozen=dict(plan_sha256=digest(out/'plan.json'),sample_sha256=digest(out/'sample.json'),direct_control_sha256=digest(out/'direct-control.json'),
                source_sha256={str(p.relative_to(Path(__file__).parent)):digest(p) for p in files},
                native_sha256={p.name:digest(p) for p in native.glob('*.so')},
                export_sha256=digest(ROOT/'serving-export/model.safetensors'),
                roots_per_block=32,mcts_roots_per_block=256)
    p=out/'execution.json'
    if p.exists():assert json.loads(p.read_text())==frozen,'Frozen evaluation implementation changed'
    else:atomic(p,frozen)
    return frozen


def run(oracle,spec):
    from .tree import batch
    from .native_mcts import run as released_mcts
    from .adaptive_pilot import traverse
    from .adaptive_policy import output
    proof=ROOT/'engine-queue/011-l40s-recovery.result.json'
    check=json.loads(proof.read_text())
    for key in ('port_vs_reference','cached_vs_fresh'):
        assert abs(check[key]['legal_ce_delta'])<.005 and check[key]['mean_legal_policy_kl']<.005, 'Port verification failed'
    out=ROOT/'golden-balanced-v1';plan=json.loads((out/'plan.json').read_text())
    frozen=freeze()
    control=json.loads((out/'direct-control.json').read_text());plan['methods']['calibrated_direct']=dict(alpha=control['alpha'],beta=0.)
    sample=json.loads((out/'sample.json').read_text());rows=sample['positions']
    assert sample['plan_sha256']==frozen['plan_sha256']
    assert 'clock' not in plan['methods'],'This checkpoint has no side-channel inputs'
    import torch
    runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
    path=out/'runtime.json'
    if path.exists():assert json.loads(path.read_text())==runtime,'Do not silently mix hardware/runtime in one confirmation'
    else:atomic(path,runtime)
    start=time.monotonic();cost=[]
    for lo in range(0,len(rows),32):
        if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():raise RuntimeError('STOP requested')
        dest=out/f'tree-{lo:06d}.npz';part=rows[lo:lo+32]
        if not dest.exists():
            oracle.reset();q,z,stats=batch(part,oracle,batch_size=1024)
            legal=np.isfinite(q[0]);x=z[:,378:2346].astype(float);target=np.array([r['target']-378 for r in part]);ar=np.arange(len(part))
            values={}
            norm=logsumexp(x,axis=1);raw=np.exp(x-norm[:,None]);values['port_raw']=metrics(raw,part)
            for name in ('legal','calibrated_direct','calibrated_two_ply','calibrated_four_ply'):
                p=plan['methods'][name];scores=np.where(legal,p['alpha']*x+p['beta']*np.nan_to_num(q[p.get('depth',1)-1]),-np.inf)
                prob=np.exp(scores-logsumexp(scores,axis=1,keepdims=True));values[name]=metrics(prob,part)
            stats.update(new_tokens=oracle.new_tokens,forward_seconds=oracle.forward_seconds)
            payload={name:np.stack(v,axis=1) for name,v in values.items()}
            tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,**payload,stats=json.dumps(stats),game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]))
            tmp.replace(dest)
        with np.load(dest) as f:cost.append(json.loads(str(f['stats'])))
        if lo%512==0:print('balanced golden tree',lo+len(part),'/',len(rows),flush=True)
    for lo in range(0,len(rows),256):
        if (ROOT/'STOP').exists() or GLOBAL_STOP.exists():raise RuntimeError('STOP requested')
        dest=out/f'mcts-{lo:06d}.npz';part=rows[lo:lo+256]
        if dest.exists():continue
        begin=time.monotonic();oracle.reset();z=oracle([r['prefix'] for r in part])
        a,sa=released_mcts(part,z,oracle,adaptive=True,mean_n_sims=plan['methods']['released_adaptive']['mean_sims'])
        oracle.reset();z=oracle([r['prefix'] for r in part]);params=plan['methods']['repaired_fixed']
        b,sb=traverse(part,z,oracle,np.full(len(part),params['n_sims'],int),np.full(len(part),1.25),repairs=True)
        legal=np.zeros_like(b['values'],bool)
        for i,r in enumerate(part):legal[i,np.array(r['legal'])-378]=True
        prob=output(z[:,378:2346],b['values'],legal,params['alpha'],params['beta'],'reverse')
        # Keep the MCTS-batch direct baseline too, exposing any BF16 batch-layout drift.
        x=z[:,378:2346].astype(float);x=np.where(legal,x,-np.inf)
        direct=np.exp(x-logsumexp(x,axis=1,keepdims=True))
        payload={name:np.stack(metrics(p,part),axis=1) for name,p in [('released_adaptive',a['policy']),('repaired_fixed',prob),('mcts_batch_legal',direct)]}
        tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,**payload,stats=json.dumps(dict(seconds=time.monotonic()-begin,released=sa,repaired=sb)),game=np.array([r['game'] for r in part]),ply=np.array([r['ply'] for r in part]))
        tmp.replace(dest)
        if lo%1024==0:print('balanced golden MCTS',lo+len(part),'/',len(rows),flush=True)
    return dict(stage='All preregistered methods scored; no fitting or selection',positions=len(rows),
                elapsed_seconds=time.monotonic()-start,tree_seconds=sum(c['seconds'] for c in cost),execution=frozen)
