"""Inference budget ladder, separating computation from changing selectivity."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .sample_data import read
from .growforest_native import load as forest_load
from .scaled_count_native import load as backup_load,test
from .handles import HandleOracle


def prepare():
    out=ROOT/'aug-deep-scale-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-tune-expanded-v1/sample.json'
    rows=json.loads(source.read_text())['positions'];chosen=[]
    for cell in range(16):
        for fold in range(2):
            ix=[i for i,r in enumerate(rows) if r['cell']==cell and r['fold']==fold]
            ix.sort(key=lambda i:hashlib.sha256(f"deep-scale-921:{rows[i]['game']}:{rows[i]['ply']}".encode()).digest())
            assert len(ix)>=128;chosen+=ix[:128]
    # Deterministic, score-independent interleaving to keep batches diverse.
    chosen.sort(key=lambda i:hashlib.sha256(f"deep-order-427:{rows[i]['game']}:{rows[i]['ply']}".encode()).digest())
    content=dict(positions=[rows[i] for i in chosen],parent_indices=chosen,parent_sha256=digest(source),
        selection='128 positions per cell and original fit/confirmation fold, by fixed game/ply hash, no scores. All16cells;4096positions. Reused development, potentially training-seen.')
    pp=out/'sample.json'
    if pp.exists():assert json.loads(pp.read_text())==content
    else:atomic(pp,content)
    return out


def run(oracle,spec):
    module=forest_load();reducer=backup_load();test(reducer)
    out=prepare();d=read(out.name);rows=d['rows'];n=len(rows)
    names=('constant','count_decay','budget_normalized')
    deps=('deep_scale.py','growforest.cpp','threadforest.cpp','handleforest.cpp','coverage.cpp','compact.cpp','backups.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py','scaled_count.cpp','diff_backup.cpp','scaled_count_native.py')
    plan=dict(budgets=[1000,4000,16000],methods=list(names),roots_per_batch=64,threads=4,
        sample_sha256=digest(out/'sample.json'),sources={f:digest(Path(__file__).with_name(f)) for f in deps},
        backup='constant tau=.1; current count decay tau=.2/sqrt(1+(descendants-1)/16); normalized tau=.2/sqrt(1+(descendants-1)/(16*budget/1000)). Last two exactly equal at1000.',
        semantics='Grow each tree through1000/4000/16000 on one KV cache; each evaluated node charged once. Same-batch1000 control. Keep root priors/coverage and PUCT2.5 unchanged. Refit the modern output pipeline per method/budget inside training-game folds; no golden access or neural-weight updates.',
        duration='First complete64-root block measures ETA. Atomic completed blocks resume after preemption. Keep compact trees for future CPU-only analyses. No external memory.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    stats=[];begin_all=time.monotonic()
    for lo in range(0,n,plan['roots_per_batch']):
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        path=out/f'{lo:06d}.npz'
        if not path.exists():
            part=rows[lo:lo+plan['roots_per_batch']];size=len(part);begin=time.monotonic();oracle.reset()
            maximum=[0 if len(r['legal'])==1 else 16000 for r in part]
            assert sum(maximum)+sum(len(r['prefix']) for r in part)<oracle.runner.max_total_num_tokens
            assert sum(maximum)+size<oracle.capacity
            bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits
            prefill_tokens=oracle.new_tokens;prefill_seconds=time.monotonic()-begin
            tree=module.Tree([r['prefix'] for r in part],z,[min(b,1000) for b in maximum],[2.5]*size,plan['threads'])
            ids=d['ids'][lo:lo+size].astype(np.int32);qs=[];counts=[];timings=[]
            for budget in plan['budgets']:
                if budget!=1000:tree.grow([min(b,budget) for b in maximum])
                tick=time.monotonic();start_forward=oracle.forward_seconds;start_tokens=oracle.new_tokens
                while not tree.done:
                    h=tree.select()
                    if len(h):tree.update(bridge(h))
                search_seconds=time.monotonic()-tick
                data=tree.compact();base=reducer.Backup(data,budget,16.)
                constant=base.reduce(np.log(.1),0.,ids)[0]
                decayed=base.reduce(np.log(.2),-.5,ids)[0]
                normalized=reducer.Backup(data,budget,16.*budget/1000).reduce(np.log(.2),-.5,ids)[0]
                if budget==1000:np.testing.assert_array_equal(decayed,normalized)
                qs.append(np.stack([constant,decayed,normalized]));counts.append(np.array(tree.evals))
                assert bridge.queries==sum(tree.evals)
                timings.append(dict(budget=budget,search_seconds=search_seconds,
                    forward_seconds=oracle.forward_seconds-start_forward,new_tokens=oracle.new_tokens-start_tokens,
                    queries=int(sum(tree.evals)),tree=tree.stats()))
            stat=dict(job=os.environ.get('SLURM_JOB_ID'),seconds=time.monotonic()-begin,prefill_tokens=prefill_tokens,
                prefill_seconds=prefill_seconds,new_tokens=oracle.new_tokens,stages=timings)
            del tree,bridge,base
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez_compressed(f,**data,z=z,q=np.array(qs),nodes=np.array(counts),ids=ids,mask=d['mask'][lo:lo+size],
                    stats=json.dumps(stat),game=[r['game'] for r in part],ply=[r['ply'] for r in part])
            tmp.replace(path);del data
            stat['seconds_including_serialization']=time.monotonic()-begin
            atomic(out/'progress.json',dict(completed=lo+size,total=n,latest_block=stat,
                approximate_remaining_seconds=(n-lo-size)/size*stat['seconds_including_serialization']))
            print('deep-scale',lo+size,n,stat['seconds_including_serialization'],flush=True)
        with np.load(path) as f:stats.append(json.loads(str(f['stats'])))
        if spec.get('max_blocks') and len(stats)>=spec['max_blocks']:break
    worker=dict(completed=sum(len(np.load(p)['game']) for p in sorted(out.glob('[0-9]*.npz'))),positions=n,
        seconds=time.monotonic()-begin_all,blocks=stats,plan_sha256=digest(pp))
    atomic(out/'worker.json',worker)
    return worker
