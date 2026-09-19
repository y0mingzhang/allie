"""Frozen transfer trees. Labels are used only for inventory checks, never search."""
import json,os,time
from pathlib import Path
import numpy as np
from search.engine.service import ROOT,GLOBAL_STOP,atomic
from search.engine.balanced_eval import digest
from search.engine.growforest_native import load
from search.engine.scaled_count_native import load as backup_load
from search.engine.bellman_projection_native import load as projection_load
from search.engine.budget_policy import route
from .handles import ShipHandles

OUT=ROOT/'transfer-v1'
BUDGETS=[64,128,256,512,1000]


def inventory(split):
    source=ROOT/('golden-balanced-v1' if split=='gold' else 'aug-tune-expanded-v1')/'sample.json'
    rows=json.loads(source.read_text())['positions']
    if split=='aug':
        # Score-independent deterministic slice; retain the existing game folds.
        counts={};picked=[]
        for i,r in enumerate(rows):
            key=(r['cell'],r['fold']);counts[key]=counts.get(key,0)+1
            if counts[key]<=128:picked.append(r)
        rows=picked
        assert len(rows)==4096
        assert not {r['game'] for r in rows if r['fold']==0}&{r['game'] for r in rows if r['fold']==1}
    folder=Path('/data/group_data/dei-group/yimingz3/allie/strat-eval-v1') if split=='gold' else ROOT/'aug-tune-v1'
    with np.load(folder/'strat.npz') as f:tokens,labels=f['rows'],f['labels']
    with np.load(folder/'feats.npz') as f:side=f['feats']
    features=[]
    for r in rows:
        rr,cc=r['row'],r['column'];lo=cc-len(r['prefix'])
        assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,lo:cc],r['prefix'])
        features.append(side[rr,lo:cc].copy())
    return rows,features,dict(sample=digest(source),strat=digest(folder/'strat.npz'),feats=digest(folder/'feats.npz'))


def freeze():
    old=json.loads((ROOT/'golden-bellman-projection-v1/plan.json').read_text())
    adaptive=json.loads((ROOT/'golden-value-of-compute-v1/plan.json').read_text())
    plan=dict(budgets=BUDGETS,roots_per_block=128,threads=4,clock_rule='predicted',
        old_parameters=old['parameters'],adaptive=adaptive['methods']['elo0.01'],budget_policies=adaptive['budget_policies'],
        cpuct=2.5,backup=dict(tau=.2,count_scale=16.,exponent=-.5),projection_strength=3.,
        development='First128 positions per cell per original August fold, chosen without scores.4096 total; August potentially training-seen. Fit fold0 only; report fold1 unchanged.',
        evaluation='Existing8192 golden positions,512/cell. Same-checkpoint full canonical anchor, paired game bootstrap. Reused golden, no method selection here.',
        comparison='Frozen old policy parameters first. Separate calibration-only per-Elo alpha/beta refits on August. Fixed NN-budget curve plus live frozen Elo router, critic projection, shallow policy expectation and Allie baselines.',
        cost='Actual non-root neural evaluations; root prefill tokens/time reported separately. Sequential fixed budgets share cached work. Standalone wall times are not inferred by dividing shared time.',
        clock='Predict elapsed time from this node think-time head; round/clamp and add increment. cf3 previous OWN think time. No observed future moves/clocks. Zero-elapsed sensitivity on development.',
        cm='Vertically anchored prior golden scaling-law shape at model training useful FLOPs/6; conditional estimate, not measured extra training. CE is primary.',
        old_parameters_sha256=digest(ROOT/'golden-bellman-projection-v1/plan.json'),
        router_sha256=digest(ROOT/'golden-value-of-compute-v1/plan.json'),
        laws_sha256=digest(ROOT/'training-cm-laws.json'))
    pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return plan


def run(oracle,spec):
    plan=freeze();size=spec['model'];split=spec['split'];mode=spec.get('mode','fixed')
    assert (OUT/('parity-'+size)/'results.json').exists()
    rows,features,hashes=inventory(split);clock=spec.get('clock_rule','predicted')
    assert mode in ('fixed','adaptive') and clock in ('predicted','zero')
    folder=OUT/f'{size}-{split}-{mode}-{clock}';folder.mkdir(exist_ok=True)
    snap=dict(plan_sha256=digest(OUT/'plan.json'),spec=spec,inventory=hashes,
        source={p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in ('model.py','oracle.py','handles.py','context.py','board.py')],Path(__file__).parent/'sglang_models/allie.py']},
        export=digest(OUT/('ship-'+size+'-export')/'provenance.json'))
    fp=folder/'plan.json'
    if fp.exists():assert json.loads(fp.read_text())==snap,'Changed collection source/config'
    else:atomic(fp,snap)
    native,reducer,projection=load(),backup_load(),projection_load();start=time.monotonic()
    for lo in range(0,len(rows),plan['roots_per_block']):
        dst=folder/f'{lo:06d}.npz'
        if dst.exists():continue
        if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
        part=rows[lo:lo+plan['roots_per_block']];n=len(part);width=max(len(r['legal']) for r in part)
        ids=np.zeros((n,width),np.int32);mask=np.zeros((n,width),bool)
        for i,r in enumerate(part):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True
        tick=time.monotonic();oracle.reset();bridge=ShipHandles(oracle,[r['prefix'] for r in part],features[lo:lo+n],clock)
        z=bridge.root_logits.copy();prefill=oracle.new_tokens;forced=np.array([len(r['legal'])==1 for r in part])
        budgets=BUDGETS if mode=='fixed' else [128]
        tree=native.Tree([r['prefix'] for r in part],z,np.where(forced,0,budgets[0]).tolist(),[2.5]*n,4)
        def advance():
            while not tree.done:
                h=tree.select()
                if len(h):tree.update(bridge(h))
        qq=[];nodes=[];timings=[]
        for b in budgets:
            if b!=budgets[0]:tree.grow(np.where(forced,0,b).tolist())
            advance();compact=tree.compact()
            qq.append(reducer.Backup(compact,b,16.).reduce(np.log(.2),-.5,ids)[0]);nodes.append(np.array(tree.evals));timings.append(time.monotonic()-tick)
        allocated=np.full(n,1000)
        if mode=='adaptive':
            index=route(np.array([r['cell'] for r in part]),None,plan['adaptive'],[128,256,512,1000])
            allocated=np.array([128,256,512,1000])[index];allocated[forced]=0
            tree.grow(allocated.tolist());advance();compact=tree.compact()
            qq=[reducer.Backup(compact,1000,16.).reduce(np.log(.2),-.5,ids)[0]];nodes=[np.array(tree.evals)];timings=[time.monotonic()-tick]
        assert int(nodes[-1].sum())==bridge.queries
        projector=projection.Projection(compact,1000);pt=time.monotonic();info=projector.project(3.)
        projected=projector.reduce(np.log(.2),-.5,ids)[0];projection_seconds=time.monotonic()-pt
        stat=tree.stats();del tree,projector,compact
        stat.update(seconds=time.monotonic()-tick,forward_seconds=oracle.forward_seconds,prefill_tokens=prefill,
            root_queries=n,projection_seconds=projection_seconds,job=os.environ.get('SLURM_JOB_ID'))
        tmp=dst.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,root=z,q=np.array(qq),projected_q=projected,nodes=np.array(nodes),
            ids=ids,mask=mask,allocated=allocated,timings=timings,stats=json.dumps(stat),
            game=[r['game'] for r in part],ply=[r['ply'] for r in part])
        tmp.replace(dst);print('TRANSFER',size,split,mode,clock,lo+n,len(rows),round(stat['seconds'],3),flush=True)
    return dict(model=size,split=split,mode=mode,positions=len(rows),seconds=time.monotonic()-start,folder=str(folder))


if __name__=='__main__':freeze()
