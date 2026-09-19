"""Paired search-quality test of FP32 WDL readout, with unchanged neural weights."""
import json,time,types,os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from .service import ROOT,GLOBAL_STOP,atomic
from .balanced_eval import digest
from .sample_data import read
from .growforest_native import load
from .scaled_count_native import load as backup_load
from .handles import HandleOracle
from .stack_fit import fit_stack
from .analyze_august import means

OUT=ROOT/'aug-precise-critic-v1'


@contextmanager
def precise_critic(oracle):
    processor=oracle.runner.model.logits_processor;original=processor._compute_lm_head
    def forward(self,hidden_states,lm_head,embedding_bias=None):
        result=original(hidden_states,lm_head,embedding_bias).clone()
        wdl=torch.nn.functional.linear(hidden_states.float(),lm_head.weight[2413:2416].float())
        result[...,2413:2416]=23*torch.sigmoid((wdl+5)/7.5)
        return result
    processor._compute_lm_head=types.MethodType(forward,processor)
    try:yield
    finally:processor._compute_lm_head=original


def freeze():
    source=ROOT/'aug-conditioned-search-v1';OUT.mkdir(exist_ok=True)
    plan=dict(sample_sha256=digest(ROOT/'aug-deep-scale-v1/sample.json'),control_study=source.name,control_method='actual',
        control_plan_sha256=digest(source/'plan.json'),precision_diagnostic_sha256=digest(ROOT/'critic-precision-audit-v1/results.json'),
        sources={name:digest(Path(__file__).with_name(name)) for name in ('precise_search.py','growforest.cpp','threadforest.cpp','handleforest.cpp','coverage.cpp','compact.cpp','mcts_native.hpp','board.cpp','handles.py','direct.py','scaled_count.cpp','diff_backup.cpp','stack_fit.py')},
        roots_per_block=256,threads=4,simulations=1000,
        semantics='Only3WDL readout rows useFP32 linear and softcap. All weights, transformer, move/time logits and rating inputs unchanged. Same128→1000 schedule and soft backup. The critic affects both PUCT selection and leaf values. No quality claim from the prior one-tree numerical diagnostic.',
        evaluation='Same4096August cohort, nested game-independent output calibration perarm, choose using fitCV and report all checking results/intervals. Actual BF16 control from same A100, verified against numerical audit. No golden or CM conversion.',
        cost='Every NN node, root prefill and actual wall time reported. FP32 readout overhead is paid even if node count is unchanged.')
    pp=OUT/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    return plan


def run(oracle,spec):
    plan=freeze();d=read('aug-deep-scale-v1');rows=d['rows'];n=len(rows);begin=time.monotonic()
    source=ROOT/plan['control_study'];assert (source/'results.json').exists(),'Wait for completed BF16 control'
    if not spec.get('smoke'):
        p=ROOT/'engine-queue/105-precise-critic-smoke.result.json'
        assert p.exists() and json.loads(p.read_text()).get('smoke')=='passed'
    native,reducer=load(),backup_load();folder=OUT/'fp32_critic';folder.mkdir(exist_ok=True)
    with precise_critic(oracle):
        for lo in range(0,n,plan['roots_per_block']):
            if spec.get('smoke') and lo>0:break
            dest=folder/f'{lo:06d}.npz'
            if dest.exists():continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists():raise RuntimeError('STOP')
            hi=min(n,lo+plan['roots_per_block']);part=rows[lo:hi];ids=d['ids'][lo:hi].astype(np.int32);forced=d['mask'][lo:hi].sum(1)==1
            tick=time.monotonic();oracle.reset();bridge=HandleOracle(oracle,[r['prefix'] for r in part]);z=bridge.root_logits.astype(float)
            with np.load(source/'actual'/dest.name) as f:
                keep=np.ones(2432,bool);keep[2413:2416]=False
                np.testing.assert_array_equal(z[:,keep],f['root'][:,keep])
            prefill=oracle.new_tokens;tree=native.Tree([r['prefix'] for r in part],z,np.where(forced,0,128).tolist(),[2.5]*len(part),plan['threads'])
            def advance():
                while not tree.done:
                    h=tree.select()
                    if len(h):tree.update(bridge(h))
            advance();tree.grow(np.where(forced,0,1000).tolist());advance();compact=tree.compact()
            q=reducer.Backup(compact,1000,16.).reduce(np.log(.2),-.5,ids)[0];nodes=np.array(tree.evals);assert int(nodes.sum())==bridge.queries
            stats=tree.stats();del tree,compact
            assert np.isfinite(q).all() and np.max(abs(q))<=1+1e-12
            stats.update(seconds=time.monotonic()-tick,prefill_tokens=prefill,forward_seconds=oracle.forward_seconds,root_queries=len(part),job=os.environ.get('SLURM_JOB_ID'))
            temp=dest.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,q=q,root=z,nodes=nodes,stats=json.dumps(stats),game=d['games'][lo:hi],ply=[r['ply'] for r in part])
            temp.replace(dest);print('precise-critic',hi,n,stats['seconds'],flush=True)
    if spec.get('smoke'):return dict(smoke='passed',positions=plan['roots_per_block'],seconds=time.monotonic()-begin)
    return analyze(time.monotonic()-begin)


def analyze(elapsed=None):
    plan=freeze();d=read('aug-deep-scale-v1');rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];n=len(rows);source=ROOT/plan['control_study']
    with np.load(source/'scores.npz') as f:
        np.testing.assert_array_equal(f['games'],games);base=f['loss'][list(f['names']).index('actual')]
    rec0=json.loads((source/'results.json').read_text())['results']['actual']
    root=np.zeros((n,2432));q=np.zeros(d['mask'].shape);nodes=np.zeros(n);stats=[]
    for path in sorted((OUT/'fp32_critic').glob('[0-9]*.npz')):
        with np.load(path) as f:
            lo=int(path.stem);hi=lo+len(f['game']);np.testing.assert_array_equal(f['game'],games[lo:hi]);root[lo:hi]=f['root'];q[lo:hi]=f['q'];nodes[lo:hi]=f['nodes'];stats.append(json.loads(str(f['stats'])))
    assert len(stats)==n//plan['roots_per_block']
    inv=ROOT/'aug-tune-v1';manifest=json.loads((inv/'manifest.json').read_text());assert digest(inv/'feats.npz')==manifest['files_sha256']['feats.npz']
    with np.load(inv/'feats.npz') as f:side=f['feats']
    seconds=np.array([side[r['row'],r['column']-1,0] for r in rows])
    rec,ll,_=fit_stack(rows,root,q,d['ids'],d['mask'],d['target'],cells,fm,d['cv'],seconds);cost=means(nodes[~fm],cells[~fm])
    rec.update(mean_nodes=float(cost.mean()),expert_mean_nodes=float(cost[3::4].mean()),seconds=sum(s['seconds'] for s in stats),prefill_tokens=sum(s['prefill_tokens'] for s in stats))
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    draws=np.random.default_rng(71256).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@ct;assert (den>0).all()
    delta=ll-base;pt=means(delta[~fm],cells[~fm]);s=np.zeros((ng,16));np.add.at(s,(ix,cells[~fm]),delta[~fm]);b=draws@s/den
    rec['delta_vs_parent']=dict(macro=float(pt.mean()),expert=float(pt[3::4].mean()),macro_ci95=np.quantile(b.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(b[:,3::4].mean(1),[.025,.975]).tolist(),cells=pt.tolist())
    records=dict(bf16=rec0,fp32_critic=rec);selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    atomic(OUT/'results.json',dict(results=records,fit_cv_selected=selected,elapsed_seconds=elapsed,plan_sha256=digest(OUT/'plan.json')))
    np.savez_compressed(OUT/'scores.npz',names=list(records),loss=np.stack([base,ll]),cells=cells,games=games,fit=fm)
    print('PRECISE RESULT',rec['confirmation'],rec['delta_vs_parent'],selected,flush=True)
    return dict(study=OUT.name,fit_cv_selected=selected,seconds=elapsed)
