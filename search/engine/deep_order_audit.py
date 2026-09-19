"""Frozen mixed-depth repeat/permutation audit, with one diagnostic tree trace."""
import json, time, os
from pathlib import Path
import numpy as np
from .service import ROOT, GLOBAL_STOP, atomic
from .balanced_eval import digest
from .sample_data import read
from .growforest_native import load
from .scaled_count_native import load as backup_load
from .handles import HandleOracle
from .budget_policy import policy, route
from .deep_frontier_live import features
from .analyze_august import means

OUT = ROOT / 'aug-deep-order-audit-v1'


def freeze():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'aug-deep-frontier-live-v1'
    base = json.loads((source / 'plan.json').read_text())
    for name, sha in base['sources'].items():
        assert digest(Path(__file__).with_name(name)) == sha
    n = len(read('aug-deep-scale-v1')['rows'])
    plan = dict(base=base, base_plan_sha256=digest(source/'plan.json'),
        source_sha256=digest(Path(__file__)),
        methods={'fixed1000-repeat': 'fixed1000', 'fixed1000-repeat2': 'fixed1000', 'fixed1000-shuffle': 'fixed1000', 'state2000-repeat': 'state2000', 'state2000-shuffle': 'state2000'},
        shuffle=np.random.default_rng(630952).permutation(n).tolist(),
        reference_blocks={name: {p.name: digest(p) for p in sorted((source/name).glob('[0-9]*.npz'))}
                          for name in ('fixed1000', 'state2000')},
        trace_position=3463,
        purpose='Same 256-root batches and frozen coefficients. Repeat fixed1000 twice on this allocation, then fixed permutation. Repeat and permute state2000 too. New hardware may differ from cached reference; use within-allocation contrasts to isolate ordering. No selection/refit or golden scoring. Target-selected outlier trace is diagnostic only.',
        threshold='Absolute macro or expert CE drift above .001 requires investigation before promotion. Report full policy differences and paired algorithm differences, not just this threshold.')
    pp = OUT/'plan.json'
    if pp.exists(): assert json.loads(pp.read_text()) == plan
    else: atomic(pp, plan)
    return plan


def run(oracle, spec):
    plan = freeze(); base = plan['base']; start = time.monotonic()
    d = read('aug-deep-scale-v1'); rows = d['rows']; n = len(rows)
    assert digest(ROOT/'aug-deep-scale-v1/sample.json') == base['sample_sha256']
    inv = ROOT/'aug-tune-v1'
    for filename in ('strat.npz', 'feats.npz'): assert digest(inv/filename) == base['sidecars'][filename]
    with np.load(inv/'feats.npz') as f: side = f['feats']
    seconds = np.array([side[r['row'],r['column']-1,0] for r in rows])
    module, reducer = load(), backup_load(); budgets = np.array(base['budgets'])
    for name, reference in plan['methods'].items():
        order = np.array(plan['shuffle']) if name.endswith('shuffle') else np.arange(n)
        folder = OUT/name; folder.mkdir(exist_ok=True)
        for lo in range(0,n,base['roots_per_block']):
            path = folder/f'{lo:06d}.npz'
            if path.exists(): continue
            if GLOBAL_STOP.exists() or (ROOT/'STOP').exists(): raise RuntimeError('STOP')
            ix = order[lo:lo+base['roots_per_block']]; part = [rows[i] for i in ix]; size = len(ix)
            ids, mask, sec = d['ids'][ix].astype(np.int32), d['mask'][ix], seconds[ix]
            forced = mask.sum(1)==1; tick = time.monotonic(); oracle.reset()
            bridge = HandleOracle(oracle,[r['prefix'] for r in part]); z = bridge.root_logits.astype(float)
            prefill_tokens = oracle.new_tokens; prefill_seconds = time.monotonic()-tick
            tree = module.Tree([r['prefix'] for r in part],z,np.where(forced,0,128).tolist(),[2.5]*size,base['threads'])
            trace_paths = {}; trace = {}
            if reference=='fixed1000' and plan['trace_position'] in ix:
                ti = int(np.flatnonzero(ix==plan['trace_position'])[0])
                trace_paths[ti] = tuple(part[ti]['prefix']); trace[trace_paths[ti]] = z[ti]
            def advance():
                while not tree.done:
                    handles = tree.select()
                    if len(handles):
                        logits = bridge(handles)
                        if trace:
                            for handle, value in zip(handles,logits):
                                node,parent,token,_ = map(int,handle)
                                if parent in trace_paths:
                                    key = trace_paths[parent]+(token,); trace_paths[node]=key; trace[key]=value
                        tree.update(logits)
            advance(); compact = tree.compact()
            q128 = reducer.Backup(compact,128,16.).reduce(np.log(.2),-.5,ids)[0]; del compact
            index = np.full(size,3) if reference=='fixed1000' else route(d['cells'][ix],features(part,z,q128,ids,mask,sec),base['methods'][reference],budgets)
            chosen = budgets[index].copy(); chosen[forced]=0
            assert sum(chosen)+sum(len(r['prefix']) for r in part) < oracle.runner.max_total_num_tokens
            assert sum(chosen)+size < oracle.capacity
            tree.grow(chosen.tolist()); advance(); compact=tree.compact(); q=np.zeros(mask.shape)
            scales=16*np.maximum(1,budgets[index]/1000)
            for scale in np.unique(scales):
                take=scales==scale
                q[take]=reducer.Backup(compact,16000,float(scale)).reduce(np.log(.2),-.5,ids)[0][take]
            nodes=np.array(tree.evals); assert int(nodes.sum())==bridge.queries
            stats=tree.stats(); del tree,compact
            p=np.zeros(mask.shape)
            for bi,b in enumerate(budgets):
                take=index==bi
                if take.any():p[take]=policy([part[i] for i in np.flatnonzero(take)],z[take],q[take],ids[take],mask[take],sec[take],base['budget_policies'][str(b)])
            stats.update(seconds=time.monotonic()-tick,prefill_tokens=prefill_tokens,prefill_seconds=prefill_seconds,
                         forward_seconds=oracle.forward_seconds,job=os.environ.get('SLURM_JOB_ID'))
            if trace:
                atomic(folder/'trace-paths.json',[list(k) for k in trace])
                np.savez_compressed(folder/'trace-logits.npz',z=np.stack(list(trace.values())))
            temp=path.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,policy=p,root=z,q=q,q128=q128,nodes=nodes,budgets=chosen,index=index,position_index=ix,stats=json.dumps(stats),game=d['games'][ix],ply=[r['ply'] for r in part])
            temp.replace(path); print('deep-order',name,lo+size,n,stats['seconds'],flush=True)
    return analyze(time.monotonic()-start)


def analyze(elapsed=None):
    from scipy.special import softmax
    plan=freeze(); d=read('aug-deep-scale-v1'); n=len(d['rows']); ar=np.arange(n); fm=d['fit']; cells=d['cells']
    source=ROOT/'aug-deep-frontier-live-v1'; data={}; timing={}
    folders={name:source/name for name in ('fixed1000','state2000')} | {name:OUT/name for name in plan['methods']}
    for name,folder in folders.items():
        item=dict(policy=np.zeros(d['mask'].shape),root=np.zeros((n,2432)),q=np.zeros(d['mask'].shape),nodes=np.zeros(n),budgets=np.zeros(n,int)); seen=np.zeros(n,bool); timing[name]=0.
        for path in sorted(folder.glob('[0-9]*.npz')):
            if name in plan['reference_blocks']:assert digest(path)==plan['reference_blocks'][name][path.name]
            with np.load(path) as f:
                ix=f['position_index'] if 'position_index' in f else np.arange(int(path.stem),int(path.stem)+len(f['game']))
                assert not seen[ix].any(); seen[ix]=True
                np.testing.assert_array_equal(f['game'],d['games'][ix]);np.testing.assert_array_equal(f['ply'],[d['rows'][i]['ply'] for i in ix])
                for k in item:item[k][ix]=f[k]
                timing[name]+=json.loads(str(f['stats']))['seconds']
        assert seen.all(); np.testing.assert_allclose(item['policy'].sum(1),1.,atol=1e-13)
        item['loss']=-np.log(item['policy'][ar,d['target']]);data[name]=item
    result={}
    contrasts = dict(plan['methods'])
    contrasts.update({'fixed1000-repeat2':'fixed1000-repeat', 'fixed1000-shuffle':'fixed1000-repeat', 'state2000-shuffle':'state2000-repeat'})
    for name,ref in contrasts.items():
        a,b=data[name],data[ref];delta=a['loss']-b['loss'];ce=means(a['loss'][~fm],cells[~fm]);gap=means(delta[~fm],cells[~fm]);diff=np.abs(a['policy']-b['policy'])
        result[name]=dict(reference=ref,macro_ce=float(ce.mean()),expert_ce=float(ce[3::4].mean()),mean_nodes=float(means(a['nodes'][~fm],cells[~fm]).mean()),
            delta_macro=float(gap.mean()),delta_expert=float(gap[3::4].mean()),exceeds_001=bool(max(abs(gap.mean()),abs(gap[3::4].mean()))>.001),
            max_policy_diff=float(diff.max()),policy_diff_quantiles=np.quantile(diff.max(1),[.5,.9,.99,1]).tolist(),max_q_diff=float(np.max(np.abs(a['q']-b['q']))),max_root_logit_diff=float(np.max(np.abs(a['root']-b['root']))),
            changed_budget_fraction=float(np.mean(a['budgets']!=b['budgets'])),seconds=timing[name])
    traces={}
    for name in ('fixed1000-repeat','fixed1000-shuffle'):
        keys=json.loads((OUT/name/'trace-paths.json').read_text())
        with np.load(OUT/name/'trace-logits.npz') as f:traces[name]={tuple(k):z for k,z in zip(keys,f['z'])}
    a,b=traces.values();common=sorted(set(a)&set(b));za=np.stack([a[k] for k in common]);zb=np.stack([b[k] for k in common])
    va=softmax(za[:,2413:2416].astype(float),axis=1);vb=softmax(zb[:,2413:2416].astype(float),axis=1)
    trace=dict(position=plan['trace_position'],paths=[len(a),len(b)],common_paths=len(common),max_logit_diff=float(np.max(np.abs(za-zb))),
        value_diff_quantiles=np.quantile(np.abs((va[:,0]-va[:,2])-(vb[:,0]-vb[:,2])),[.5,.9,.99,1]).tolist(),
        root_ce=[float(data[k]['loss'][plan['trace_position']]) for k in ('fixed1000','fixed1000-repeat','fixed1000-shuffle')])
    pairs={}
    for label,a,b in [('original','state2000','fixed1000'),('new_hardware','state2000-repeat','fixed1000-repeat'),('shuffle','state2000-shuffle','fixed1000-shuffle')]:
        gap=means((data[a]['loss']-data[b]['loss'])[~fm],cells[~fm]);pairs[label]=dict(macro=float(gap.mean()),expert=float(gap[3::4].mean()))
    output=dict(results=result,paired_algorithm_deltas=pairs,trace=trace,elapsed_seconds=elapsed,plan_sha256=digest(OUT/'plan.json'))
    atomic(OUT/'results.json',output);print('DEEP ORDER AUDIT',json.dumps(output),flush=True)
    return output


if __name__=='__main__':analyze()
