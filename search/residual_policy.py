"""Test search as a correction to an amortized human policy, using cached dev trees.

If a critic exactly predicts outcomes under the model's continuation policy, the
tower property makes shallow and deeper policy-expectation values agree. Their
disagreement is model inconsistency, not automatically an improvement. MCTS adds
an optimizing continuation, so its disagreement also contains a behavioral shift.
Fit separate shallow and search coefficients instead of assuming both deserve
the same positive reward. Selection is game CV inside dev fold0, never golden.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1] / 'results/search-v1'


def main():
    start = time.monotonic()
    rows = json.loads((ROOT/'dev.json').read_text())['positions']
    n = len(rows)
    ar = np.arange(n)
    ids = np.zeros((n, max(len(r['legal']) for r in rows)), int)
    mask = np.zeros_like(ids, bool)
    y = np.zeros(n, int)
    elo = np.zeros(n)
    for i, r in enumerate(rows):
        ids[i, :len(r['legal'])] = np.array(r['legal'])-378
        mask[i, :len(r['legal'])] = True
        y[i] = r['legal'].index(r['target'])
        offset = 3 if (len(r['prefix'])-11) % 2 == 0 else 7
        elo[i] = sum(r['prefix'][offset+j]*10**(3-j) for j in range(4))
    fit = np.array([r['fold'] == 0 for r in rows])
    cells = np.array([r['cell'] % 4 for r in rows])
    expert = cells == 3
    folds = np.array([int(hashlib.sha256(('mcts-output-cv:'+r['game']).encode()).hexdigest()[:8], 16) % 3 for r in rows])
    ef = np.clip((elo-1000)/1600, 0, 1)[:, None]
    hashes = {}
    depth = []
    depth_nodes = np.zeros(6)
    for lo in range(0, n, 16):
        path = ROOT/f'depth6-pilot/{lo:06d}.npz'
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            assert list(f['game']) == [r['game'] for r in rows[lo:lo+16]]
            assert list(f['ply']) == [r['ply'] for r in rows[lo:lo+16]]
            depth.append(f['q'])
            depth_nodes += json.loads(str(f['stats']))['leaves_by_depth']
    depth = np.concatenate(depth, axis=1)
    gather = lambda a: np.where(mask, a[ar[:, None], ids], 0.)
    shallow, deep = gather(2*depth[0]-1), gather(2*depth[-1]-1)
    assert np.isfinite(shallow).all() and np.isfinite(deep).all()
    all_results = {}
    for budget, method in [(1000, 'fixed_repairs'), (4000, 'released_fixed')]:
        source = ROOT/f'mcts{budget}-pilot'
        plan = json.loads((source/'plan.json').read_text())
        bs = plan['spec']['roots_per_batch']
        chunks = []
        nodes = 0
        for lo in range(0, n, bs):
            path = source/f'{method}-{lo:06d}.npz'
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as f:
                assert list(f['game']) == [r['game'] for r in rows[lo:lo+bs]]
                assert list(f['ply']) == [r['ply'] for r in rows[lo:lo+bs]]
                chunks.append({k:f[k] for k in ('root','values','visits')})
                nodes += json.loads(str(f['stats']))['evaluated_leaves']
        data = {k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
        z = gather(data['root'][:,378:2346].astype(float))
        q = gather(data['values'].astype(float))
        v = gather(data['visits'].astype(float))
        logp = np.where(mask, z, -np.inf)
        logp -= logsumexp(logp, axis=1, keepdims=True)
        prior = np.exp(logp)
        ratio = np.where(mask, np.log(np.maximum(v+16*prior,1e-300))
                         -np.log(v.sum(1,keepdims=True)+16)-np.where(mask,logp,0.),0.)
        split = lambda a:[a*(1-ef), a*ef]
        positive = [(0.,32.)]*2
        signed = [(-32.,32.)]*2
        arms = {
            'elo_q': (split(q), positive, nodes),
            'elo_q_visit': (split(q)+[ratio], positive+[(-2.,2.)], nodes),
            'elo_q_shallow': (split(q)+split(shallow), positive+signed, nodes+depth_nodes[0]),
            'elo_shallow': (split(shallow), positive, depth_nodes[0]),
            'elo_residual_only': (split(q-shallow), signed, nodes+depth_nodes[0]),
            'elo_q_expectation': (split(q)+split(deep), positive+signed, nodes+depth_nodes.sum()),
        }
        records = {}
        for name,(parts,bounds,cost) in arms.items():
            x = np.stack([z,*parts],axis=-1)
            bounds = [(.5,2.),*bounds]

            def optimize(m):
                xx,ok,yy=x[m],mask[m],y[m]
                aa=np.arange(len(yy))
                def objective(w):
                    zz=np.where(ok,xx@w,-np.inf)
                    norm=logsumexp(zz,axis=1)
                    p=np.exp(zz-norm[:,None])
                    return float((norm-zz[aa,yy]).mean()),np.einsum('na,nak->k',p,xx)/len(yy)-xx[aa,yy].mean(0)
                opt=minimize(objective,[1.,*([0.]*(x.shape[-1]-1))],jac=True,method='L-BFGS-B',bounds=bounds,
                             options=dict(ftol=1e-12,gtol=1e-8,maxiter=500))
                assert opt.success,opt.message
                return opt.x

            def losses(w):
                logits=np.where(mask,x@w,-np.inf)
                return logsumexp(logits,axis=1)-logits[ar,y],logits.argmax(1)==y

            cv=np.full(n,np.nan)
            for fold in range(3):
                ls,_=losses(optimize(fit&(folds!=fold)))
                m=fit&(folds==fold);cv[m]=ls[m]
            w=optimize(fit);ls,correct=losses(w)
            records[name]=dict(coefficients=w.tolist(),estimated_standalone_dev_nodes=float(cost/n),
                cost_note='Sum of source tree node counts; overlapping shallow/MCTS evaluations could be reused in an implementation, but this estimate does not assume that saving.',
                cv_ce=float(cv[fit].mean()),cv_expert_ce=float(cv[fit&expert].mean()),
                cv_rating_macro=float(np.mean([cv[fit&(cells==k)].mean() for k in range(4)])),
                confirmation=dict(ce=float(ls[~fit].mean()),expert_ce=float(ls[~fit&expert].mean()),
                    accuracy=float(correct[~fit].mean()),expert_accuracy=float(correct[~fit&expert].mean())))
            print(budget,name,records[name]['confirmation'],flush=True)
        all_results[str(budget)]=dict(results=records,
            selected_by_fit_cv_macro=min(records,key=lambda k:records[k]['cv_rating_macro']),
            selected_by_fit_cv_expert=min(records,key=lambda k:records[k]['cv_expert_ce']))
    out=ROOT/'residual-policy-dev';out.mkdir(exist_ok=True)
    report=dict(results=all_results,elapsed_seconds=time.monotonic()-start,cache_sha256=hashes,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Cached development-only ablation. Fixed three-way game CV in fit fold; every reused confirmation-fold arm reported. No golden-law CM conversion.',
        training_equivalent_cm=None)
    tmp=out/'results.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'results.json')


if __name__=='__main__':
    main()
