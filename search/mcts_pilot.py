"""Run required Allie baselines on the immutable development pilot only."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from allie_mcts import budgets, run, REFERENCE
from client import Oracle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/search-v1/mcts-pilot'


def main():
    OUT.mkdir(exist_ok=True)
    data = ROOT / 'results/search-v1/dev.json'
    rows = json.loads(data.read_text())['positions']
    logits = np.concatenate([np.load(ROOT/f'results/search-v1/cache-{i:05d}.npz')['root']
                             for i in range(0,len(rows),128)])
    fit = np.array([r['fold']==0 for r in rows])
    matched = int(round(budgets(logits[fit]).mean()))
    methods = dict(fixed50=dict(n_sims=50), adaptive50=dict(adaptive=True,mean_n_sims=50),
                   fixed_matched=dict(n_sims=matched))
    manifest = dict(reference=REFERENCE, data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        methods=methods, fixed_matched_selected_on='fit fold0 predicted simulation count; no move labels',
        checkpoint=Oracle().ready['checkpoint_sha256'],
        implementation_sha256=hashlib.sha256((ROOT/'search/allie_mcts.py').read_bytes()).hexdigest())
    plan = OUT/'plan.json'
    if plan.exists():
        assert json.loads(plan.read_text())==manifest
    else:
        plan.write_text(json.dumps(manifest,indent=2)+'\n')
    oracle=Oracle();started=time.monotonic()
    for name,params in methods.items():
        for start in range(0,len(rows),128):
            dest=OUT/f'{name}-{start:05d}.npz'
            if dest.exists():continue
            pred,stats=run(rows[start:start+128],logits[start:start+128],oracle,**params)
            tmp=dest.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez(f,**pred,stats=json.dumps(stats))
            tmp.replace(dest)
            print(name,start+len(pred['policy']),stats,'total_seconds',time.monotonic()-started,flush=True)
            gc.collect()
    report={}
    for name in methods:
        arrays=[np.load(OUT/f'{name}-{i:05d}.npz') for i in range(0,len(rows),128)]
        p=np.concatenate([z['policy'] for z in arrays]); n=np.concatenate([z['simulations'] for z in arrays])
        target=np.array([r['target']-378 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
        losses=-np.log(p[np.arange(len(rows)),target]);correct=p.argmax(1)==target
        assert np.isfinite(losses).all()
        report[name]={}
        for fold in (0,1):
            mask=np.array([r['fold']==fold for r in rows]); e=mask&expert
            report[name][f'fold{fold}']=dict(ce=float(losses[mask].mean()),expert_ce=float(losses[e].mean()),
                accuracy=float(correct[mask].mean()),expert_accuracy=float(correct[e].mean()),
                mean_simulations=float(n[mask].mean()))
        report[name]['cost']={key:sum(json.loads(str(z['stats']))[key] for z in arrays)
                              for key in ('seconds','simulations','evaluated_leaves','terminal_visits','requests','useful_prefix_tokens')}
        for z in arrays:z.close()
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
