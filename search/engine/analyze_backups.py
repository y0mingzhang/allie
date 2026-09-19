"""Shared fixed output calibration isolates backup semantics at equal tree cost."""
import json
from pathlib import Path
import time
import numpy as np
from .adaptive_policy import output

ROOT=Path(__file__).resolve().parents[2]/'results/search-v1'


def main():
    start=time.monotonic();folder=ROOT/'backup-ladder-dev'
    plan=json.loads((folder/'plan.json').read_text());rows=json.loads((ROOT/'dev.json').read_text())['positions']
    n=len(rows);ar=np.arange(n)
    ids=np.zeros((n,max(len(r['legal']) for r in rows)),int);mask=np.zeros_like(ids,bool);y=np.zeros(n,int)
    for i,r in enumerate(rows):ids[i,:len(r['legal'])]=np.array(r['legal'])-378;mask[i,:len(r['legal'])]=True;y[i]=r['legal'].index(r['target'])
    fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows]);games=np.array([r['game'] for r in rows])
    chunks=[];zs=[];costs=[];stats=[]
    for lo in range(0,n,128):
        with np.load(folder/f'{lo:06d}.npz') as f:
            assert list(f['game'])==[r['game'] for r in rows[lo:lo+128]]
            chunks.append(f['q']);zs.append(f['root']);costs.append(f['evaluated_nodes']);stats.append(json.loads(str(f['stats'])))
            with np.load(ROOT/f'mcts1000-pilot/fixed_repairs-{lo:06d}.npz') as old:
                np.testing.assert_array_equal(f['q'][-1,0],old['values'])
                np.testing.assert_array_equal(f['root'],old['root'])
                np.testing.assert_array_equal(f['visits'][-1],old['visits'])
    q=np.concatenate(chunks,axis=2);z=np.concatenate(zs)[:,378:2346].astype(float)[ar[:,None],ids]
    costs=np.concatenate(costs,axis=1)
    params=json.loads((ROOT/'mcts1000-pilot/results.json').read_text())['calibration']['reverse']
    # Same fixed reverse-KL output at every budget and backup. No fitting here.
    records={}
    for b,budget in enumerate(plan['budgets']):
        losses={}
        for k,name in enumerate(plan['backup_names']):
            values=q[b,k][ar[:,None],ids]
            p=output(z,values,mask,**params,direction='reverse')
            loss=-np.log(p[ar,y]);correct=p.argmax(1)==y;losses[name]=loss
            records[f'{budget}_{name}']=dict(budget=budget,backup=name,mean_nodes=float(costs[b].mean()),
                expert_mean_nodes=float(costs[b,expert].mean()),
                metrics={label:dict(ce=float(loss[m].mean()),expert_ce=float(loss[m&expert].mean()),
                    accuracy=float(correct[m].mean()),expert_accuracy=float(correct[m&expert].mean())) for label,m in [('fit',fit),('confirmation',~fit)]})
        # Paired game-level uncertainty vs the original backed-up MCTS value.
        _,ix=np.unique(games[~fit],return_inverse=True);g=ix.max()+1
        w=np.random.default_rng(613802).multinomial(g,np.full(g,1/g),size=2000)
        for name in plan['backup_names']:
            rec=records[f'{budget}_{name}'];rec['delta_vs_mcts']={}
            delta=(losses[name]-losses['mcts_average'])[~fit]
            for label,m in [('all',np.ones((~fit).sum(),bool)),('expert',expert[~fit])]:
                total=np.bincount(ix,weights=delta*m,minlength=g);count=np.bincount(ix,weights=m,minlength=g)
                sample=(w@total)/(w@count).clip(1)
                rec['delta_vs_mcts'][label]=dict(delta=float(delta[m].mean()),ci95=np.quantile(sample,[.025,.975]).tolist())
            print(budget,name,rec['mean_nodes'],rec['metrics']['confirmation'],flush=True)
    report=dict(stage='Blitz development-only equal-tree comparison, fixed reverse-KL calibration inherited from fixed1000 fit fold',
        calibration=params,results=records,exact_final_mcts_cache_identity=True,training_equivalent_cm=None,
        timing=dict(gpu_worker_seconds=sum(x['seconds'] for x in stats),analysis_seconds=time.monotonic()-start),
        note='No golden conversion or method selection. Same nodes at each budget; alternate backups add CPU work. Balanced dev is needed for generalization.')
    temp=folder/'results.partial';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(folder/'results.json')


if __name__=='__main__':main()
