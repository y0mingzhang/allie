"""Fit output policy on fit games only; reuse it across budget-allocation variants."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .adaptive_policy import output

ROOT = Path(__file__).resolve().parents[2]/'results/search-v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    out=Path(args.directory).resolve();assert out.is_relative_to(ROOT.resolve())
    plan=json.loads((out/'plan.json').read_text());spec=plan['spec']
    rows=json.loads((ROOT/'dev.json').read_text())['positions'];n=len(rows)
    bs=int(spec.get('roots_per_batch',256));methods=spec.get('methods',['released_fixed','released_time','decoupled_time','shuffled_time','entropy'])
    assert 'released_fixed' in methods
    fit=np.array([r['fold']==0 for r in rows]);expert=np.array([r['cell']%4==3 for r in rows])
    target=np.array([r['target']-378 for r in rows]);ar=np.arange(n)
    maxlegal=max(len(r['legal']) for r in rows)
    ids=np.zeros((n,maxlegal),int);legal=np.zeros_like(ids,bool);y=np.zeros(n,int)
    for i,r in enumerate(rows):
        a=np.array(r['legal'])-378;ids[i,:len(a)]=a;legal[i,:len(a)]=True;y[i]=np.flatnonzero(a==target[i])[0]
    trees={};cost={};roots=None
    for method in methods:
        chunks=[];cost[method]=[]
        for lo in range(0,n,bs):
            with np.load(out/f'{method}-{lo:06d}.npz') as f:
                assert list(f['game'])==[r['game'] for r in rows[lo:lo+bs]]
                assert list(f['ply'])==[r['ply'] for r in rows[lo:lo+bs]]
                chunks.append({k:f[k] for k in ('root','values','policy','visits','simulations')})
                cost[method].append(json.loads(str(f['stats'])))
        arrays={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        if roots is None:roots=arrays['root']
        else:np.testing.assert_array_equal(roots,arrays['root'])
        trees[method]=arrays
    z=roots[:,378:2346].astype(float)[ar[:,None],ids]
    q=trees['released_fixed']['values'][ar[:,None],ids].astype(float)
    xf=np.stack([z[fit],q[fit]],-1);mask=legal[fit];yf=y[fit];af=np.arange(fit.sum())
    def objective(w):
        logits=np.where(mask,np.einsum('nak,k->na',xf,w),-np.inf)
        norm=logsumexp(logits,axis=1);p=np.exp(logits-norm[:,None])
        return float((norm-logits[af,yf]).mean()),np.einsum('na,nak->k',p,xf)/len(yf)-xf[af,yf].mean(0)
    opt=minimize(objective,[1.,2.],jac=True,method='L-BFGS-B',bounds=[(.5,2.),(0.,32.)],options=dict(ftol=1e-12,gtol=1e-8))
    assert opt.success,opt.message
    trials=[]
    for alpha in (.8,.9,1.,1.1):
        for beta in (.25,.5,1.,2.,4.,8.):
            p=output(z[fit],q[fit],legal[fit],alpha,beta,'reverse')
            trials.append((float(-np.log(p[af,yf]).mean()),alpha,beta))
    _,ra,rb=min(trials)
    calibration=dict(forward=dict(alpha=float(opt.x[0]),beta=float(opt.x[1])),reverse=dict(alpha=ra,beta=rb))
    games=np.array([r['game'] for r in rows]);ug,ix=np.unique(games[~fit],return_inverse=True)
    rng=np.random.default_rng(779123)
    weights=rng.multinomial(len(ug),np.full(len(ug),1/len(ug)),size=2000)
    results={};losses={}
    def record(name,p,tree):
        loss=-np.log(p[ar,y]);assert np.isfinite(loss).all();losses[name]=loss
        pred=ids[ar,p.argmax(1)];correct=pred==target
        results[name]=dict(metrics={label:dict(ce=float(loss[m].mean()),expert_ce=float(loss[m&expert].mean()),
            accuracy=float(correct[m].mean()),expert_accuracy=float(correct[m&expert].mean())) for label,m in [('fit',fit),('confirmation',~fit)]},
            tree=tree,training_equivalent_cm=None)
    record('legal',output(z,q,legal,beta=0),None)
    for method in methods:
        t=trees[method];q=t['values'][ar[:,None],ids].astype(float)
        if method in ('released_fixed','released_time','fixed_repairs','released_time_repairs'):
            record(method,t['policy'][ar[:,None],ids],method)
        for direction,params in calibration.items():
            record(method+'_'+direction,output(z,q,legal,direction=direction,**params),method)
    for name,result in results.items():
        result['paired_delta_vs_legal']={}
        for label,mask in [('overall',np.ones(n,bool)),('expert',expert)]:
            m=mask[~fit];delta=(losses[name]-losses['legal'])[~fit]
            numer=np.bincount(ix,weights=delta*m,minlength=len(ug));denom=np.bincount(ix,weights=m,minlength=len(ug))
            boot=(weights@numer)/(weights@denom).clip(1)
            result['paired_delta_vs_legal'][label]=dict(delta=float(delta[m].mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    report=dict(stage='Reused 2048-position development pilot; parameters fit on fold0, reported on game-disjoint fold1. Not golden macro.',
        positions=int((~fit).sum()),expert_positions=int((~fit&expert).sum()),calibration=calibration,
        calibration_reference='released_fixed only; same output parameters for every allocation variant',
        results=results,cost={m:dict(seconds=sum(x['end_to_end_seconds'] for x in c),leaves=sum(x['evaluated_leaves'] for x in c),simulations=int(trees[m]['simulations'].sum())) for m,c in cost.items()},
        cm_note='Pending: golden scaling laws do not apply to these development metrics.',plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest())
    dest=out/'results.json';tmp=dest.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(dest)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
