"""Use pre-move clocks to calibrate direct/search policies; no future think time."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .engine.service import ROOT,atomic
from .engine.balanced_eval import digest
from .engine.retrieval_common import data
from .engine.analyze_august import means

ARMS=['trouble_beta','trouble_both','seconds_both','seconds_format_beta']
def features(seconds,cells,kind):
    known=seconds>=0
    x=((seconds<=15)&known).astype(float) if kind.startswith('trouble') else np.where(known,np.clip(np.log1p(np.maximum(seconds,0))/np.log(61)-1,-1,1),0.)
    if kind=='seconds_format_beta':return x[:,None]*(cells[:,None]//4==np.arange(4)),False
    return x[:,None],kind.endswith('both')

def evaluate(theta,z,q,mask,target,alpha,beta,x,both,weights=None,policy=False):
    m=x.shape[1]
    wa=theta[:m] if both else np.zeros(m);wb=theta[m:] if both else theta
    au=alpha*np.exp(x@wa);bu=beta*np.exp(x@wb)
    a=np.clip(au,.4,1.6);b=np.clip(bu,0,40)
    da=np.where((au>.4)&(au<1.6),au,0);db=np.where((bu>0)&(bu<40),bu,0)
    logits=np.where(mask,a[:,None]*z+b[:,None]*q,-np.inf);lp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(lp)
    ar=np.arange(len(p));loss=-lp[ar,target]
    if policy:return p,loss
    w=np.ones(len(z))/len(z) if weights is None else weights/weights.sum()
    ga=x.T@(w*da*((p*z).sum(1)-z[ar,target]))
    gb=x.T@(w*db*((p*q).sum(1)-q[ar,target]))
    return float(w@loss),np.r_[ga,gb] if both else gb

def test():
    rng=np.random.default_rng(1210);n=23;z=rng.normal(size=(n,7));q=rng.normal(size=(n,7));mask=np.ones_like(z,bool);target=np.zeros(n,int)
    alpha=np.ones(n);beta=np.full(n,2.);x=rng.uniform(-1,1,size=(n,2))
    for both in (False,True):
        theta=rng.normal(0,.1,size=2*(1+both));f,g=evaluate(theta,z,q,mask,target,alpha,beta,x,both)
        numeric=[]
        for j in range(len(theta)):
            a=theta.copy();b=theta.copy();a[j]+=1e-6;b[j]-=1e-6
            numeric.append((evaluate(a,z,q,mask,target,alpha,beta,x,both)[0]-evaluate(b,z,q,mask,target,alpha,beta,x,both)[0])/2e-6)
        np.testing.assert_allclose(g,numeric,atol=2e-8,rtol=2e-7)
    print('PASS clock calibration analytic gradient and bounded support')

def main():
    test();start=time.monotonic();out=ROOT/'aug-clock-calibration-v1';out.mkdir(exist_ok=True)
    d=data();rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];target=d['target'];mask=d['mask'];n=len(rows);ar=np.arange(n)
    src=ROOT/'aug-tune-v1';manifest=json.loads((src/'manifest.json').read_text())
    for name in ('strat.npz','clocks.npz','feats.npz'):assert digest(src/name)==manifest['files_sha256'][name]
    with np.load(src/'strat.npz') as z:tokens=z['rows'];labels=z['labels']
    with np.load(src/'feats.npz') as z:raw=z['feats']
    seconds=[]
    for r in rows:
        rr,cc=r['row'],r['column'];assert tokens[rr,cc]==r['target'] and labels[rr,cc]==r['cell']
        np.testing.assert_array_equal(tokens[rr,cc-len(r['prefix']):cc],r['prefix'])
        seconds.append(raw[rr,cc-1,0])
    seconds=np.array(seconds)
    plan=dict(sample_sha256=digest(src/'sample.json'),files_sha256=manifest['files_sha256'],arms=ARMS,
        source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).parent/'engine/retrieval_common.py']},
        input='Only mover clock BEFORE target move: feats[row,target_column-1,0]. This is previous clock reading of the same player, not time spent on target. Missing clocks leave coefficients unchanged.',
        fit='Known-clock functions and bounded multiplicative alpha/beta corrections. Elo-only base fit separately in each game-CV train fold, then correction fit there. Report direct+clock and search+clock, every arm, on August fold1. No golden until frozen.',
        caveat='Adds a legitimate pre-move clock input to a checkpoint that has no clock input; distinguish information gain from better search. No new NN calls.')
    plan_path=out/'plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:atomic(plan_path,plan)
    control=ROOT/'aug-adaptive-root-v1';cp=json.loads((control/'plan.json').read_text())
    logits=np.zeros_like(d['ids'],float);q=np.zeros_like(logits)
    for lo in range(0,n,cp['roots_per_batch']):
        with np.load(control/'quota_static'/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);kk=z['ids'].shape[1]
            logits[lo:hi]=z['z'][:,378:2346][np.arange(hi-lo)[:,None],d['ids'][lo:hi]]
            q[lo:hi,:kk]=z['q'][cp['budgets'].index(1000)]
    logits=np.where(mask,logits,0.);q=np.where(mask,q,0.)
    records={};losses={}
    def record(name,p,loss,oof,params,base,kind):
        conf=means(loss[~fm],cells[~fm]);cvce=means(oof[fm],cells[fm])
        records[name]=dict(parameters=params,base=base,kind=kind,training_equivalent_cm=None,
            mean_nodes=float(d['cost'].mean()) if base=='search' else 0.,
            fit_game_cv=dict(macro_ce=float(cvce.mean()),expert_ce=float(cvce[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        losses[name]=loss
    for base,versions in d['controls'].items():
        p,params=versions[0];oof=np.full(n,np.nan)
        for f in range(3):
            val=fm&(cv==f);pp=versions[f+1][0];oof[val]=-np.log(pp[ar[val],target[val]])
        record(base,p,-np.log(p[ar,target]),oof,dict(base=params),base,None)
        bq=q if base=='search' else np.zeros_like(q)
        for kind in ARMS:
            if base=='direct' and not kind.endswith('both'):continue
            x,both=features(seconds,cells,kind)
            def fit_predict(params,train):
                aa=np.array([params[str(c%4)]['alpha'] for c in cells]);bb=np.array([params[str(c%4)]['beta'] for c in cells])
                if base=='direct':bb[:]=0
                ct=np.bincount(cells[train],minlength=16);w=1/ct[cells[train]]
                obj=lambda t:evaluate(t,logits[train],bq[train],mask[train],target[train],aa[train],bb[train],x[train],both,w)
                initial=np.zeros(x.shape[1]*(1+both));res=minimize(obj,initial,jac=True,method='L-BFGS-B',bounds=[(-2,2)]*len(initial),options=dict(ftol=1e-11,gtol=1e-7,maxiter=300))
                assert res.success,res
                pp,ll=evaluate(res.x,logits,bq,mask,target,aa,bb,x,both,policy=True)
                return pp,ll,res.x.tolist()
            p,loss,t=fit_predict(versions[0][1],fm);oof=np.full(n,np.nan)
            for f in range(3):
                _,ll,_=fit_predict(versions[f+1][1],fm&(cv!=f));val=fm&(cv==f);oof[val]=ll[val]
            record(base+'_'+kind,p,loss,oof,dict(base=params,theta=t),base,kind)
    selected={metric:min(records,key=lambda x:records[x]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;ct=np.zeros((ng,16));np.add.at(ct,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@ct;assert (den>0).all()
    for name,rec in records.items():
        delta=losses[name]-losses[rec['base']];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['confirmation_delta_vs_base']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August pre-move clock calibration; no observed future and no golden CM.',results=records,fit_cv_selected=selected,
        missing_clock=int((seconds<0).sum()),time_trouble=int(((seconds>=0)&(seconds<=15)).sum()),seconds=time.monotonic()-start,plan_sha256=digest(plan_path))
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm,seconds=seconds)
    for name,x in records.items():print(name,x['confirmation']['macro_ce'],x['confirmation']['expert_ce'],x['parameters'].get('theta'),flush=True)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
