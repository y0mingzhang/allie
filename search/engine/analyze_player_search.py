"""Bayesian player-specific search strength from strictly previous choices."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from .service import ROOT,atomic
from .balanced_eval import digest
from .retrieval_common import data
from .fit_policy import fit
from .analyze_august import means

FACTORS=np.array([.25,.5,1.,2.,4.])
WIDTHS=[.25,.5,1.]
POWERS=[.25,1.]

def posterior(loglik,sigma,power):
    logprior=-.5*(np.log(FACTORS)/sigma)**2
    return softmax(logprior[None,:]+power*loglik,axis=1)

def mix(base,q,beta,mask,weights):
    # Factor 1 equals the calibrated search control exactly.
    logp=np.log(np.maximum(base,1e-300))
    variants=softmax(np.where(mask[None,:,:],logp[None,:,:]+
        (FACTORS[:,None,None]-1)*beta[None,:,None]*q[None,:,:],-np.inf),axis=2)
    return np.einsum('nk,knm->nm',weights,variants)

def test():
    rng=np.random.default_rng(266);p=rng.dirichlet(np.ones(7),size=20);q=rng.normal(size=p.shape);m=np.ones_like(p,bool)
    w=np.zeros((20,5));w[:,2]=1
    np.testing.assert_allclose(mix(p,q,np.ones(20),m,w),p,atol=1e-15)
    neutral=posterior(np.zeros((20,5)),.5,1)
    np.testing.assert_allclose(neutral,posterior(rng.normal(size=(20,5)),.5,0),atol=1e-15)
    ll=np.tile(np.arange(5),(20,1));post=posterior(ll,.5,1)
    assert ((post@FACTORS)>(neutral@FACTORS)).all()
    pp=mix(p,q,np.ones(20),m,post);np.testing.assert_allclose(pp.sum(1),1,atol=1e-15);assert (pp>0).all()
    print('PASS latent mixture fixed-control limit, no-evidence/prior limit, posterior direction, normalization')

def main():
    test();start=time.monotonic();out=ROOT/'aug-player-search-v1';plan=json.loads((out/'plan.json').read_text());worker=json.loads((out/'worker.json').read_text())
    assert worker['plan_sha256']==digest(out/'plan.json')
    assert worker['index_sha256']==digest(out/'history-index.npy')
    d=data();rows=d['rows'];cells=d['cells'];games=d['games'];fm=d['fit'];cv=d['cv'];target=d['target'];n=len(target);ar=np.arange(n);group=cells%4
    indices=np.load(out/'history-index.npy');valid=indices>=0;nh=worker['unique_past_positions']
    saved=[]
    for path in sorted(out.glob('history-*.npz')):
        with np.load(path) as z:saved.append({k:z[k] for k in ('z','q','mask','target','nodes','prefix_lengths')})
    kh=max(x['z'].shape[1] for x in saved);hz=np.zeros((nh,kh));hq=np.zeros_like(hz);hm=np.zeros((nh,kh),bool);ht=np.zeros(nh,int);hn=np.zeros(nh,int);hl=np.zeros(nh,int)
    offset=0
    for x in saved:
        end=offset+len(x['target']);kk=x['z'].shape[1];hz[offset:end,:kk]=x['z'];hq[offset:end,:kk]=x['q'];hm[offset:end,:kk]=x['mask'];ht[offset:end]=x['target'];hn[offset:end]=x['nodes'];hl[offset:end]=x['prefix_lengths'];offset=end
    assert offset==nh
    # Re-read the exact current-position Q used by retrieval_common's controls.
    cq=np.zeros_like(d['mask'],float)
    cp=json.loads((ROOT/'aug-adaptive-root-v1/plan.json').read_text())
    for lo in range(0,n,cp['roots_per_batch']):
        with np.load(ROOT/'aug-adaptive-root-v1/quota_static'/f'{lo:06d}.npz') as z:
            hi=lo+len(z['game']);cq[lo:hi,:z['ids'].shape[1]]=z['q'][cp['budgets'].index(1000)]
    # Charge each query for its own historical roots and children, even if
    # preparation physically shared them between queries. Tokens separately.
    child_cost=np.where(valid,hn[np.maximum(indices,0)],0).sum(1)
    root_cost=valid.sum(1)
    prefills=np.where(valid,hl[np.maximum(indices,0)],0).sum(1)
    versions=[]
    for fold,train in [(None,fm)]+[(f,fm&(cv!=f)) for f in range(3)]:
        hparams={};evidence=np.zeros((n,len(FACTORS)))
        for g in range(4):
            tr=train&(group==g);counts=np.bincount(cells[tr],minlength=16);hweight=np.zeros(nh)
            for i in np.flatnonzero(tr):
                use=indices[i,valid[i]]
                if len(use):np.add.at(hweight,use,1/(counts[cells[i]]*len(use)))
            active=hweight>0
            fp=fit(hz[active],hq[active],hm[active],ht[active],'forward',hweight[active]);assert fp['converged'];hparams[str(g)]=fp
            all_logits=fp['alpha']*hz[None,:,:]+(FACTORS*fp['beta'])[:,None,None]*hq[None,:,:]
            all_logp=np.where(hm[None,:,:],all_logits,-np.inf)
            ll=(all_logp[:,np.arange(nh),ht]-logsumexp(all_logp,axis=2)).T
            for i in np.flatnonzero(group==g):
                use=indices[i,valid[i]]
                if len(use):evidence[i]=ll[use].sum(0)
        versions.append((evidence,hparams))
    records={};losses={};baseversions=d['controls']['search']
    def predict(v,sigma,power):
        p,params=baseversions[v];beta=np.array([params[str(g)]['beta'] for g in group])
        if sigma==0:return p
        w=posterior(versions[v][0],sigma,power)
        return mix(p,cq,beta,d['mask'],w)
    specs=[('search',0.,0.)]+[(f'prior_s{s}',s,0.) for s in WIDTHS]+[(f'adapt_s{s}_e{e}',s,e) for s in WIDTHS for e in POWERS]
    for name,sigma,power in specs:
        p=predict(0,sigma,power);loss=-np.log(p[ar,target]);oof=np.full(n,np.nan)
        for f in range(3):
            val=fm&(cv==f);pp=predict(f+1,sigma,power);oof[val]=-np.log(pp[ar[val],target[val]])
        conf=means(loss[~fm],cells[~fm]);cc=means(oof[fm],cells[fm]);extra=power>0
        rec=dict(prior_sigma=sigma,evidence_power=power,training_equivalent_cm=None,
            mean_nodes=float(means(d['cost']+(child_cost+root_cost if extra else 0),cells).mean()),
            extra_past_children=float(child_cost.mean()) if extra else 0.,
            extra_past_roots=float(root_cost.mean()) if extra else 0.,
            extra_prefill_tokens=float(prefills.mean()) if extra else 0.,
            fit_game_cv=dict(macro_ce=float(cc.mean()),expert_ce=float(cc[3::4].mean())),
            confirmation=dict(macro_ce=float(conf.mean()),expert_ce=float(conf[3::4].mean()),cells=conf.tolist()))
        records[name]=rec;losses[name]=loss
    selected={metric:min(records,key=lambda name:records[name]['fit_game_cv'][metric]) for metric in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(98318).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count;assert (den>0).all()
    for name,rec in records.items():
        refs=['search']+([f"prior_s{rec['prior_sigma']}"] if rec['evidence_power'] else [])
        for ref in refs:
            delta=losses[name]-losses[ref];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
            rec['delta_vs_'+ref]=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
                macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    report=dict(stage='August game-disjoint confirmation; potentially training-seen. Golden CM pending.',results=records,fit_cv_selected=selected,
        parameters=dict(root=baseversions[0][1],history=versions[0][1],factors=FACTORS.tolist()),
        analysis_seconds=time.monotonic()-start,worker=worker,plan_sha256=digest(out/'plan.json'),source_sha256=digest(Path(__file__)),
        caveat='Current root prior already sees the game. Added signal is how one-ply policy/value reweighting agrees with strictly earlier same-player choices; no current target enters posterior. Static mixtures have no history-query cost. History calibration excludes confirmation games and is refit inside fit-game CV.')
    atomic(out/'results.json',report)
    with (out/'scores.npz').open('wb') as f:np.savez_compressed(f,names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    for name,r in records.items():print(name,r['confirmation']['macro_ce'],r['confirmation']['expert_ce'],r['mean_nodes'],flush=True)
    print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
