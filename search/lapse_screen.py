"""A deliberate no-search/lapse component in the existing human-policy mixture.

CPU only. No extra model queries, external data, or neural-weight changes.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
    q=p.with_suffix('.partial');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def means(x,c):return np.array([x[c==i].mean() for i in range(16)])


def main():
    start=time.monotonic();out=ROOT/'aug-lapse-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-budget-surface-v2';sp=source/'policies.npz';meta=source/'results.json'
    with np.load(sp) as f:
        parents=f['policy'][-1].astype(float);root=f['root'];q=f['q'][-1]
        mask,ids,target,cells,games,fm,cv,nodes=(f[k] for k in ('mask','ids','target','cells','games','fit','cv','nodes'))
        nodes=nodes[-1]
    n,k=mask.shape;ar=np.arange(n)
    assert not set(games[fm])&set(games[~fm])
    logit=np.where(mask,root[:,378:2346][ar[:,None],ids],0.)
    parameters=json.loads(meta.read_text())['fold_parameters']['1000']
    menus=[('parent',None,False)]+[(kind+('_elo' if by else ''),kind,by) for kind in ('no_deliberation','uniform_lapse','anti_value') for by in (False,True)]
    plan=dict(sources=dict(script=digest(Path(__file__)),surface=digest(sp),calibration=digest(meta)),menus=menus,
        formula='pi_new=(1-epsilon)*parent+epsilon*component. Components: calibrated root prior (beta0); uniform over legal actions; same calibrated prior with reversed value tilt (-beta). Epsilon in[0,1], shared or four Elo groups. All retain legal support and normalization.',
        validation='Parent and component calibration use their own matching training-game folds. Fit only mixture coefficients within each fold; select on3fold game CV. Report all arms on reused, disjoint August development games. No target is used to select a component at inference.',
        hypothesis='Current latent-strength mixture includes only positive search strengths [.25,.5,1,2,4]. A distinct non-deliberative or mistake component may better preserve human-error tails.',
        cost='No extra neural queries; same current1000 trees. Diagnostic oracle chooses components with the target and is not a deployable method or quality claim. No golden CM.')
    plan=json.loads(json.dumps(plan));pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    components=[]
    for vi,entry in enumerate(parameters):
        alpha=np.array([entry['root'][str(g)]['alpha'] for g in cells%4])
        beta=np.array([entry['root'][str(g)]['beta'] for g in cells%4])
        components.append(dict(no_deliberation=softmax(np.where(mask,alpha[:,None]*logit,-np.inf),axis=1),
            uniform_lapse=mask/mask.sum(1,keepdims=True),
            anti_value=softmax(np.where(mask,alpha[:,None]*logit-beta[:,None]*q,-np.inf),axis=1)))
    records,losses={},{}
    for name,kind,by in menus:
        group=cells%4 if by else np.zeros(n,int);oof=np.full(n,np.nan);fitted=[]
        for vi,(fold,train) in enumerate([(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]):
            p=np.where(mask,np.maximum(parents[vi],1e-300),0.);p/=p.sum(1,keepdims=True)
            c=p if kind is None else components[vi][kind];pt=p[ar,target];ct=c[ar,target]
            pred=p.copy();coeff={}
            for g in np.unique(group):
                tr=train&(group==g);take=group==g
                counts=np.bincount(cells[tr],minlength=16);w=1/counts[cells[tr]];w/=w.sum()
                objective=lambda eps:float(-w@np.log((1-eps)*pt[tr]+eps*ct[tr]))
                eps=0.
                if kind is not None:
                    opt=minimize_scalar(objective,bounds=(0,1),method='bounded',options=dict(xatol=1e-10));assert opt.success
                    eps=min((0.,float(opt.x),1.),key=objective)
                pred[take]=(1-eps)*p[take]+eps*c[take];coeff[str(int(g))]=eps
            np.testing.assert_allclose(pred.sum(1),1,atol=1e-13);assert (pred[mask]>0).all() and (pred[~mask]==0).all()
            ll=-np.log(pred[ar,target]);fitted.append(dict(fold=fold,epsilon=coeff))
            if fold is None:losses[name]=ll
            else:oof[fm&(cv==fold)]=ll[fm&(cv==fold)]
        a,b=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        records[name]=dict(parameters=fitted,training_equivalent_cm=None,mean_nodes=float(nodes.mean()),
            confirmation=dict(macro_ce=float(a.mean()),expert_ce=float(a[3::4].mean()),cells=a.tolist()),
            fit_game_cv=dict(macro_ce=float(b.mean()),expert_ce=float(b[3::4].mean())))
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('macro_ce','expert_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1
    count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    draws=np.random.default_rng(67462).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=draws@count
    for name,rec in records.items():
        delta=losses[name]-losses['parent'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16))
        np.add.at(sums,(ix,cells[~fm]),delta[~fm]);boot=draws@sums/den
        rec['delta_vs_parent']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(boot[:,3::4].mean(1),[.025,.975]).tolist())
    # Fit games only: an unattainable label-informed diagnostic, never a policy.
    pv=parents[cv+1,ar];pv/=pv.sum(1,keepdims=True)
    probs=[pv[ar,target]]
    for kind in components[0]:
        cp=np.stack([x[kind][ar,target] for x in components]);probs.append(cp[cv+1,ar])
    delta=-np.log(np.max(probs,axis=0))+np.log(probs[0]);oc=means(delta[fm],cells[fm])
    result=dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp),
        fit_only_label_informed_oracle=dict(macro_delta=float(oc.mean()),expert_delta=float(oc[3::4].mean()),deployable=False))
    atomic(out/'results.json',result)
    np.savez_compressed(out/'scores.npz',names=list(losses),loss=np.stack(list(losses.values())),cells=cells,games=games,fit=fm)
    print(json.dumps({name:dict(ce=x['confirmation']['macro_ce'],expert=x['confirmation']['expert_ce'],epsilon=x['parameters'][0]['epsilon'],cv=x['fit_game_cv']) for name,x in records.items()},indent=2))
    print('SELECTED',selected,'seconds',result['seconds'],'ORACLE',result['fit_only_label_informed_oracle'])


if __name__=='__main__':main()
