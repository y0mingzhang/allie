"""Calibrate predicted expected game score using August game-result labels only."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax,expit
from scipy.optimize import minimize
from .service import ROOT,atomic
from .balanced_eval import digest
from .sample_data import read
from .analyze_august import means


def objective(theta,logit,y,w):
    z=theta[0]*logit+theta[1]
    loss=np.logaddexp(0.,z)-y*z
    residual=expit(z)-y
    return float(w@loss),np.array([w@(residual*logit),w@residual])


def main():
    start=time.monotonic();out=ROOT/'aug-value-calibration-v2';out.mkdir(exist_ok=True)
    d=read('aug-tune-expanded-v1')
    rows,cells,games,fm,cv=(d[k] for k in ('rows','cells','games','fit','cv'))
    source=ROOT/'aug-value-outcomes-v1';manifest=json.loads((source/'manifest.json').read_text())
    assert digest(source/'labels.npz')==manifest['labels_sha256']
    with np.load(source/'labels.npz') as f:
        np.testing.assert_array_equal(f['game'],games);np.testing.assert_array_equal(f['ply'],[r['ply'] for r in rows]);wdl=f['mover_wdl']
    y=np.array([1.,.5,0.])[wdl]
    surface=ROOT/'aug-budget-surface-v2'
    with np.load(surface/'policies.npz') as f:
        np.testing.assert_array_equal(f['games'],games);root=f['root']
    probs=softmax(root[:,2413:2416],axis=1);p=probs[:,0]+.5*probs[:,1];p=np.clip(p,1e-12,1-1e-12)
    logit=np.log(p)-np.log1p(-p)
    menus=['raw','global_temperature','global_affine','elo_affine']
    plan=dict(source_sha256=digest(Path(__file__)),sample_sha256=digest(ROOT/'aug-tune-expanded-v1/sample.json'),
        outcomes_sha256=digest(source/'labels.npz'),root_source_sha256=digest(surface/'policies.npz'),menus=menus,
        target='Expected game score: win1/draw0.5/loss0 from next-mover perspective. Binary cross-entropy is strictly proper for this conditional mean; it is NOT move CE and is NOT three-class WDL CE.',
        formula='p=(V+1)/2; recalibrated p=sigmoid(alpha*logit(p)+bias). Alpha positive. Scalar-only mapping can be applied to cached tree critics; terminals must remain exact.',
        validation='Macro-weighted fits and3 game-fold CV, separate August confirmation; outcome labels never become inference inputs. Potentially model-training-seen. No golden scores or neural-weight updates.')
    pp=out/'plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:atomic(pp,plan)
    records,losses,predictions={}, {}, {}
    for name in menus:
        oof=np.full(len(y),np.nan);params=[]
        for fold,train in [(None,fm),*[(f,fm&(cv!=f)) for f in range(3)]]:
            result=np.zeros(len(y));result_logit=np.zeros(len(y));parameters={};groups=cells%4 if name=='elo_affine' else np.zeros(len(y),int)
            for g in np.unique(groups):
                tr=train&(groups==g);take=groups==g
                count=np.bincount(cells[tr],minlength=16);w=1/count[cells[tr]];w/=w.sum()
                theta=np.array([1.,0.])
                if name!='raw':
                    bounds=[(.2,5.),(0.,0.) if name=='global_temperature' else (-3.,3.)]
                    opt=minimize(lambda t:objective(t,logit[tr],y[tr],w),theta,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-12,gtol=1e-8))
                    assert opt.success,opt;theta=opt.x
                result_logit[take]=theta[0]*logit[take]+theta[1];result[take]=expit(result_logit[take]);parameters[str(g)]=theta.tolist()
            params.append(dict(fold=fold,groups=parameters))
            z=result_logit;loss=np.logaddexp(0.,z)-y*z;assert np.isfinite(loss).all()
            if fold is None:losses[name]=loss;predictions[name]=result
            else:oof[fm&(cv==fold)]=loss[fm&(cv==fold)]
        ce,cc=means(losses[name][~fm],cells[~fm]),means(oof[fm],cells[fm])
        squared=means((predictions[name][~fm]-y[~fm])**2,cells[~fm])
        records[name]=dict(parameters=params,confirmation=dict(score_ce=float(ce.mean()),expert_score_ce=float(ce[3::4].mean()),
            squared_score_error=float(squared.mean()),expert_squared_score_error=float(squared[3::4].mean()),cells=ce.tolist()),
            fit_game_cv=dict(score_ce=float(cc.mean()),expert_score_ce=float(cc[3::4].mean())),training_equivalent_cm=None)
        print(name,records[name]['confirmation'],params[0],flush=True)
    selected={m:min(records,key=lambda k:records[k]['fit_game_cv'][m]) for m in ('score_ce','expert_score_ce')}
    _,ix=np.unique(games[~fm],return_inverse=True);ng=ix.max()+1;count=np.zeros((ng,16));np.add.at(count,(ix,cells[~fm]),1)
    w=np.random.default_rng(18443).multinomial(ng,np.full(ng,1/ng),size=2000).astype(float);den=w@count
    for name,rec in records.items():
        delta=losses[name]-losses['raw'];point=means(delta[~fm],cells[~fm]);sums=np.zeros((ng,16));np.add.at(sums,(ix,cells[~fm]),delta[~fm]);draw=w@sums/den
        rec['delta_vs_raw']=dict(macro=float(point.mean()),expert=float(point[3::4].mean()),
            macro_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(draw[:,3::4].mean(1),[.025,.975]).tolist())
    reliability=[]
    for lo in np.arange(0,1,.1):
        take=(p>=lo)&(p<lo+.1)&~fm
        if take.any():reliability.append(dict(lower=float(lo),n=int(take.sum()),prediction=float(p[take].mean()),observed_score=float(y[take].mean())))
    atomic(out/'results.json',dict(results=records,fit_cv_selected=selected,seconds=time.monotonic()-start,plan_sha256=digest(pp),
        raw_value_quantiles=np.quantile(2*p-1,[0,.01,.1,.5,.9,.99,1]).tolist(),raw_reliability=reliability))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
