"""Does early search instability predict later value changes or CE gains?"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    source=ROOT/'aug-budget-surface-v2/policies.npz'
    with np.load(source) as f:
        ii=np.flatnonzero(f['fit']);cv=f['cv'][ii];c=f['cells'][ii];ar=np.arange(len(ii))
        policy=f['policy'][:,:,ii].astype(float);policy/=policy.sum(-1,keepdims=True)
        q=f['q'][:,ii];root=f['root'][ii];mask=f['mask'][ii];ids=f['ids'][ii];target=f['target'][ii]
    z=np.where(mask,root[:,378:2346][ar[:,None],ids],-np.inf);prior=softmax(z,axis=1)
    early=q[1]-q[0];late=q[3]-q[2]
    early_var=(prior*early**2).sum(1);late_var=(prior*late**2).sum(1)
    curvature=prior*(1-prior)
    movement=np.column_stack([np.log(1e-6+early_var),np.log(1e-6+(curvature*early**2).sum(1)),(prior*early).sum(1)])
    rows=json.loads((ROOT/'aug-tune-expanded-v1/sample.json').read_text())['positions']
    ply=np.array([len(rows[i]['prefix'])-11 for i in ii]);entropy=-(prior*np.log(np.maximum(prior,1e-300))).sum(1)
    qmean=(prior*q[1]).sum(1);spread=np.sqrt((prior*(q[1]-qmean[:,None])**2).sum(1))
    categorical=np.c_[np.eye(4)[c%4,1:],np.eye(4)[c//4,1:]]
    state=np.c_[entropy,np.log(.01+spread),np.log1p(ply)]
    records={}
    for outcome in ('late_value_change','late_ce_gain'):
        prediction={key:np.zeros(len(ii)) for key in ('state','state_plus_instability')};truth=np.zeros(len(ii))
        for fold in range(3):
            tr=cv!=fold;val=cv==fold
            # Each endpoint's policy is calibrated excluding this evaluation fold.
            prob=policy[:,fold+1,ar,target]
            y=np.log(1e-6+late_var) if outcome=='late_value_change' else np.log(prob[-1])-np.log(prob[-2])
            truth[val]=y[val]
            for name,features in [('state',state),('state_plus_instability',np.c_[state,movement])]:
                mean=features[tr].mean(0);scale=np.maximum(features[tr].std(0),1e-6)
                x=np.c_[np.ones(len(ii)),categorical,np.clip((features-mean)/scale,-3,3)]
                weight=1/np.bincount(c[tr],minlength=16)[c[tr]];weight/=weight.sum()
                penalty=np.eye(x.shape[1])*1e-3;penalty[0,0]=0
                coef=np.linalg.solve(x[tr].T@(weight[:,None]*x[tr])+penalty,x[tr].T@(weight*y[tr]))
                prediction[name][val]=x[val]@coef
        records[outcome]={}
        for name,pred in prediction.items():
            records[outcome][name]={}
            for pop,take in [('all',np.ones(len(ii),bool)),('expert',c%4==3)]:
                error=np.mean((pred[take]-truth[take])**2);variance=np.var(truth[take])
                records[outcome][name][pop]=dict(mse=float(error),r2=float(1-error/variance),
                    prediction_correlation=float(np.corrcoef(pred[take],truth[take])[0,1]))
    result=dict(scope='Original August fitting games only. Three held-out game folds for each diagnostic. No confirmation/golden method selection. No additional NN queries.',
        early_to_late_spearman=float(spearmanr(early_var,late_var).statistic),results=records,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        interpretation='Later bootstrap changes are a proxy for estimator instability, not ground-truth critic error. A gain in that proxy need not imply better human prediction. CE-gain target checks that distinction. This is not a deployed allocation rule or CM claim.')
    out=ROOT/'diagnosis-fit-v1/stability.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
