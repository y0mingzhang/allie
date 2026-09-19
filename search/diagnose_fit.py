"""Fit-fold, out-of-fold diagnostics: where does search still mispredict humans?"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'


def main():
    out=ROOT/'diagnosis-fit-v1';out.mkdir(exist_ok=True)
    source=ROOT/'aug-budget-surface-v2/policies.npz'
    with np.load(source) as f:
        fit=f['fit'];indices=np.flatnonzero(fit);cv=f['cv'][fit];ar=np.arange(len(indices))
        p=f['policy'][-1,cv+1,indices].astype(float);p/=p.sum(1,keepdims=True)
        root=f['root'][fit];ids=f['ids'][fit];mask=f['mask'][fit];target=f['target'][fit]
        cells=f['cells'][fit];games=f['games'][fit];q=f['q'][-1,fit]
    p=np.where(mask,np.maximum(p,1e-300),0.);p/=p.sum(1,keepdims=True)
    z=np.where(mask,root[:,378:2346][ar[:,None],ids],-np.inf);raw=softmax(z,axis=1)
    loss=-np.log(p[ar,target]);rawloss=-np.log(raw[ar,target]);entropy=-(p*np.log(np.maximum(p,1e-300))).sum(1)
    confidence=p.max(1);correct=p.argmax(1)==target
    value=softmax(root[:,2413:2416],axis=1)@np.array([1.,0.,-1.])
    rows=json.loads((ROOT/'aug-tune-expanded-v1/sample.json').read_text())['positions']
    ply=np.array([len(rows[i]['prefix'])-11 for i in indices])
    weight=1/np.bincount(cells,minlength=16)[cells]/16
    bins=[0.,1e-4,1e-3,.01,.05,.1,.2,.5,1.0000001]
    def summarize(take):
        w=weight[take];w=w/w.sum()
        return dict(positions=int(take.sum()),population_share=float(weight[take].sum()),
            raw_ce=float(w@rawloss[take]),search_ce=float(w@loss[take]),delta=float(w@(loss-rawloss)[take]),
            predicted_entropy=float(w@entropy[take]),confidence=float(w@confidence[take]),accuracy=float(w@correct[take]))
    groups={'all':np.ones(len(cells),bool),'expert':cells%4==3}
    for c in range(16):groups[f'cell{c}']=cells==c
    for name,take in [('early',ply<20),('middle',(ply>=20)&(ply<60)),('late',(ply>=60)&(ply<100)),('very_late',ply>=100),
                      ('losing',value<-.8),('unclear',np.abs(value)<=.8),('winning',value>.8)]:groups[name]=take
    reports={k:summarize(v) for k,v in groups.items() if v.any()}
    reliability={}
    for name,take in [('all',groups['all']),('expert',groups['expert'])]:
        w=weight[take];w/=w.sum();pp=p[take];pt=pp[np.arange(len(pp)),target[take]]
        reliability[name]=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            predicted=float(w@np.where((pp>=lo)&(pp<hi),pp,0).sum(1))
            observed=float(w@((pt>=lo)&(pt<hi)));ce=float(w@(np.where((pt>=lo)&(pt<hi),-np.log(pt),0)))
            reliability[name].append(dict(lo=lo,hi=hi,predicted_mass=predicted,observed_mass=observed,ce_contribution=ce))
    result=dict(scope='Only original August fitting games, with each prediction calibrated excluding its game-CV fold. No confirmation or golden diagnostics used to propose methods. Target-based bins describe errors and are not inference-time inputs.',
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        groups=reports,reliability=reliability)
    dest=out/'results.json'
    if dest.exists():assert json.loads(dest.read_text())==result
    else:dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in reports.items() if not k.startswith('cell')},indent=2))
    print('RELIABILITY',json.dumps(reliability,indent=2))


if __name__=='__main__':main()
