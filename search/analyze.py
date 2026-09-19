"""Select on fit games, report once on independent development confirmation games."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,softmax

ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'

def metrics(nll,pred,rows):
    cell=np.array([p['cell'] for p in rows]); target=np.array([p['target']-378 for p in rows])
    ces=[float(nll[cell==c].mean()) if np.any(cell==c) else None for c in range(16)]
    acc=[float((pred[cell==c]==target[cell==c]).mean()) if np.any(cell==c) else None for c in range(16)]
    return dict(dev_ce=float(nll.mean()),expert_dev_ce=float(nll[cell%4==3].mean()),
        dev_accuracy=float((pred==target).mean()),expert_dev_accuracy=float((pred[cell%4==3]==target[cell%4==3]).mean()),
        cell_ce=ces,cell_accuracy=acc,counts=np.bincount(cell,minlength=16).tolist())

def main():
    data=json.loads((ROOT/'dev.json').read_text()); rows=data['positions']
    arrays=[]
    for lo in range(0,len(rows),128):
        with np.load(ROOT/f'cache-{lo:05d}.npz') as z: arrays.append({k:z[k] for k in z.files})
    raw=np.concatenate([z['root'] for z in arrays]).astype(np.float64)
    candidates=np.concatenate([z['candidates'] for z in arrays])
    child=np.concatenate([z['child_wdl'] for z in arrays]); terminal=np.concatenate([z['terminal'] for z in arrays])
    rootvalue=softmax(raw[:,2413:2416],axis=1)@np.array([1.,.5,0.])
    childvalue=1-softmax(child.astype(np.float64),axis=2)@np.array([1.,.5,0.])
    childvalue=np.where(np.isfinite(terminal),terminal,childvalue)
    legal=np.zeros((len(rows),1968),bool); advantage=np.zeros_like(legal,dtype=np.float64)
    for i,p in enumerate(rows):
        legal[i,np.array(p['legal'])-378]=True
        valid=candidates[i]>=378
        assert np.isfinite(childvalue[i,valid]).all()
        advantage[i,candidates[i,valid]-378]=childvalue[i,valid]-rootvalue[i]
    target=np.array([p['target']-378 for p in rows]); ar=np.arange(len(rows))
    assert legal[ar,target].all()
    cells=np.array([p['cell'] for p in rows]); folds=np.array([p['fold'] for p in rows])
    time_gate=softmax(raw[:,2350:2413],axis=1)[:,6:].sum(1)
    logits=raw[:,378:2346]
    def score(params):
        enabled=(cells%4==3) if params.get('expert_only',False) else np.ones(len(rows),bool)
        x=logits/np.where(enabled,params['temperature'],1.)[:,None]
        gate=time_gate if params.get('gate')=='time' else np.ones(len(rows))
        x=x+params.get('beta',0)*(gate*enabled)[:,None]*advantage
        if params.get('legal',False): x=np.where(legal | ~enabled[:,None],x,-np.inf)
        nll=logsumexp(x,axis=1)-x[ar,target]
        return nll,x.argmax(1)
    def objective(nll):
        # Expert primary, overall regression penalty; entirely fit-fold selected.
        return float(nll[(folds==0)&(cells%4==3)].mean()+max(0,nll[folds==0].mean()-basefit))
    fit=folds==0; confirm=~fit
    base_nll,base_pred=score(dict(temperature=1.,legal=False))
    basefit=float(base_nll[fit].mean())
    trials=[]
    for t in (.8,.9,1.,1.1,1.2):
        for beta in (-2.,-1.,0.,.5,1.,2.,4.):
            for gate in (('none',) if beta==0 else ('none','time')):
                for expert_only in (False,True):
                    p=dict(temperature=t,beta=beta,gate=gate,legal=True,expert_only=expert_only)
                    nll,_=score(p);trials.append((objective(nll),p))
    selected=min(trials,key=lambda z:z[0])[1]
    calibration=min((x for x in trials if x[1]['beta']==0),key=lambda z:z[0])[1]
    frozen=dict(selected=selected,calibration=calibration,
        selection_fold=0,confirmation_fold=1,data_sha256=hashlib.sha256((ROOT/'dev.json').read_bytes()).hexdigest(),
        objective='expert dev CE + max(0, dev CE minus raw baseline dev CE), fit fold only')
    choice=ROOT/'selected.json'
    if choice.exists(): assert json.loads(choice.read_text())==frozen,'Do not retune against confirmation'
    else: choice.write_text(json.dumps(frozen,indent=2)+'\n')
    outputs={}; subset=[p for p in rows if p['fold']==1]
    specs=dict(raw=dict(temperature=1.,legal=False),legal=dict(temperature=1.,legal=True),
        calibrated=calibration,shallow=selected)
    saved={}
    for name,params in specs.items():
        nll,pred=score(params); saved[name]=nll
        outputs[name]=metrics(nll[confirm],pred[confirm],subset)
    # Paired game bootstrap, preserving within-game dependence across sides.
    games=np.array([p['game'] for p in rows]); uniq,inv=np.unique(games[confirm],return_inverse=True)
    cc=cells[confirm]; rng=np.random.default_rng(82911); intervals={}
    for baseline in ('raw','calibrated'):
        delta=(saved['shallow']-saved[baseline])[confirm]; samples=[]
        for _ in range(1000):
            w=rng.poisson(1,len(uniq))[inv]
            expert=cc%4==3
            samples.append([np.sum(w*delta)/max(1,w.sum()),np.sum(w[expert]*delta[expert])/max(1,w[expert].sum())])
        intervals[baseline]=np.quantile(samples,[.025,.975],axis=0).tolist()
    report=dict(stage='independent development confirmation, not golden eval',methods=outputs,
        selected=selected,calibration=calibration,paired_delta_ce_95pct=intervals,
        interval_layout='rows lower/upper; columns dev/expert-dev; negative is improvement',
        candidate_target_coverage=float(np.any(candidates==target[:,None]+378,axis=1).mean()),
        limitation='Small subset of the existing development splits, mostly blitz. No final claim of golden improvement.')
    (ROOT/'pilot-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
